// DR. ROBOTNIK'S RING RACERS
//-----------------------------------------------------------------------------
// Copyright (C) 2025 by Kart Krew
//
// This program is free software distributed under the
// terms of the GNU General Public License, version 2.
// See the 'LICENSE' file for more details.
//-----------------------------------------------------------------------------

// Metal backend for the RHI.
//
// The RHI is modeled after immediate-mode GL, so this backend emulates GL2 semantics on top of Metal:
//
// - Render, depth-stencil and sampler states are built lazily from the current RHI state at draw time and cached.
// - Program uniforms are stored per program (as in GL) and pushed with setVertexBytes/setFragmentBytes per draw.
// - Offscreen render targets are rendered with a flipped Y axis, so their rows are stored bottom-up like GL.
//   Sampling them with GL texture coordinates, and reading them back, then behaves exactly like the GL backend.
//   The drawable is rendered unflipped, so it displays upright.
// - Viewport and scissor rects use GL's bottom-left origin and are converted for the drawable.
// - Resources are written by the CPU in submission order: buffers in use by the GPU are renamed, and textures in use
//   by the GPU are updated through a blit in the command stream.

#include "metal_rhi.hpp"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <dispatch/dispatch.h>
#include <fmt/format.h>
#include <glm/gtc/type_ptr.hpp>

#include "../../cxxutil.hpp"
#include "metal_shaders.hpp"

// Everything lives in srb2::rhi so that Handle and Rect don't collide with the global MacTypes ones.
namespace srb2::rhi
{

namespace
{

constexpr uint32_t kMaxFramesInFlight = 3;
constexpr uint32_t kMaxVertexAttribs = 5;
constexpr uint32_t kVertexBufferBaseIndex = 1; // buffer(0) holds the uniforms
constexpr MTLPixelFormat kDepthStencilFormat = MTLPixelFormatDepth32Float_Stencil8;
constexpr MTLPixelFormat kDrawableFormat = MTLPixelFormatBGRA8Unorm;

// Must match the layout of the Uniforms struct in metal_shaders.cpp.
constexpr size_t kUniformsSize = 240;
constexpr size_t kUniformFlipYOffset = 224;

enum class UniformKind
{
	kFloat,
	kFloat2,
	kInt,
	kMat3,
	kMat4
};

struct UniformSlot
{
	const char* name;
	size_t offset;
	UniformKind kind;
};

constexpr UniformSlot kUniformSlots[] = {
	{"u_projection", 0, UniformKind::kMat4},
	{"u_modelview", 64, UniformKind::kMat4},
	{"u_texcoord0_transform", 128, UniformKind::kMat3},
	{"u_texcoord0_min", 176, UniformKind::kFloat2},
	{"u_texcoord0_max", 184, UniformKind::kFloat2},
	{"u_sampler0_size", 192, UniformKind::kFloat2},
	{"u_time", 200, UniformKind::kFloat},
	{"u_sampler0_is_indexed_alpha", 204, UniformKind::kInt},
	{"u_wipe_colorize_mode", 208, UniformKind::kInt},
	{"u_wipe_encore_swizzle", 212, UniformKind::kInt},
	{"u_postimg_water", 216, UniformKind::kInt},
	{"u_postimg_heat", 220, UniformKind::kInt},
};

const UniformSlot* find_uniform_slot(const char* name)
{
	for (const UniformSlot& slot : kUniformSlots)
	{
		if (std::strcmp(slot.name, name) == 0)
		{
			return &slot;
		}
	}
	return nullptr;
}

int vertex_attribute_index(const char* name)
{
	static constexpr const char* kNames[kMaxVertexAttribs] = {
		"a_position",
		"a_texcoord0",
		"a_color",
		"a_normal",
		"a_texcoord1"
	};
	for (uint32_t i = 0; i < kMaxVertexAttribs; i++)
	{
		if (std::strcmp(kNames[i], name) == 0)
		{
			return static_cast<int>(i);
		}
	}
	return -1;
}

int sampler_index(const char* name)
{
	static constexpr const char* kNames[kMaxSamplers] = {"s_sampler0", "s_sampler1", "s_sampler2", "s_sampler3"};
	for (uint32_t i = 0; i < kMaxSamplers; i++)
	{
		if (std::strcmp(kNames[i], name) == 0)
		{
			return static_cast<int>(i);
		}
	}
	return -1;
}

struct ProgramEntry
{
	const char* name;
	const char* vertex_function;
	const char* fragment_function;
	const char* implicit_define; // defines the GLSL source hardcodes
};

constexpr ProgramEntry kPrograms[] = {
	{"unshaded", "generic_vs", "unshaded_fs", nullptr},
	{"unshadedpaletted", "generic_vs", "unshadedpaletted_fs", nullptr},
	{"sharpbilinear", "generic_vs", "sharpbilinear_fs", nullptr},
	{"crt", "generic_vs", "crt_fs", nullptr},
	{"crtsharp", "generic_vs", "crtsharp_fs", nullptr},
	{"postimg", "generic_vs", "postimg_fs", "ENABLE_VA_TEXCOORD0"},
	{"postprocesswipe", "postprocesswipe_vs", "postprocesswipe_fs", "ENABLE_VA_TEXCOORD0"},
};

MTLPixelFormat map_texture_format(TextureFormat format)
{
	switch (format)
	{
	case TextureFormat::kLuminance:
		return MTLPixelFormatR8Unorm;
	case TextureFormat::kLuminanceAlpha:
		return MTLPixelFormatRG8Unorm;
	case TextureFormat::kRGB:
	case TextureFormat::kRGBA:
	default:
		return MTLPixelFormatRGBA8Unorm;
	}
}

uint32_t texture_format_bpp(TextureFormat format)
{
	switch (format)
	{
	case TextureFormat::kLuminance:
		return 1;
	case TextureFormat::kLuminanceAlpha:
		return 2;
	default:
		return 4;
	}
}

uint32_t pixel_format_bpp(PixelFormat format)
{
	switch (format)
	{
	case PixelFormat::kR8:
		return 1;
	case PixelFormat::kRG8:
		return 2;
	case PixelFormat::kRGB8:
		return 3;
	case PixelFormat::kRGBA8:
		return 4;
	default:
		return 0;
	}
}

// Sampling swizzles reproducing GL's legacy luminance formats and RGB's opaque alpha.
MTLTextureSwizzleChannels texture_swizzle(TextureFormat format)
{
	switch (format)
	{
	case TextureFormat::kLuminance:
		return MTLTextureSwizzleChannelsMake(
			MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne
		);
	case TextureFormat::kLuminanceAlpha:
		return MTLTextureSwizzleChannelsMake(
			MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen
		);
	case TextureFormat::kRGB:
		return MTLTextureSwizzleChannelsMake(
			MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleOne
		);
	case TextureFormat::kRGBA:
	default:
		return MTLTextureSwizzleChannelsDefault;
	}
}

MTLSamplerAddressMode map_wrap(TextureWrapMode wrap)
{
	switch (wrap)
	{
	case TextureWrapMode::kClamp:
		return MTLSamplerAddressModeClampToEdge;
	case TextureWrapMode::kMirroredRepeat:
		return MTLSamplerAddressModeMirrorRepeat;
	case TextureWrapMode::kRepeat:
	default:
		return MTLSamplerAddressModeRepeat;
	}
}

MTLSamplerMinMagFilter map_filter(TextureFilterMode filter)
{
	return filter == TextureFilterMode::kLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}

MTLVertexFormat map_vertex_format(VertexAttributeFormat format)
{
	switch (format)
	{
	case VertexAttributeFormat::kFloat:
		return MTLVertexFormatFloat;
	case VertexAttributeFormat::kFloat2:
		return MTLVertexFormatFloat2;
	case VertexAttributeFormat::kFloat3:
		return MTLVertexFormatFloat3;
	case VertexAttributeFormat::kFloat4:
	default:
		return MTLVertexFormatFloat4;
	}
}

uint32_t vertex_format_size(VertexAttributeFormat format)
{
	switch (format)
	{
	case VertexAttributeFormat::kFloat:
		return 4;
	case VertexAttributeFormat::kFloat2:
		return 8;
	case VertexAttributeFormat::kFloat3:
		return 12;
	case VertexAttributeFormat::kFloat4:
	default:
		return 16;
	}
}

MTLCompareFunction map_compare_func(CompareFunc func)
{
	switch (func)
	{
	case CompareFunc::kNever:
		return MTLCompareFunctionNever;
	case CompareFunc::kLess:
		return MTLCompareFunctionLess;
	case CompareFunc::kEqual:
		return MTLCompareFunctionEqual;
	case CompareFunc::kLessEqual:
		return MTLCompareFunctionLessEqual;
	case CompareFunc::kGreater:
		return MTLCompareFunctionGreater;
	case CompareFunc::kNotEqual:
		return MTLCompareFunctionNotEqual;
	case CompareFunc::kGreaterEqual:
		return MTLCompareFunctionGreaterEqual;
	case CompareFunc::kAlways:
	default:
		return MTLCompareFunctionAlways;
	}
}

MTLStencilOperation map_stencil_op(StencilOp op)
{
	switch (op)
	{
	case StencilOp::kKeep:
		return MTLStencilOperationKeep;
	case StencilOp::kZero:
		return MTLStencilOperationZero;
	case StencilOp::kReplace:
		return MTLStencilOperationReplace;
	case StencilOp::kIncrementClamp:
		return MTLStencilOperationIncrementClamp;
	case StencilOp::kDecrementClamp:
		return MTLStencilOperationDecrementClamp;
	case StencilOp::kInvert:
		return MTLStencilOperationInvert;
	case StencilOp::kIncrementWrap:
		return MTLStencilOperationIncrementWrap;
	case StencilOp::kDecrementWrap:
		return MTLStencilOperationDecrementWrap;
	default:
		return MTLStencilOperationKeep;
	}
}

MTLBlendFactor map_blend_factor(BlendFactor factor)
{
	switch (factor)
	{
	case BlendFactor::kZero:
		return MTLBlendFactorZero;
	case BlendFactor::kOne:
		return MTLBlendFactorOne;
	case BlendFactor::kSource:
		return MTLBlendFactorSourceColor;
	case BlendFactor::kOneMinusSource:
		return MTLBlendFactorOneMinusSourceColor;
	case BlendFactor::kSourceAlpha:
		return MTLBlendFactorSourceAlpha;
	case BlendFactor::kOneMinusSourceAlpha:
		return MTLBlendFactorOneMinusSourceAlpha;
	case BlendFactor::kDest:
		return MTLBlendFactorDestinationColor;
	case BlendFactor::kOneMinusDest:
		return MTLBlendFactorOneMinusDestinationColor;
	case BlendFactor::kDestAlpha:
		return MTLBlendFactorDestinationAlpha;
	case BlendFactor::kOneMinusDestAlpha:
		return MTLBlendFactorOneMinusDestinationAlpha;
	case BlendFactor::kConstant:
		return MTLBlendFactorBlendColor;
	case BlendFactor::kOneMinusConstant:
		return MTLBlendFactorOneMinusBlendColor;
	case BlendFactor::kConstantAlpha:
		return MTLBlendFactorBlendAlpha;
	case BlendFactor::kOneMinusConstantAlpha:
		return MTLBlendFactorOneMinusBlendAlpha;
	case BlendFactor::kSourceAlphaSaturated:
		return MTLBlendFactorSourceAlphaSaturated;
	default:
		return MTLBlendFactorOne;
	}
}

MTLBlendOperation map_blend_function(BlendFunction function)
{
	switch (function)
	{
	case BlendFunction::kSubtract:
		return MTLBlendOperationSubtract;
	case BlendFunction::kReverseSubtract:
		return MTLBlendOperationReverseSubtract;
	case BlendFunction::kAdd:
	default:
		return MTLBlendOperationAdd;
	}
}

std::optional<MTLPrimitiveType> map_primitive(PrimitiveType type)
{
	switch (type)
	{
	case PrimitiveType::kPoints:
		return MTLPrimitiveTypePoint;
	case PrimitiveType::kLines:
		return MTLPrimitiveTypeLine;
	case PrimitiveType::kLineStrip:
		return MTLPrimitiveTypeLineStrip;
	case PrimitiveType::kTriangles:
		return MTLPrimitiveTypeTriangle;
	case PrimitiveType::kTriangleStrip:
		return MTLPrimitiveTypeTriangleStrip;
	default:
		// Metal has no triangle fans. Nothing in the game uses them.
		return std::nullopt;
	}
}

template <typename T>
std::string_view key_bytes(const T& key)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return std::string_view(reinterpret_cast<const char*>(&key), sizeof(T));
}

// Pipeline state cache key. Zero-initialized before filling so padding bytes hash consistently.
struct PipelineKey
{
	struct Attrib
	{
		uint8_t enabled;
		uint8_t format;
		uint16_t pad;
		uint32_t stride;
	};
	Attrib attribs[kMaxVertexAttribs];
	uint32_t color_format;
	uint8_t has_depth_stencil;
	uint8_t blend_enabled;
	uint8_t color_mask;
	uint8_t blend_src_color;
	uint8_t blend_dst_color;
	uint8_t blend_op_color;
	uint8_t blend_src_alpha;
	uint8_t blend_dst_alpha;
	uint8_t blend_op_alpha;
};

struct DepthStencilKey
{
	struct Face
	{
		uint8_t compare;
		uint8_t fail;
		uint8_t depth_fail;
		uint8_t pass;
		uint8_t read_mask;
		uint8_t write_mask;
	};
	uint8_t depth_test;
	uint8_t depth_write;
	uint8_t depth_func;
	uint8_t stencil_test;
	Face front;
	Face back;
};

struct SamplerKey
{
	uint8_t u_wrap;
	uint8_t v_wrap;
	uint8_t min;
	uint8_t mag;
};

struct MetalTexture : public rhi::Texture
{
	id<MTLTexture> texture = nil;
	id<MTLTexture> sample_view = nil; // swizzled view used for sampling
	id<MTLSamplerState> sampler = nil;
	TextureDesc desc {};
	uint64_t last_use = 0;
};

struct MetalBuffer : public rhi::Buffer
{
	id<MTLBuffer> buffer = nil;
	BufferDesc desc {};
	uint64_t last_use = 0;
};

struct MetalRenderbuffer : public rhi::Renderbuffer
{
	id<MTLTexture> texture = nil;
	RenderbufferDesc desc {};
};

struct MetalProgram : public rhi::Program
{
	id<MTLFunction> vertex_function = nil;
	id<MTLFunction> fragment_function = nil;
	std::array<std::byte, kUniformsSize> uniforms {};
	std::unordered_map<std::string, id<MTLRenderPipelineState>> pipelines;
};

struct VertexAttribBinding
{
	bool enabled = false;
	Handle<Buffer> buffer;
	VertexAttributeFormat format = VertexAttributeFormat::kFloat;
	uint32_t offset = 0;
	uint32_t stride = 0;
};

struct PassState
{
	bool is_default = false;
	bool pending_clear = false;
	Handle<Texture> color;
	std::optional<Handle<Renderbuffer>> depth_stencil;
};

class MetalRhi final : public Rhi
{
	std::unique_ptr<MetalPlatform> platform_;

	id<MTLDevice> device_ = nil;
	id<MTLCommandQueue> queue_ = nil;
	CAMetalLayer* layer_ = nil;
	dispatch_semaphore_t frames_in_flight_;

	id<MTLCommandBuffer> cmd_ = nil;
	id<MTLRenderCommandEncoder> render_encoder_ = nil;
	id<CAMetalDrawable> drawable_ = nil;
	id<MTLTexture> default_depth_stencil_ = nil;
	id<MTLTexture> fallback_drawable_texture_ = nil; // used when the layer has no drawable (e.g. backgrounded app)

	uint64_t current_serial_ = 0;
	std::atomic<uint64_t> completed_serial_ {0};

	Slab<MetalTexture> texture_slab_;
	Slab<MetalBuffer> buffer_slab_;
	Slab<MetalRenderbuffer> renderbuffer_slab_;
	Slab<MetalProgram> program_slab_;

	std::unordered_map<std::string, id<MTLDepthStencilState>> depth_stencil_states_;
	std::unordered_map<std::string, id<MTLSamplerState>> sampler_states_;

	std::vector<PassState> pass_stack_;
	std::optional<Handle<Program>> current_program_;
	std::array<VertexAttribBinding, kMaxVertexAttribs> vertex_attribs_ {};
	Handle<Buffer> current_index_buffer_;
	std::array<Handle<Texture>, kMaxSamplers> bound_textures_ {};
	RasterizerStateDesc rasterizer_ {};
	Rect viewport_ {};

	uint8_t stencil_front_reference_ = 0;
	uint8_t stencil_front_compare_mask_ = 0xFF;
	uint8_t stencil_front_write_mask_ = 0xFF;
	uint8_t stencil_back_reference_ = 0;
	uint8_t stencil_back_compare_mask_ = 0xFF;
	uint8_t stencil_back_write_mask_ = 0xFF;

	// Description of the current render target, valid while render_encoder_ is open.
	struct Target
	{
		id<MTLTexture> color = nil;
		id<MTLTexture> depth_stencil = nil;
		uint32_t width = 0;
		uint32_t height = 0;
		bool flip_y = false;
	};
	Target target_;

	id<MTLSamplerState> sampler_state(TextureWrapMode u, TextureWrapMode v, TextureFilterMode min, TextureFilterMode mag);
	id<MTLDepthStencilState> depth_stencil_state();
	id<MTLRenderPipelineState> pipeline_state(MetalProgram& program);

	void ensure_command_buffer();
	void ensure_render_encoder();
	void end_render_encoder();
	void flush_pending_clear();
	void commit(bool wait);
	id<MTLTexture> acquire_drawable_texture();
	Target resolve_target(PassState& pass);
	Rect to_metal_rect(const Rect& gl_rect) const;
	bool apply_draw_state();
	bool resource_in_use(uint64_t last_use) const { return last_use > completed_serial_.load(); }
	void reset_default_pass_state();

public:
	explicit MetalRhi(std::unique_ptr<MetalPlatform>&& platform);
	virtual ~MetalRhi();

	virtual Handle<Program> create_program(const ProgramDesc& desc) override;
	virtual void destroy_program(Handle<Program> handle) override;

	virtual Handle<Texture> create_texture(const TextureDesc& desc) override;
	virtual void destroy_texture(Handle<Texture> handle) override;
	virtual Handle<Buffer> create_buffer(const BufferDesc& desc) override;
	virtual void destroy_buffer(Handle<Buffer> handle) override;
	virtual Handle<Renderbuffer> create_renderbuffer(const RenderbufferDesc& desc) override;
	virtual void destroy_renderbuffer(Handle<Renderbuffer> handle) override;

	virtual TextureDetails get_texture_details(Handle<Texture> texture) override;
	virtual Rect get_renderbuffer_size(Handle<Renderbuffer> renderbuffer) override;
	virtual uint32_t get_buffer_size(Handle<Buffer> buffer) override;

	virtual void update_buffer(Handle<Buffer> buffer, uint32_t offset, std::span<const std::byte> data) override;
	virtual void update_texture(
		Handle<Texture> texture,
		Rect region,
		srb2::rhi::PixelFormat data_format,
		std::span<const std::byte> data
	) override;
	virtual void update_texture_settings(
		Handle<Texture> texture,
		TextureWrapMode u_wrap,
		TextureWrapMode v_wrap,
		TextureFilterMode min,
		TextureFilterMode mag
	) override;

	virtual void push_default_render_pass(bool clear) override;
	virtual void push_render_pass(const RenderPassBeginInfo& info) override;
	virtual void pop_render_pass() override;
	virtual void bind_program(Handle<Program> program) override;
	virtual void bind_vertex_attrib(
		const char* name,
		Handle<Buffer> buffer,
		VertexAttributeFormat format,
		uint32_t offset,
		uint32_t stride
	) override;
	virtual void bind_index_buffer(Handle<Buffer> buffer) override;
	virtual void set_uniform(const char* name, float value) override;
	virtual void set_uniform(const char* name, int value) override;
	virtual void set_uniform(const char* name, glm::vec2 value) override;
	virtual void set_uniform(const char* name, glm::vec3 value) override;
	virtual void set_uniform(const char* name, glm::vec4 value) override;
	virtual void set_uniform(const char* name, glm::ivec2 value) override;
	virtual void set_uniform(const char* name, glm::ivec3 value) override;
	virtual void set_uniform(const char* name, glm::ivec4 value) override;
	virtual void set_uniform(const char* name, glm::mat2 value) override;
	virtual void set_uniform(const char* name, glm::mat3 value) override;
	virtual void set_uniform(const char* name, glm::mat4 value) override;
	virtual void set_sampler(const char* name, uint32_t slot, Handle<Texture> texture) override;
	virtual void set_rasterizer_state(const RasterizerStateDesc& desc) override;
	virtual void set_viewport(const Rect& rect) override;
	virtual void draw(uint32_t vertex_count, uint32_t first_vertex) override;
	virtual void draw_indexed(uint32_t index_count, uint32_t first_index) override;
	virtual void read_pixels(const Rect& rect, PixelFormat format, std::span<std::byte> out) override;
	virtual void copy_framebuffer_to_texture(Handle<Texture> dst_tex, const Rect& dst_region, const Rect& src_region)
		override;
	virtual void set_stencil_reference(CullMode face, uint8_t reference) override;
	virtual void set_stencil_compare_mask(CullMode face, uint8_t mask) override;
	virtual void set_stencil_write_mask(CullMode face, uint8_t mask) override;

	virtual void present() override;

	virtual void finish() override;

private:
	void write_uniform(const char* name, UniformKind kind, const void* data, size_t size);
};

MetalRhi::MetalRhi(std::unique_ptr<MetalPlatform>&& platform) : platform_(std::move(platform))
{
	@autoreleasepool
	{
		layer_ = (__bridge CAMetalLayer*)platform_->metal_layer();
		SRB2_ASSERT(layer_ != nil);

		device_ = layer_.device ? layer_.device : MTLCreateSystemDefaultDevice();
		if (device_ == nil)
		{
			throw std::runtime_error("No Metal device available");
		}
		queue_ = [device_ newCommandQueue];
		frames_in_flight_ = dispatch_semaphore_create(kMaxFramesInFlight);

		layer_.device = device_;
		layer_.pixelFormat = kDrawableFormat;
		// Allow blits and readback from the drawable (screenshots).
		layer_.framebufferOnly = NO;
	}
}

MetalRhi::~MetalRhi()
{
	@autoreleasepool
	{
		end_render_encoder();
		commit(true);
		// Wait for every in-flight command buffer; their completion handlers reference this object.
		for (uint32_t i = 0; i < kMaxFramesInFlight; i++)
		{
			dispatch_semaphore_wait(frames_in_flight_, DISPATCH_TIME_FOREVER);
		}
		for (uint32_t i = 0; i < kMaxFramesInFlight; i++)
		{
			dispatch_semaphore_signal(frames_in_flight_);
		}
	}
}

void MetalRhi::ensure_command_buffer()
{
	if (cmd_ != nil)
	{
		return;
	}

	dispatch_semaphore_wait(frames_in_flight_, DISPATCH_TIME_FOREVER);
	cmd_ = [queue_ commandBuffer];
	current_serial_ += 1;

	const uint64_t serial = current_serial_;
	std::atomic<uint64_t>* completed = &completed_serial_;
	dispatch_semaphore_t semaphore = frames_in_flight_;
	[cmd_ addCompletedHandler:^(id<MTLCommandBuffer>) {
		// A single queue completes in order.
		completed->store(serial);
		dispatch_semaphore_signal(semaphore);
	}];
}

void MetalRhi::commit(bool wait)
{
	if (cmd_ == nil)
	{
		return;
	}
	SRB2_ASSERT(render_encoder_ == nil);
	id<MTLCommandBuffer> cmd = cmd_;
	cmd_ = nil;
	[cmd commit];
	if (wait)
	{
		[cmd waitUntilCompleted];
	}
}

id<MTLTexture> MetalRhi::acquire_drawable_texture()
{
	if (drawable_ == nil)
	{
		drawable_ = [layer_ nextDrawable];
	}
	if (drawable_ != nil)
	{
		return drawable_.texture;
	}

	// No drawable available; render into a scratch texture so the frame's work stays valid.
	const CGSize size = layer_.drawableSize;
	const NSUInteger w = std::max<NSUInteger>(1, static_cast<NSUInteger>(size.width));
	const NSUInteger h = std::max<NSUInteger>(1, static_cast<NSUInteger>(size.height));
	if (fallback_drawable_texture_ == nil || fallback_drawable_texture_.width != w || fallback_drawable_texture_.height != h)
	{
		MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kDrawableFormat
																						width:w
																					   height:h
																					mipmapped:NO];
		desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		desc.storageMode = MTLStorageModePrivate;
		fallback_drawable_texture_ = [device_ newTextureWithDescriptor:desc];
	}
	return fallback_drawable_texture_;
}

MetalRhi::Target MetalRhi::resolve_target(PassState& pass)
{
	Target target;
	if (pass.is_default)
	{
		target.color = acquire_drawable_texture();
		target.width = static_cast<uint32_t>(target.color.width);
		target.height = static_cast<uint32_t>(target.color.height);
		if (default_depth_stencil_ == nil || default_depth_stencil_.width != target.width ||
			default_depth_stencil_.height != target.height)
		{
			MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kDepthStencilFormat
																							width:target.width
																						   height:target.height
																						mipmapped:NO];
			desc.usage = MTLTextureUsageRenderTarget;
			desc.storageMode = MTLStorageModePrivate;
			default_depth_stencil_ = [device_ newTextureWithDescriptor:desc];
		}
		target.depth_stencil = default_depth_stencil_;
		target.flip_y = false;
	}
	else
	{
		SRB2_ASSERT(texture_slab_.is_valid(pass.color));
		MetalTexture& color = texture_slab_[pass.color];
		color.last_use = current_serial_;
		target.color = color.texture;
		target.width = color.desc.width;
		target.height = color.desc.height;
		if (pass.depth_stencil.has_value())
		{
			SRB2_ASSERT(renderbuffer_slab_.is_valid(*pass.depth_stencil));
			target.depth_stencil = renderbuffer_slab_[*pass.depth_stencil].texture;
		}
		target.flip_y = true;
	}
	return target;
}

void MetalRhi::ensure_render_encoder()
{
	if (render_encoder_ != nil)
	{
		return;
	}
	SRB2_ASSERT(pass_stack_.empty() == false);

	ensure_command_buffer();

	PassState& pass = pass_stack_.back();
	target_ = resolve_target(pass);

	MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
	desc.colorAttachments[0].texture = target_.color;
	desc.colorAttachments[0].loadAction = pass.pending_clear ? MTLLoadActionClear : MTLLoadActionLoad;
	desc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
	desc.colorAttachments[0].storeAction = MTLStoreActionStore;
	if (target_.depth_stencil != nil)
	{
		desc.depthAttachment.texture = target_.depth_stencil;
		desc.depthAttachment.loadAction = pass.pending_clear ? MTLLoadActionClear : MTLLoadActionLoad;
		desc.depthAttachment.clearDepth = 1.0;
		desc.depthAttachment.storeAction = MTLStoreActionStore;
		desc.stencilAttachment.texture = target_.depth_stencil;
		desc.stencilAttachment.loadAction = pass.pending_clear ? MTLLoadActionClear : MTLLoadActionLoad;
		desc.stencilAttachment.clearStencil = 0;
		desc.stencilAttachment.storeAction = MTLStoreActionStore;
	}
	pass.pending_clear = false;

	render_encoder_ = [cmd_ renderCommandEncoderWithDescriptor:desc];
}

void MetalRhi::end_render_encoder()
{
	if (render_encoder_ == nil)
	{
		return;
	}
	[render_encoder_ endEncoding];
	render_encoder_ = nil;
}

void MetalRhi::flush_pending_clear()
{
	if (!pass_stack_.empty() && pass_stack_.back().pending_clear)
	{
		ensure_render_encoder();
	}
}

void MetalRhi::reset_default_pass_state()
{
	// Mirrors Gl2Rhi::apply_default_framebuffer: full viewport, scissor disabled.
	const Rect fb = platform_->get_default_framebuffer_dimensions();
	viewport_ = {0, 0, fb.w, fb.h};
	rasterizer_.scissor_test = false;
}

id<MTLSamplerState>
MetalRhi::sampler_state(TextureWrapMode u, TextureWrapMode v, TextureFilterMode min, TextureFilterMode mag)
{
	SamplerKey key {};
	key.u_wrap = static_cast<uint8_t>(u);
	key.v_wrap = static_cast<uint8_t>(v);
	key.min = static_cast<uint8_t>(min);
	key.mag = static_cast<uint8_t>(mag);
	std::string key_str(key_bytes(key));

	auto itr = sampler_states_.find(key_str);
	if (itr != sampler_states_.end())
	{
		return itr->second;
	}

	MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
	desc.sAddressMode = map_wrap(u);
	desc.tAddressMode = map_wrap(v);
	desc.minFilter = map_filter(min);
	desc.magFilter = map_filter(mag);
	desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
	id<MTLSamplerState> state = [device_ newSamplerStateWithDescriptor:desc];
	sampler_states_.emplace(std::move(key_str), state);
	return state;
}

id<MTLDepthStencilState> MetalRhi::depth_stencil_state()
{
	DepthStencilKey key {};
	// As in GL, disabling the depth test also disables depth writes.
	key.depth_test = rasterizer_.depth_test;
	key.depth_write = rasterizer_.depth_test && rasterizer_.depth_write;
	key.depth_func = static_cast<uint8_t>(rasterizer_.depth_test ? rasterizer_.depth_func : CompareFunc::kAlways);
	key.stencil_test = rasterizer_.stencil_test;
	if (rasterizer_.stencil_test)
	{
		key.front = {
			static_cast<uint8_t>(rasterizer_.front_stencil_compare),
			static_cast<uint8_t>(rasterizer_.front_fail),
			static_cast<uint8_t>(rasterizer_.front_depth_fail),
			static_cast<uint8_t>(rasterizer_.front_pass),
			stencil_front_compare_mask_,
			stencil_front_write_mask_
		};
		key.back = {
			static_cast<uint8_t>(rasterizer_.back_stencil_compare),
			static_cast<uint8_t>(rasterizer_.back_fail),
			static_cast<uint8_t>(rasterizer_.back_depth_fail),
			static_cast<uint8_t>(rasterizer_.back_pass),
			stencil_back_compare_mask_,
			stencil_back_write_mask_
		};
	}
	std::string key_str(key_bytes(key));

	auto itr = depth_stencil_states_.find(key_str);
	if (itr != depth_stencil_states_.end())
	{
		return itr->second;
	}

	MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
	desc.depthCompareFunction = map_compare_func(static_cast<CompareFunc>(key.depth_func));
	desc.depthWriteEnabled = key.depth_write ? YES : NO;
	if (key.stencil_test)
	{
		auto make_face = [](const DepthStencilKey::Face& face)
		{
			MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
			stencil.stencilCompareFunction = map_compare_func(static_cast<CompareFunc>(face.compare));
			stencil.stencilFailureOperation = map_stencil_op(static_cast<StencilOp>(face.fail));
			stencil.depthFailureOperation = map_stencil_op(static_cast<StencilOp>(face.depth_fail));
			stencil.depthStencilPassOperation = map_stencil_op(static_cast<StencilOp>(face.pass));
			stencil.readMask = face.read_mask;
			stencil.writeMask = face.write_mask;
			return stencil;
		};
		desc.frontFaceStencil = make_face(key.front);
		desc.backFaceStencil = make_face(key.back);
	}
	id<MTLDepthStencilState> state = [device_ newDepthStencilStateWithDescriptor:desc];
	depth_stencil_states_.emplace(std::move(key_str), state);
	return state;
}

id<MTLRenderPipelineState> MetalRhi::pipeline_state(MetalProgram& program)
{
	PipelineKey key {};
	for (uint32_t i = 0; i < kMaxVertexAttribs; i++)
	{
		const VertexAttribBinding& attrib = vertex_attribs_[i];
		if (!attrib.enabled)
		{
			continue;
		}
		key.attribs[i].enabled = 1;
		key.attribs[i].format = static_cast<uint8_t>(attrib.format);
		key.attribs[i].stride = attrib.stride != 0 ? attrib.stride : vertex_format_size(attrib.format);
	}
	key.color_format = static_cast<uint32_t>(target_.color.pixelFormat);
	key.has_depth_stencil = target_.depth_stencil != nil;
	key.blend_enabled = rasterizer_.blend_enabled;
	key.color_mask = (rasterizer_.color_mask.r ? MTLColorWriteMaskRed : 0) |
		(rasterizer_.color_mask.g ? MTLColorWriteMaskGreen : 0) | (rasterizer_.color_mask.b ? MTLColorWriteMaskBlue : 0) |
		(rasterizer_.color_mask.a ? MTLColorWriteMaskAlpha : 0);
	if (rasterizer_.blend_enabled)
	{
		key.blend_src_color = static_cast<uint8_t>(rasterizer_.blend_source_factor_color);
		key.blend_dst_color = static_cast<uint8_t>(rasterizer_.blend_dest_factor_color);
		key.blend_op_color = static_cast<uint8_t>(rasterizer_.blend_color_function);
		key.blend_src_alpha = static_cast<uint8_t>(rasterizer_.blend_source_factor_alpha);
		key.blend_dst_alpha = static_cast<uint8_t>(rasterizer_.blend_dest_factor_alpha);
		key.blend_op_alpha = static_cast<uint8_t>(rasterizer_.blend_alpha_function);
	}
	std::string key_str(key_bytes(key));

	auto itr = program.pipelines.find(key_str);
	if (itr != program.pipelines.end())
	{
		return itr->second;
	}

	MTLVertexDescriptor* vertex_desc = [MTLVertexDescriptor vertexDescriptor];
	for (uint32_t i = 0; i < kMaxVertexAttribs; i++)
	{
		if (!key.attribs[i].enabled)
		{
			continue;
		}
		vertex_desc.attributes[i].format = map_vertex_format(static_cast<VertexAttributeFormat>(key.attribs[i].format));
		vertex_desc.attributes[i].offset = 0;
		vertex_desc.attributes[i].bufferIndex = kVertexBufferBaseIndex + i;
		vertex_desc.layouts[kVertexBufferBaseIndex + i].stride = key.attribs[i].stride;
		vertex_desc.layouts[kVertexBufferBaseIndex + i].stepFunction = MTLVertexStepFunctionPerVertex;
	}

	MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
	desc.vertexFunction = program.vertex_function;
	desc.fragmentFunction = program.fragment_function;
	desc.vertexDescriptor = vertex_desc;
	MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[0];
	color.pixelFormat = target_.color.pixelFormat;
	color.writeMask = key.color_mask;
	color.blendingEnabled = key.blend_enabled ? YES : NO;
	if (key.blend_enabled)
	{
		color.sourceRGBBlendFactor = map_blend_factor(rasterizer_.blend_source_factor_color);
		color.destinationRGBBlendFactor = map_blend_factor(rasterizer_.blend_dest_factor_color);
		color.rgbBlendOperation = map_blend_function(rasterizer_.blend_color_function);
		color.sourceAlphaBlendFactor = map_blend_factor(rasterizer_.blend_source_factor_alpha);
		color.destinationAlphaBlendFactor = map_blend_factor(rasterizer_.blend_dest_factor_alpha);
		color.alphaBlendOperation = map_blend_function(rasterizer_.blend_alpha_function);
	}
	if (key.has_depth_stencil)
	{
		desc.depthAttachmentPixelFormat = kDepthStencilFormat;
		desc.stencilAttachmentPixelFormat = kDepthStencilFormat;
	}

	NSError* error = nil;
	id<MTLRenderPipelineState> state = [device_ newRenderPipelineStateWithDescriptor:desc error:&error];
	if (state == nil)
	{
		I_Error("Metal pipeline creation failed: %s", error ? error.localizedDescription.UTF8String : "unknown error");
	}
	program.pipelines.emplace(std::move(key_str), state);
	return state;
}

Rect MetalRhi::to_metal_rect(const Rect& gl_rect) const
{
	// Offscreen targets are stored bottom-up like GL, so GL coordinates apply directly.
	if (target_.flip_y)
	{
		return gl_rect;
	}
	Rect rect = gl_rect;
	rect.y = static_cast<int32_t>(target_.height) - (gl_rect.y + static_cast<int32_t>(gl_rect.h));
	return rect;
}

bool MetalRhi::apply_draw_state()
{
	SRB2_ASSERT(current_program_.has_value());
	SRB2_ASSERT(program_slab_.is_valid(*current_program_));
	MetalProgram& program = program_slab_[*current_program_];

	ensure_render_encoder();

	// Scissor first: an empty scissor rect means nothing would be drawn, and Metal rejects it.
	MTLScissorRect scissor = {0, 0, target_.width, target_.height};
	if (rasterizer_.scissor_test)
	{
		Rect rect = to_metal_rect(rasterizer_.scissor);
		int64_t x0 = std::clamp<int64_t>(rect.x, 0, target_.width);
		int64_t y0 = std::clamp<int64_t>(rect.y, 0, target_.height);
		int64_t x1 = std::clamp<int64_t>(static_cast<int64_t>(rect.x) + rect.w, 0, target_.width);
		int64_t y1 = std::clamp<int64_t>(static_cast<int64_t>(rect.y) + rect.h, 0, target_.height);
		if (x1 <= x0 || y1 <= y0)
		{
			return false;
		}
		scissor = {static_cast<NSUInteger>(x0), static_cast<NSUInteger>(y0), static_cast<NSUInteger>(x1 - x0), static_cast<NSUInteger>(y1 - y0)};
	}
	[render_encoder_ setScissorRect:scissor];

	Rect viewport_rect = viewport_;
	if (viewport_rect.w == 0 || viewport_rect.h == 0)
	{
		viewport_rect = {0, 0, target_.width, target_.height};
	}
	viewport_rect = to_metal_rect(viewport_rect);
	[render_encoder_ setViewport:(MTLViewport) {
		static_cast<double>(viewport_rect.x),
		static_cast<double>(viewport_rect.y),
		static_cast<double>(viewport_rect.w),
		static_cast<double>(viewport_rect.h),
		0.0,
		1.0
	}];

	[render_encoder_ setRenderPipelineState:pipeline_state(program)];
	[render_encoder_ setDepthStencilState:depth_stencil_state()];
	[render_encoder_ setStencilFrontReferenceValue:stencil_front_reference_ backReferenceValue:stencil_back_reference_];
	[render_encoder_ setBlendColorRed:rasterizer_.blend_color.r
								green:rasterizer_.blend_color.g
								 blue:rasterizer_.blend_color.b
								alpha:rasterizer_.blend_color.a];

	switch (rasterizer_.cull)
	{
	case CullMode::kFront:
		[render_encoder_ setCullMode:MTLCullModeFront];
		break;
	case CullMode::kBack:
		[render_encoder_ setCullMode:MTLCullModeBack];
		break;
	default:
		[render_encoder_ setCullMode:MTLCullModeNone];
		break;
	}
	// Flipping Y inverts the apparent winding.
	bool ccw = rasterizer_.winding == FaceWinding::kCounterClockwise;
	if (target_.flip_y)
	{
		ccw = !ccw;
	}
	[render_encoder_ setFrontFacingWinding:ccw ? MTLWindingCounterClockwise : MTLWindingClockwise];

	std::array<std::byte, kUniformsSize> uniforms = program.uniforms;
	const float flip_y = target_.flip_y ? -1.f : 1.f;
	std::memcpy(uniforms.data() + kUniformFlipYOffset, &flip_y, sizeof(flip_y));
	[render_encoder_ setVertexBytes:uniforms.data() length:uniforms.size() atIndex:0];
	[render_encoder_ setFragmentBytes:uniforms.data() length:uniforms.size() atIndex:0];

	for (uint32_t i = 0; i < kMaxVertexAttribs; i++)
	{
		const VertexAttribBinding& attrib = vertex_attribs_[i];
		if (!attrib.enabled || !buffer_slab_.is_valid(attrib.buffer))
		{
			continue;
		}
		MetalBuffer& buffer = buffer_slab_[attrib.buffer];
		buffer.last_use = current_serial_;
		[render_encoder_ setVertexBuffer:buffer.buffer offset:attrib.offset atIndex:kVertexBufferBaseIndex + i];
	}

	for (uint32_t i = 0; i < kMaxSamplers; i++)
	{
		if (bound_textures_[i] == kNullHandle || !texture_slab_.is_valid(bound_textures_[i]))
		{
			continue;
		}
		MetalTexture& texture = texture_slab_[bound_textures_[i]];
		if (texture.texture == nil)
		{
			continue;
		}
		texture.last_use = current_serial_;
		[render_encoder_ setFragmentTexture:texture.sample_view atIndex:i];
		[render_encoder_ setFragmentSamplerState:texture.sampler atIndex:i];
	}

	return true;
}

} // namespace

MetalPlatform::~MetalPlatform() = default;

std::unique_ptr<Rhi> create_metal_rhi(std::unique_ptr<MetalPlatform>&& platform)
{
	return std::make_unique<MetalRhi>(std::move(platform));
}

Handle<Program> MetalRhi::create_program(const ProgramDesc& desc)
{
	@autoreleasepool
	{
		const ProgramEntry* entry = nullptr;
		for (const ProgramEntry& candidate : kPrograms)
		{
			if (std::strcmp(candidate.name, desc.name) == 0)
			{
				entry = &candidate;
				break;
			}
		}
		if (entry == nullptr)
		{
			throw std::runtime_error(fmt::format("Metal RHI has no program named {}", desc.name));
		}

		NSMutableDictionary<NSString*, NSObject*>* macros = [NSMutableDictionary dictionary];
		for (const char* define : desc.defines)
		{
			macros[[NSString stringWithUTF8String:define]] = @1;
		}
		if (entry->implicit_define)
		{
			macros[[NSString stringWithUTF8String:entry->implicit_define]] = @1;
		}

		MTLCompileOptions* options = [MTLCompileOptions new];
		options.preprocessorMacros = macros;

		NSError* error = nil;
		id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kMetalShaderSource]
													   options:options
														 error:&error];
		if (library == nil)
		{
			throw std::runtime_error(fmt::format(
				"Metal shader compilation failed for {}: {}",
				desc.name,
				error ? error.localizedDescription.UTF8String : "unknown error"
			));
		}

		MetalProgram program;
		program.vertex_function = [library newFunctionWithName:[NSString stringWithUTF8String:entry->vertex_function]];
		program.fragment_function = [library newFunctionWithName:[NSString stringWithUTF8String:entry->fragment_function]];
		SRB2_ASSERT(program.vertex_function != nil && program.fragment_function != nil);
		return program_slab_.insert(std::move(program));
	}
}

void MetalRhi::destroy_program(Handle<Program> handle)
{
	SRB2_ASSERT(program_slab_.is_valid(handle));
	program_slab_.remove(handle);
	if (current_program_ == handle)
	{
		current_program_ = std::nullopt;
	}
}

Handle<Texture> MetalRhi::create_texture(const TextureDesc& desc)
{
	@autoreleasepool
	{
		const MTLPixelFormat format = map_texture_format(desc.format);
		MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
																							width:std::max<uint32_t>(desc.width, 1)
																						   height:std::max<uint32_t>(desc.height, 1)
																						mipmapped:NO];
		tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		if (desc.format == TextureFormat::kRGB || desc.format == TextureFormat::kRGBA)
		{
			tex_desc.usage |= MTLTextureUsageRenderTarget;
		}
		// Shared storage lets idle textures be written directly from the CPU (unified memory on iOS).
		tex_desc.storageMode = MTLStorageModeShared;

		MetalTexture texture;
		texture.desc = desc;
		texture.texture = [device_ newTextureWithDescriptor:tex_desc];
		SRB2_ASSERT(texture.texture != nil);
		if (desc.format == TextureFormat::kRGBA)
		{
			texture.sample_view = texture.texture;
		}
		else
		{
			texture.sample_view = [texture.texture newTextureViewWithPixelFormat:format
																	 textureType:MTLTextureType2D
																		  levels:NSMakeRange(0, 1)
																		  slices:NSMakeRange(0, 1)
																		 swizzle:texture_swizzle(desc.format)];
		}
		texture.sampler = sampler_state(desc.u_wrap, desc.v_wrap, desc.min, desc.mag);
		return texture_slab_.insert(std::move(texture));
	}
}

void MetalRhi::destroy_texture(Handle<Texture> handle)
{
	SRB2_ASSERT(texture_slab_.is_valid(handle));
	// The command buffer retains anything still in use by the GPU.
	texture_slab_.remove(handle);
}

Handle<Buffer> MetalRhi::create_buffer(const BufferDesc& desc)
{
	@autoreleasepool
	{
		MetalBuffer buffer;
		buffer.desc = desc;
		buffer.buffer = [device_ newBufferWithLength:std::max<uint32_t>(desc.size, 4) options:MTLResourceStorageModeShared];
		SRB2_ASSERT(buffer.buffer != nil);
		return buffer_slab_.insert(std::move(buffer));
	}
}

void MetalRhi::destroy_buffer(Handle<Buffer> handle)
{
	SRB2_ASSERT(buffer_slab_.is_valid(handle));
	buffer_slab_.remove(handle);
}

Handle<Renderbuffer> MetalRhi::create_renderbuffer(const RenderbufferDesc& desc)
{
	@autoreleasepool
	{
		// Packed depth-stencil, as the RHI requires. iOS GPUs have no D24S8, so use D32FS8.
		MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kDepthStencilFormat
																							width:std::max<uint32_t>(desc.width, 1)
																						   height:std::max<uint32_t>(desc.height, 1)
																						mipmapped:NO];
		tex_desc.usage = MTLTextureUsageRenderTarget;
		tex_desc.storageMode = MTLStorageModePrivate;

		MetalRenderbuffer rb;
		rb.desc = desc;
		rb.texture = [device_ newTextureWithDescriptor:tex_desc];
		SRB2_ASSERT(rb.texture != nil);
		return renderbuffer_slab_.insert(std::move(rb));
	}
}

void MetalRhi::destroy_renderbuffer(Handle<Renderbuffer> handle)
{
	SRB2_ASSERT(renderbuffer_slab_.is_valid(handle));
	renderbuffer_slab_.remove(handle);
}

TextureDetails MetalRhi::get_texture_details(Handle<Texture> texture)
{
	SRB2_ASSERT(texture_slab_.is_valid(texture));
	auto& t = texture_slab_[texture];
	return {t.desc.width, t.desc.height, t.desc.format};
}

Rect MetalRhi::get_renderbuffer_size(Handle<Renderbuffer> renderbuffer)
{
	SRB2_ASSERT(renderbuffer_slab_.is_valid(renderbuffer));
	auto& rb = renderbuffer_slab_[renderbuffer];
	return {0, 0, rb.desc.width, rb.desc.height};
}

uint32_t MetalRhi::get_buffer_size(Handle<Buffer> buffer)
{
	SRB2_ASSERT(buffer_slab_.is_valid(buffer));
	return buffer_slab_[buffer].desc.size;
}

void MetalRhi::update_buffer(Handle<Buffer> handle, uint32_t offset, std::span<const std::byte> data)
{
	if (data.empty())
	{
		return;
	}

	SRB2_ASSERT(buffer_slab_.is_valid(handle));
	MetalBuffer& b = buffer_slab_[handle];
	SRB2_ASSERT(offset < b.desc.size && offset + data.size() <= b.desc.size);

	@autoreleasepool
	{
		if (resource_in_use(b.last_use))
		{
			// The GPU may still read the current contents: rename the buffer, keeping the old contents.
			id<MTLBuffer> renamed = [device_ newBufferWithLength:b.buffer.length options:MTLResourceStorageModeShared];
			std::memcpy(renamed.contents, b.buffer.contents, b.buffer.length);
			b.buffer = renamed;
			b.last_use = 0;
		}
		std::memcpy(static_cast<std::byte*>(b.buffer.contents) + offset, data.data(), data.size());
	}
}

void MetalRhi::update_texture(
	Handle<Texture> texture,
	Rect region,
	srb2::rhi::PixelFormat data_format,
	std::span<const std::byte> data
)
{
	if (data.empty())
	{
		return;
	}

	SRB2_ASSERT(texture_slab_.is_valid(texture));
	MetalTexture& t = texture_slab_[texture];
	SRB2_ASSERT(region.x + region.w <= t.desc.width && region.y + region.h <= t.desc.height);

	const uint32_t src_bpp = pixel_format_bpp(data_format);
	const uint32_t dst_bpp = texture_format_bpp(t.desc.format);
	SRB2_ASSERT(src_bpp != 0);
	const size_t src_row_span = ((src_bpp * region.w) + kPixelRowUnpackAlignment - 1) & ~(kPixelRowUnpackAlignment - 1);
	SRB2_ASSERT(src_row_span * region.h == data.size_bytes());

	// Metal has no 3-byte formats; expand RGB data to RGBA.
	std::vector<std::byte> expanded;
	const std::byte* pixels = data.data();
	size_t row_span = src_row_span;
	if (src_bpp != dst_bpp)
	{
		SRB2_ASSERT(src_bpp == 3 && dst_bpp == 4);
		row_span = static_cast<size_t>(region.w) * 4;
		expanded.resize(row_span * region.h);
		for (uint32_t y = 0; y < region.h; y++)
		{
			const std::byte* src_row = data.data() + y * src_row_span;
			std::byte* dst_row = expanded.data() + y * row_span;
			for (uint32_t x = 0; x < region.w; x++)
			{
				dst_row[x * 4 + 0] = src_row[x * 3 + 0];
				dst_row[x * 4 + 1] = src_row[x * 3 + 1];
				dst_row[x * 4 + 2] = src_row[x * 3 + 2];
				dst_row[x * 4 + 3] = std::byte {0xFF};
			}
		}
		pixels = expanded.data();
	}

	@autoreleasepool
	{
		const MTLRegion mtl_region = MTLRegionMake2D(region.x, region.y, region.w, region.h);
		if (!resource_in_use(t.last_use))
		{
			[t.texture replaceRegion:mtl_region mipmapLevel:0 withBytes:pixels bytesPerRow:row_span];
			return;
		}

		// The GPU may still read the texture: order the upload in the command stream instead.
		flush_pending_clear();
		end_render_encoder();
		ensure_command_buffer();
		id<MTLBuffer> staging = [device_ newBufferWithBytes:pixels
													 length:row_span * region.h
													options:MTLResourceStorageModeShared];
		id<MTLBlitCommandEncoder> blit = [cmd_ blitCommandEncoder];
		[blit copyFromBuffer:staging
				 sourceOffset:0
			sourceBytesPerRow:row_span
		  sourceBytesPerImage:row_span * region.h
				   sourceSize:MTLSizeMake(region.w, region.h, 1)
					toTexture:t.texture
			 destinationSlice:0
			 destinationLevel:0
			destinationOrigin:MTLOriginMake(region.x, region.y, 0)];
		[blit endEncoding];
		t.last_use = current_serial_;
	}
}

void MetalRhi::update_texture_settings(
	Handle<Texture> texture,
	TextureWrapMode u_wrap,
	TextureWrapMode v_wrap,
	TextureFilterMode min,
	TextureFilterMode mag
)
{
	SRB2_ASSERT(texture_slab_.is_valid(texture));
	MetalTexture& t = texture_slab_[texture];
	t.desc.u_wrap = u_wrap;
	t.desc.v_wrap = v_wrap;
	t.desc.min = min;
	t.desc.mag = mag;
	@autoreleasepool
	{
		t.sampler = sampler_state(u_wrap, v_wrap, min, mag);
	}
}

void MetalRhi::push_default_render_pass(bool clear)
{
	@autoreleasepool
	{
		flush_pending_clear();
		end_render_encoder();

		// Track the window's pixel size. Only resize between frames, never while holding a drawable.
		if (drawable_ == nil)
		{
			const Rect fb = platform_->get_default_framebuffer_dimensions();
			const CGSize size = CGSizeMake(fb.w, fb.h);
			if (fb.w > 0 && fb.h > 0 && !CGSizeEqualToSize(layer_.drawableSize, size))
			{
				layer_.drawableSize = size;
			}
		}

		PassState pass;
		pass.is_default = true;
		pass.pending_clear = clear;
		pass_stack_.push_back(pass);
		reset_default_pass_state();
	}
}

void MetalRhi::push_render_pass(const RenderPassBeginInfo& info)
{
	@autoreleasepool
	{
		flush_pending_clear();
		end_render_encoder();

		SRB2_ASSERT(texture_slab_.is_valid(info.color_attachment));
		PassState pass;
		pass.is_default = false;
		// Gl2Rhi never clears when pushing a render pass, so neither do we.
		pass.pending_clear = false;
		pass.color = info.color_attachment;
		pass.depth_stencil = info.depth_stencil_attachment;
		pass_stack_.push_back(pass);
	}
}

void MetalRhi::pop_render_pass()
{
	SRB2_ASSERT(pass_stack_.empty() == false);
	@autoreleasepool
	{
		flush_pending_clear();
		end_render_encoder();
		current_program_ = std::nullopt;
		pass_stack_.pop_back();
		if (!pass_stack_.empty() && pass_stack_.back().is_default)
		{
			reset_default_pass_state();
		}
	}
}

void MetalRhi::bind_program(Handle<Program> program)
{
	SRB2_ASSERT(pass_stack_.empty() == false);
	SRB2_ASSERT(program_slab_.is_valid(program));
	current_program_ = program;
	vertex_attribs_ = {};
}

void MetalRhi::bind_vertex_attrib(
	const char* name,
	Handle<Buffer> buffer,
	VertexAttributeFormat format,
	uint32_t offset,
	uint32_t stride
)
{
	SRB2_ASSERT(current_program_.has_value());
	SRB2_ASSERT(buffer_slab_.is_valid(buffer));
	SRB2_ASSERT(buffer_slab_[buffer].desc.type == BufferType::kVertexBuffer);

	const int index = vertex_attribute_index(name);
	SRB2_ASSERT(index >= 0);
	if (index < 0)
	{
		return;
	}
	vertex_attribs_[index] = {true, buffer, format, offset, stride};
}

void MetalRhi::bind_index_buffer(Handle<Buffer> buffer)
{
	SRB2_ASSERT(current_program_.has_value());
	SRB2_ASSERT(buffer_slab_.is_valid(buffer));
	SRB2_ASSERT(buffer_slab_[buffer].desc.type == BufferType::kIndexBuffer);
	current_index_buffer_ = buffer;
}

void MetalRhi::write_uniform(const char* name, UniformKind kind, const void* data, size_t size)
{
	SRB2_ASSERT(current_program_.has_value());
	const UniformSlot* slot = find_uniform_slot(name);
	if (slot == nullptr || slot->kind != kind)
	{
		// Uniforms that no Metal program reads.
		return;
	}
	MetalProgram& program = program_slab_[*current_program_];
	std::memcpy(program.uniforms.data() + slot->offset, data, size);
}

void MetalRhi::set_uniform(const char* name, float value)
{
	write_uniform(name, UniformKind::kFloat, &value, sizeof(value));
}

void MetalRhi::set_uniform(const char* name, int value)
{
	int32_t v = value;
	write_uniform(name, UniformKind::kInt, &v, sizeof(v));
}

void MetalRhi::set_uniform(const char* name, glm::vec2 value)
{
	write_uniform(name, UniformKind::kFloat2, glm::value_ptr(value), sizeof(float) * 2);
}

void MetalRhi::set_uniform(const char*, glm::vec3)
{
}

void MetalRhi::set_uniform(const char*, glm::vec4)
{
}

void MetalRhi::set_uniform(const char*, glm::ivec2)
{
}

void MetalRhi::set_uniform(const char*, glm::ivec3)
{
}

void MetalRhi::set_uniform(const char*, glm::ivec4)
{
}

void MetalRhi::set_uniform(const char*, glm::mat2)
{
}

void MetalRhi::set_uniform(const char* name, glm::mat3 value)
{
	// MSL float3x3 columns are padded to 16 bytes.
	float padded[12] = {};
	for (int column = 0; column < 3; column++)
	{
		for (int row = 0; row < 3; row++)
		{
			padded[column * 4 + row] = value[column][row];
		}
	}
	write_uniform(name, UniformKind::kMat3, padded, sizeof(padded));
}

void MetalRhi::set_uniform(const char* name, glm::mat4 value)
{
	write_uniform(name, UniformKind::kMat4, glm::value_ptr(value), sizeof(float) * 16);
}

void MetalRhi::set_sampler(const char* name, uint32_t slot, Handle<Texture> texture)
{
	SRB2_ASSERT(slot < kMaxSamplers);
	SRB2_ASSERT(current_program_.has_value() && pass_stack_.empty() == false);
	SRB2_ASSERT(texture_slab_.is_valid(texture));

	int index = sampler_index(name);
	if (index < 0)
	{
		index = static_cast<int>(slot);
	}
	bound_textures_[index] = texture;
}

void MetalRhi::set_rasterizer_state(const RasterizerStateDesc& desc)
{
	rasterizer_ = desc;
	// Gl2Rhi resets the dynamic stencil state with the rasterizer state.
	stencil_front_reference_ = 0;
	stencil_back_reference_ = 0;
	stencil_front_compare_mask_ = 0xFF;
	stencil_back_compare_mask_ = 0xFF;
	stencil_front_write_mask_ = 0xFF;
	stencil_back_write_mask_ = 0xFF;
}

void MetalRhi::set_viewport(const Rect& rect)
{
	SRB2_ASSERT(pass_stack_.empty() == false);
	viewport_ = rect;
}

void MetalRhi::draw(uint32_t vertex_count, uint32_t first_vertex)
{
	SRB2_ASSERT(pass_stack_.empty() == false);
	std::optional<MTLPrimitiveType> primitive = map_primitive(rasterizer_.primitive);
	if (!primitive || vertex_count == 0)
	{
		return;
	}

	@autoreleasepool
	{
		if (!apply_draw_state())
		{
			return;
		}
		[render_encoder_ drawPrimitives:*primitive vertexStart:first_vertex vertexCount:vertex_count];
	}
}

void MetalRhi::draw_indexed(uint32_t index_count, uint32_t first_index)
{
	SRB2_ASSERT(current_index_buffer_ != kNullHandle);
	SRB2_ASSERT(buffer_slab_.is_valid(current_index_buffer_));
	std::optional<MTLPrimitiveType> primitive = map_primitive(rasterizer_.primitive);
	if (!primitive || index_count == 0)
	{
		return;
	}

	MetalBuffer& ib = buffer_slab_[current_index_buffer_];
	SRB2_ASSERT((index_count + first_index) * 2 <= ib.desc.size);

	@autoreleasepool
	{
		if (!apply_draw_state())
		{
			return;
		}
		ib.last_use = current_serial_;
		[render_encoder_ drawIndexedPrimitives:*primitive
									indexCount:index_count
									 indexType:MTLIndexTypeUInt16
								   indexBuffer:ib.buffer
							 indexBufferOffset:static_cast<NSUInteger>(first_index) * 2];
	}
}

void MetalRhi::read_pixels(const Rect& rect, PixelFormat format, std::span<std::byte> out)
{
	SRB2_ASSERT(pass_stack_.empty() == false);

	const uint32_t dst_bpp = pixel_format_bpp(format);
	SRB2_ASSERT(dst_bpp != 0);
	const size_t pack_stride = (rect.w * dst_bpp + (kPixelRowPackAlignment - 1)) & ~(kPixelRowPackAlignment - 1);
	SRB2_ASSERT(out.size_bytes() == pack_stride * rect.h);
	if (rect.w == 0 || rect.h == 0)
	{
		return;
	}

	@autoreleasepool
	{
		// Open the pass so its target is resolved (and any clear has happened), then read it back.
		ensure_render_encoder();
		end_render_encoder();

		SRB2_ASSERT(rect.x >= 0 && rect.y >= 0);
		SRB2_ASSERT(rect.x + rect.w <= target_.width && rect.y + rect.h <= target_.height);

		const Rect src = to_metal_rect(rect);
		const bool bgra = target_.color.pixelFormat == MTLPixelFormatBGRA8Unorm;
		const size_t readback_stride = static_cast<size_t>(rect.w) * 4;
		id<MTLBuffer> readback = [device_ newBufferWithLength:readback_stride * rect.h
													   options:MTLResourceStorageModeShared];

		id<MTLBlitCommandEncoder> blit = [cmd_ blitCommandEncoder];
		[blit copyFromTexture:target_.color
					  sourceSlice:0
					  sourceLevel:0
					 sourceOrigin:MTLOriginMake(src.x, src.y, 0)
					   sourceSize:MTLSizeMake(rect.w, rect.h, 1)
						 toBuffer:readback
				destinationOffset:0
		   destinationBytesPerRow:readback_stride
		 destinationBytesPerImage:readback_stride * rect.h];
		[blit endEncoding];
		commit(true);

		const uint8_t* pixels = static_cast<const uint8_t*>(readback.contents);
		for (uint32_t row = 0; row < rect.h; row++)
		{
			// Output rows are bottom-up like glReadPixels. Offscreen targets already are; the drawable is top-down.
			const uint32_t src_row = target_.flip_y ? row : (rect.h - 1 - row);
			const uint8_t* src_px = pixels + src_row * readback_stride;
			uint8_t* dst_px = reinterpret_cast<uint8_t*>(out.data()) + row * pack_stride;
			for (uint32_t x = 0; x < rect.w; x++)
			{
				const uint8_t* p = src_px + x * 4;
				const uint8_t rgba[4] = {bgra ? p[2] : p[0], p[1], bgra ? p[0] : p[2], p[3]};
				std::memcpy(dst_px + x * dst_bpp, rgba, dst_bpp);
			}
		}
	}
}

void MetalRhi::copy_framebuffer_to_texture(Handle<Texture> dst_tex, const Rect& dst_region, const Rect& src_region)
{
	SRB2_ASSERT(pass_stack_.empty() == false);
	SRB2_ASSERT(texture_slab_.is_valid(dst_tex));
	SRB2_ASSERT(dst_region.w == src_region.w && dst_region.h == src_region.h);

	MetalTexture& dst = texture_slab_[dst_tex];
	SRB2_ASSERT(dst_region.x >= 0 && dst_region.y >= 0);
	SRB2_ASSERT(dst_region.x + dst_region.w <= dst.desc.width && dst_region.y + dst_region.h <= dst.desc.height);
	if (src_region.w == 0 || src_region.h == 0)
	{
		return;
	}

	@autoreleasepool
	{
		ensure_render_encoder();
		end_render_encoder();

		SRB2_ASSERT(src_region.x >= 0 && src_region.y >= 0);
		SRB2_ASSERT(src_region.x + src_region.w <= target_.width && src_region.y + src_region.h <= target_.height);

		if (target_.color.pixelFormat != dst.texture.pixelFormat || !target_.flip_y)
		{
			// Only offscreen RGBA targets can be copied with a blit. The game only copies from its backbuffer texture.
			SRB2_ASSERT(false && "copy_framebuffer_to_texture is only supported from offscreen RGB(A) targets");
			return;
		}

		id<MTLBlitCommandEncoder> blit = [cmd_ blitCommandEncoder];
		[blit copyFromTexture:target_.color
				  sourceSlice:0
				  sourceLevel:0
				 sourceOrigin:MTLOriginMake(src_region.x, src_region.y, 0)
				   sourceSize:MTLSizeMake(src_region.w, src_region.h, 1)
					toTexture:dst.texture
			 destinationSlice:0
			 destinationLevel:0
			destinationOrigin:MTLOriginMake(dst_region.x, dst_region.y, 0)];
		[blit endEncoding];
		dst.last_use = current_serial_;
	}
}

void MetalRhi::set_stencil_reference(CullMode face, uint8_t reference)
{
	SRB2_ASSERT(face != CullMode::kNone);
	if (face == CullMode::kFront)
	{
		stencil_front_reference_ = reference;
	}
	else
	{
		stencil_back_reference_ = reference;
	}
}

void MetalRhi::set_stencil_compare_mask(CullMode face, uint8_t mask)
{
	SRB2_ASSERT(face != CullMode::kNone);
	if (face == CullMode::kFront)
	{
		stencil_front_compare_mask_ = mask;
	}
	else
	{
		stencil_back_compare_mask_ = mask;
	}
}

void MetalRhi::set_stencil_write_mask(CullMode face, uint8_t mask)
{
	SRB2_ASSERT(face != CullMode::kNone);
	if (face == CullMode::kFront)
	{
		stencil_front_write_mask_ = mask;
	}
	else
	{
		stencil_back_write_mask_ = mask;
	}
}

void MetalRhi::present()
{
	@autoreleasepool
	{
		flush_pending_clear();
		end_render_encoder();
		if (drawable_ != nil)
		{
			ensure_command_buffer();
			[cmd_ presentDrawable:drawable_];
			drawable_ = nil;
		}
		commit(false);
	}
}

void MetalRhi::finish()
{
	// Gl2Rhi uses this to drop cached framebuffer objects; Metal has none. Submit anything outstanding.
	@autoreleasepool
	{
		if (pass_stack_.empty())
		{
			end_render_encoder();
			commit(false);
		}
	}
}

} // namespace srb2::rhi
