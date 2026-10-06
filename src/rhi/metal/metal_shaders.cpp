// DR. ROBOTNIK'S RING RACERS
//-----------------------------------------------------------------------------
// Copyright (C) 2025 by Kart Krew
//
// This program is free software distributed under the
// terms of the GNU General Public License, version 2.
// See the 'LICENSE' file for more details.
//-----------------------------------------------------------------------------

// Metal Shading Language ports of the rhi_glsl_* shaders found in shaders.pk3.
// These are compiled at runtime by the Metal RHI, with the RHI program defines
// (ENABLE_VA_TEXCOORD0, ENABLE_S_SAMPLER2, ...) passed as preprocessor macros.
// Keep in sync with the GLSL sources.

#include "metal_shaders.hpp"

namespace srb2::rhi
{

const char* const kMetalShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Must match the layout of MetalUniforms in metal_rhi.mm.
struct Uniforms
{
	float4x4 u_projection;
	float4x4 u_modelview;
	float3x3 u_texcoord0_transform;
	float2 u_texcoord0_min;
	float2 u_texcoord0_max;
	float2 u_sampler0_size;
	float u_time;
	int u_sampler0_is_indexed_alpha;
	int u_wipe_colorize_mode;
	int u_wipe_encore_swizzle;
	int u_postimg_water;
	int u_postimg_heat;
	float rhi_flip_y; // set by the backend: -1 when rendering to a texture, 1 for the drawable
};

struct VertexIn
{
	float3 a_position [[attribute(0)]];
#ifdef ENABLE_VA_TEXCOORD0
	float2 a_texcoord0 [[attribute(1)]];
#endif
#ifdef ENABLE_VA_COLOR
	float4 a_color [[attribute(2)]];
#endif
};

struct Varyings
{
	float4 position [[position]];
	float point_size [[point_size]];
	float2 v_texcoord0;
	float4 v_color;
};

// GLSL mod() floors; MSL fmod() truncates.
template <typename T>
static inline T glsl_mod(T x, T y)
{
	return x - y * floor(x / y);
}

// Converts GL clip space to Metal clip space.
// Y is flipped for offscreen targets so textures keep GL's bottom-up row order in memory.
// Depth is remapped from [-1, 1] to [0, 1].
static inline float4 rhi_clip_position(float4 pos, constant Uniforms& u)
{
	pos.y *= u.rhi_flip_y;
	pos.z = (pos.z + pos.w) * 0.5;
	return pos;
}

// rhi_glsl_vertex_{unshaded,unshadedpaletted,sharpbilinear,crt,crtsharp,postimg}
vertex Varyings generic_vs(VertexIn in [[stage_in]], constant Uniforms& u [[buffer(0)]])
{
	Varyings out;
	out.position = rhi_clip_position(u.u_projection * u.u_modelview * float4(in.a_position, 1.0), u);
	out.point_size = 1.0;

#ifdef ENABLE_VA_COLOR
	out.v_color = in.a_color;
#else
	out.v_color = float4(1.0);
#endif

#ifdef ENABLE_VA_TEXCOORD0
	float3 texcoord0 = float3(in.a_texcoord0, 1.0);
#else
	float3 texcoord0 = float3(0.0, 0.0, 1.0);
#endif
	texcoord0 = u.u_texcoord0_transform * texcoord0;
	out.v_texcoord0 = texcoord0.xy / texcoord0.z;
	return out;
}

// rhi_glsl_vertex_postprocesswipe
vertex Varyings postprocesswipe_vs(VertexIn in [[stage_in]], constant Uniforms& u [[buffer(0)]])
{
	Varyings out;
	out.position = rhi_clip_position(u.u_projection * float4(in.a_position, 1.0), u);
	out.point_size = 1.0;
	out.v_color = float4(1.0);
#ifdef ENABLE_VA_TEXCOORD0
	out.v_texcoord0 = in.a_texcoord0;
#else
	out.v_texcoord0 = float2(0.0);
#endif
	return out;
}

// rhi_glsl_fragment_unshaded
fragment float4 unshaded_fs(
	Varyings in [[stage_in]],
	texture2d<float> s_sampler0 [[texture(0)]],
	sampler s_sampler0_s [[sampler(0)]]
)
{
	float4 color = s_sampler0.sample(s_sampler0_s, in.v_texcoord0) * in.v_color;
	if (color.a < 0.01)
	{
		discard_fragment();
	}
	return color;
}

// rhi_glsl_fragment_unshadedpaletted
fragment float4 unshadedpaletted_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]], // luminance texture indexing palette
	texture2d<float> s_sampler1 [[texture(1)]], // palette, 256x1
#ifdef ENABLE_S_SAMPLER2
	texture2d<float> s_sampler2 [[texture(2)]], // colormap, 256x1 red
#endif
	sampler s_sampler0_s [[sampler(0)]],
	sampler s_sampler1_s [[sampler(1)]]
#ifdef ENABLE_S_SAMPLER2
	, sampler s_sampler2_s [[sampler(2)]]
#endif
)
{
	float4 index_color = s_sampler0.sample(s_sampler0_s, in.v_texcoord0);

	float index = 0.0;
#ifdef ENABLE_S_SAMPLER2
	index = s_sampler2.sample(s_sampler2_s, float2(index_color.r, in.v_texcoord0.y)).r;
#else
	index = index_color.r;
#endif
	index = floor((index * 255.0) + 0.5);

	float4 color = s_sampler1.sample(s_sampler1_s, float2(index / 255.0, 0.0));

#ifdef ENABLE_U_SAMPLER0_IS_INDEXED_ALPHA
	if (u.u_sampler0_is_indexed_alpha != 0)
	{
		// Luminance-alpha textures are swizzled to (L, L, L, A), as in legacy GL.
		color = color * float4(1.0, 1.0, 1.0, index_color.a) * in.v_color;
	}
	else
	{
		color = color * in.v_color;
	}
#else
	color = color * in.v_color;
#endif

	if (color.a < 0.01)
	{
		discard_fragment();
	}
	return color;
}

// rhi_glsl_fragment_sharpbilinear
fragment float4 sharpbilinear_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]],
	sampler s_sampler0_s [[sampler(0)]]
)
{
	const float scale = 4.0;
	float2 texel = in.v_texcoord0 * u.u_sampler0_size;
	float2 texel_floored = floor(texel);
	float2 s = fract(texel);
	float region_range = 0.5 - 0.5 / scale;
	float2 center_dist = s - float2(0.5, 0.5);
	float2 f = (center_dist - clamp(center_dist, -region_range, region_range)) * scale + 0.5;

	float2 mod_texel = texel_floored + f;

	float4 color = s_sampler0.sample(s_sampler0_s, mod_texel / u.u_sampler0_size) * in.v_color;
	if (color.a < 0.01)
	{
		discard_fragment();
	}
	return color;
}

// rhi_glsl_fragment_crt and rhi_glsl_fragment_crtsharp
// Copyright (C) Sally
// RHI port by Eidolon.
// Mercilessly hacked at for ASPECT RATIO FUCKERY by Tyron + Frey.

#define CRT_WIDTH 320.0
#define CRT_HEIGHT 200.0
#define CRT_BRIGHT 1.0
#define CRT_CONTRAST 70.0

#define BLUR_PI 6.28318530718
#define BLUR_DIRS 9.0
#define BLUR_INC (BLUR_PI / BLUR_DIRS)
#define BLUR_DIR_SCALAR (9.0 / BLUR_DIRS)
#define BLUR_QUALITY 2.0
#define BLUR_QUALITY_INC (1.0 / BLUR_QUALITY)

#define SMUDGE_CONTRAST 0.65
#define SCANLINE_INTENSITY 0.8

struct CrtParams
{
	float dotscale;
	float blur_dist;
	float smudge_dist;
};

struct CrtTextures
{
	texture2d<float> image;
	sampler image_s;
	texture2d<float> pattern;
	sampler pattern_s;
	float2 sampler0_size;
};

static float crt_brightness(float3 color)
{
	return (0.299 * color.r) + (0.587 * color.g) + (0.114 * color.b);
}

static float3 crt_img_color(thread const CrtTextures& t, CrtParams p, float2 input_uv, float2 img_size)
{
	float2 image_uv = (floor((input_uv * img_size) + 0.5)) / img_size;
	float3 image_color = t.image.sample(t.image_s, image_uv).rgb;

	float contrast_mod = clamp(crt_brightness(image_color) * CRT_CONTRAST, 0.0, 1.0);
	image_color *= contrast_mod;

	float2 scale_uv = float2(t.sampler0_size.x / t.sampler0_size.y, 1.0); // Keep dot pattern aspect-correct!
	float pattern_scale = p.dotscale;
	float2 pattern_uv = input_uv * scale_uv * pattern_scale;
	float3 pattern_color = t.pattern.sample(t.pattern_s, pattern_uv).rgb;

	return image_color * pattern_color * CRT_BRIGHT;
}

static float3 crt_blur(thread const CrtTextures& t, CrtParams p, float2 uv, float2 img_size, float blur_scale)
{
	float3 acc = float3(0.0);
	for (float d = 0.0; d < BLUR_PI; d += BLUR_INC)
	{
		for (float i = BLUR_QUALITY_INC; i <= 1.0; i += BLUR_QUALITY_INC)
		{
			float2 blur_offset = float2(cos(d), sin(d) * 0.5) * (p.blur_dist * blur_scale) * i;
			float2 blur_uv = uv + (blur_offset / img_size.x);
			acc += crt_img_color(t, p, blur_uv, img_size) * BLUR_QUALITY_INC * BLUR_DIR_SCALAR;
		}
	}
	return acc;
}

static float4 crt_main(thread const CrtTextures& t, CrtParams p, float2 uv)
{
	float2 resolution = t.sampler0_size;
	float2 img_size = float2(((resolution.x * 400.0) / resolution.y), 400.0);
	float blur_scale = 1920.0 / 1080.0;

	// Blur layer
	float3 final_color = crt_img_color(t, p, uv, img_size) + crt_blur(t, p, uv, img_size, blur_scale);

	// Smudge layer
	float2 overlay_offset = float2(-(p.smudge_dist * blur_scale) / img_size);
	float2 overlay_uv = uv + overlay_offset;
	float3 overlay_color = crt_img_color(t, p, overlay_uv, img_size) + crt_blur(t, p, overlay_uv, img_size, blur_scale);

	// Increase overlay contrast
	overlay_color *= (overlay_color * SMUDGE_CONTRAST);

	// Combine the main and overlay
	final_color = mix(final_color, overlay_color, 0.5);

	// Apply scanlines
	float scan_pos = glsl_mod(uv.y * CRT_HEIGHT, 1.0);
	scan_pos = clamp(scan_pos, 0.0, 1.0);

	float scanline = abs(scan_pos - 0.5) * 2.0;
	scanline *= 1.0 - crt_brightness(final_color);
	scanline = clamp(scanline, 0.0, 1.0);
	final_color = mix(final_color, float3(0.0), scanline * SCANLINE_INTENSITY);

	return float4(final_color, 1.0);
}

fragment float4 crt_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]],
	texture2d<float> s_sampler1 [[texture(1)]], // dot pattern 12x4
	sampler s_sampler0_s [[sampler(0)]],
	sampler s_sampler1_s [[sampler(1)]]
)
{
	CrtTextures t = {s_sampler0, s_sampler0_s, s_sampler1, s_sampler1_s, u.u_sampler0_size};
	CrtParams p = {CRT_WIDTH * 2.0, 1.5, 0.75};
	return crt_main(t, p, in.v_texcoord0);
}

fragment float4 crtsharp_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]],
	texture2d<float> s_sampler1 [[texture(1)]], // dot pattern 12x4
	sampler s_sampler0_s [[sampler(0)]],
	sampler s_sampler1_s [[sampler(1)]]
)
{
	CrtTextures t = {s_sampler0, s_sampler0_s, s_sampler1, s_sampler1_s, u.u_sampler0_size};
	CrtParams p = {CRT_WIDTH * 4.0, 0.5, 0.35};
	return crt_main(t, p, in.v_texcoord0);
}

// rhi_glsl_fragment_postimg
fragment float4 postimg_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]], // screen
#ifdef ENABLE_S_SAMPLER1
	texture2d<float> s_sampler1 [[texture(1)]], // palette
	sampler s_sampler1_s [[sampler(1)]],
#endif
	sampler s_sampler0_s [[sampler(0)]]
)
{
	float2 texcoord0 = in.v_texcoord0;
	float2 sampler0_uvsize = u.u_texcoord0_max - u.u_texcoord0_min;

	int reclamp_uvs = 0;
	if (u.u_postimg_water > 0)
	{
		texcoord0.x += (sin((u.u_time / 35.0) * 2.0 + ((texcoord0.y - u.u_texcoord0_min.y) / sampler0_uvsize.y) * 20.0) * 0.01);
		reclamp_uvs = 1;
	}
	if (u.u_postimg_heat > 0)
	{
		texcoord0.x += (max(sin((u.u_time / 35.0) * 1.2 - ((texcoord0.y - u.u_texcoord0_min.y) / sampler0_uvsize.y) * 5.0) - 0.995, 0.0));
		texcoord0.x -= (max(sin(((u.u_time + 24.0) / 35.0) * 0.2 - ((texcoord0.y - u.u_texcoord0_min.y) / sampler0_uvsize.y) * 1.5) - 0.990, 0.0));
		reclamp_uvs = 1;
	}

	if (reclamp_uvs > 0)
	{
		texcoord0 = max(min(texcoord0, u.u_texcoord0_max - float2(0.0001)), u.u_texcoord0_min);
	}

#ifdef ENABLE_S_SAMPLER1
	float4 index_color = s_sampler0.sample(s_sampler0_s, texcoord0);
	return s_sampler1.sample(s_sampler1_s, float2(floor((index_color.r * 255.0) + 0.5) / 255.0, 0.0));
#else
	return s_sampler0.sample(s_sampler0_s, texcoord0);
#endif
}

// rhi_glsl_fragment_postprocesswipe
static float4 wipe_modulate(float4 start, float4 end, float mask_value)
{
	return mix(start, end, mask_value);
}

static float4 wipe_invert(float4 start, float mask_value)
{
	return mix(start, float4(float3(1.0) - start.rgb, 1.0), mask_value);
}

static float4 wipe_megadrive_to_black(float4 start, float mask_value)
{
	float fade_alpha = clamp(mask_value, 0.0, 1.0);
	float rsub = clamp(fade_alpha * 3.0, 0.0, 3.0);
	float r = start.r - rsub;
	float gsub = clamp(-r, 0.0, 3.0);
	float g = start.g - gsub;
	float bsub = clamp(-g, 0.0, 3.0);
	float b = start.b - bsub;
	return float4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}

static float4 wipe_megadrive_to_white(float4 start, float mask_value)
{
	float fade_alpha = clamp(mask_value, 0.0, 1.0);
	float radd = clamp(fade_alpha * 3.0, 0.0, 3.0);
	float r = min(start.r + radd, 1.0);
	float gadd = max(radd - 1.0, 0.0);
	float g = min(start.g + gadd, 1.0);
	float badd = max(gadd - 1.0, 0.0);
	float b = min(start.b + badd, 1.0);
	return float4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}

fragment float4 postprocesswipe_fs(
	Varyings in [[stage_in]],
	constant Uniforms& u [[buffer(0)]],
	texture2d<float> s_sampler0 [[texture(0)]], // wipe start
	texture2d<float> s_sampler1 [[texture(1)]], // wipe end
	texture2d<float> s_sampler2 [[texture(2)]], // mask texture
	sampler s_sampler0_s [[sampler(0)]],
	sampler s_sampler1_s [[sampler(1)]],
	sampler s_sampler2_s [[sampler(2)]]
)
{
	float2 coord = in.v_texcoord0;
	float wipe_encore_swizzle = float(u.u_wipe_encore_swizzle);
	float y_baseres = floor(coord.y * 200.0);
	float x_phase = wipe_encore_swizzle * 128.0 + (y_baseres * wipe_encore_swizzle * 8.0);
	float screen_y_fac = glsl_mod(-floor(y_baseres), 2.0) * 2.0 - 1.0;
	coord.x = coord.x + (screen_y_fac * sin(x_phase) * wipe_encore_swizzle * 8.0) / 320.0;

	float4 start = s_sampler0.sample(s_sampler0_s, coord);
	float4 end = s_sampler1.sample(s_sampler1_s, coord);

	// Fade mask luminance values range from 0 to 32 (exclusive)
	// A value of 31 = fully starting image (fade mask of 0)
	float fade_mask = 1.0 - (s_sampler2.sample(s_sampler2_s, in.v_texcoord0).r * 255.0 / 31.0);

	float4 final_color = float4(0.0, 0.0, 0.0, 1.0);
	if (u.u_wipe_colorize_mode == 0)
	{
		final_color = wipe_modulate(start, end, fade_mask);
	}
	else if (u.u_wipe_colorize_mode == 1)
	{
		final_color = wipe_invert(start, fade_mask);
	}
	else if (u.u_wipe_colorize_mode == 2)
	{
		final_color = wipe_megadrive_to_black(start, fade_mask);
	}
	else if (u.u_wipe_colorize_mode == 3)
	{
		final_color = wipe_megadrive_to_white(start, fade_mask);
	}
	return final_color;
}
)MSL";

} // namespace srb2::rhi
