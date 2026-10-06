// DR. ROBOTNIK'S RING RACERS
//-----------------------------------------------------------------------------
// Copyright (C) 2025 by Kart Krew
//
// This program is free software distributed under the
// terms of the GNU General Public License, version 2.
// See the 'LICENSE' file for more details.
//-----------------------------------------------------------------------------

#ifndef SRB2_SDL_RHI_METAL_PLATFORM_HPP
#define SRB2_SDL_RHI_METAL_PLATFORM_HPP

#include "../rhi/metal/metal_rhi.hpp"
#include "../rhi/rhi.hpp"

#include <SDL3/SDL.h>

namespace srb2::rhi
{

struct SdlMetalPlatform final : public MetalPlatform
{
	SDL_Window* window = nullptr;
	SDL_MetalView view = nullptr;

	SdlMetalPlatform(SDL_Window* window);
	virtual ~SdlMetalPlatform();

	virtual void* metal_layer() override;
	virtual Rect get_default_framebuffer_dimensions() override;
};

} // namespace srb2::rhi

#endif // SRB2_SDL_RHI_METAL_PLATFORM_HPP
