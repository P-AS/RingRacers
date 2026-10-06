// DR. ROBOTNIK'S RING RACERS
//-----------------------------------------------------------------------------
// Copyright (C) 2025 by Kart Krew
//
// This program is free software distributed under the
// terms of the GNU General Public License, version 2.
// See the 'LICENSE' file for more details.
//-----------------------------------------------------------------------------

#include "rhi_metal_platform.hpp"

#include <stdexcept>

#include <SDL3/SDL.h>

#include "../cxxutil.hpp"

using namespace srb2;
using namespace srb2::rhi;

SdlMetalPlatform::SdlMetalPlatform(SDL_Window* window) : window(window)
{
	SRB2_ASSERT(window != nullptr);
	view = SDL_Metal_CreateView(window);
	if (view == nullptr)
	{
		throw std::runtime_error(SDL_GetError());
	}
}

SdlMetalPlatform::~SdlMetalPlatform()
{
	if (view != nullptr)
	{
		SDL_Metal_DestroyView(view);
	}
}

void* SdlMetalPlatform::metal_layer()
{
	return SDL_Metal_GetLayer(view);
}

rhi::Rect SdlMetalPlatform::get_default_framebuffer_dimensions()
{
	SRB2_ASSERT(window != nullptr);
	int w;
	int h;
	SDL_GetWindowSizeInPixels(window, &w, &h);
	return {0, 0, static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}
