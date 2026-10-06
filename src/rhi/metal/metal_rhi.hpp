// DR. ROBOTNIK'S RING RACERS
//-----------------------------------------------------------------------------
// Copyright (C) 2025 by Kart Krew
//
// This program is free software distributed under the
// terms of the GNU General Public License, version 2.
// See the 'LICENSE' file for more details.
//-----------------------------------------------------------------------------

#ifndef SRB2_RHI_METAL_RHI_HPP
#define SRB2_RHI_METAL_RHI_HPP

#include <memory>

#include "../rhi.hpp"

namespace srb2::rhi
{

/// @brief Platform-specific implementation details for the Metal backend.
struct MetalPlatform
{
	virtual ~MetalPlatform();

	/// @brief The CAMetalLayer the default render pass draws into.
	virtual void* metal_layer() = 0;
	virtual Rect get_default_framebuffer_dimensions() = 0;
};

/// @brief Creates the Metal backend. The implementation is Objective-C++, so it is hidden behind this factory.
std::unique_ptr<Rhi> create_metal_rhi(std::unique_ptr<MetalPlatform>&& platform);

} // namespace srb2::rhi

#endif // SRB2_RHI_METAL_RHI_HPP
