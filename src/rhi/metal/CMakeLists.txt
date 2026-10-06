target_sources(SRB2SDL2 PRIVATE
	metal_rhi.mm
	metal_rhi.hpp
	metal_shaders.cpp
	metal_shaders.hpp
)
set_source_files_properties(metal_rhi.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
find_library(SRB2_METAL_FRAMEWORK Metal REQUIRED)
find_library(SRB2_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
target_link_libraries(SRB2SDL2 PRIVATE ${SRB2_METAL_FRAMEWORK} ${SRB2_QUARTZCORE_FRAMEWORK})
