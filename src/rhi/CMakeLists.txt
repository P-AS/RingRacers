target_sources(SRB2SDL2 PRIVATE
	handle.hpp
	rhi.cpp
	rhi.hpp
	shader_load_context.cpp
	shader_load_context.hpp
)

if(CMAKE_SYSTEM_NAME MATCHES "iOS")
	add_subdirectory(metal)
else()
	add_subdirectory(gl2)
endif()
