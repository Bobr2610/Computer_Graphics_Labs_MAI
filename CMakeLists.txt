cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(vulkan-starter-app LANGUAGES C CXX)

add_executable(${PROJECT_NAME} source/main.cpp source/graphics_internal.cpp source/application.cpp source/graphics.cpp)

if(LINUX)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(APPLE)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
elseif(WIN32)
	if(MSVC)
		target_compile_options(${PROJECT_NAME} PRIVATE /W4 /wd4201)
		target_link_options(${PROJECT_NAME} PRIVATE /subsystem:windows /entry:mainCRTStartup)
	else()
		target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
		target_link_options(${PROJECT_NAME} PRIVATE -mwindows)
	endif()
	target_compile_definitions(${PROJECT_NAME} PRIVATE -D_USE_MATH_DEFINES)
else()
	message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} is not supported")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED TRUE
	CXX_EXTENSIONS FALSE
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG 3.5.1
	GIT_SHALLOW TRUE
)

FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
	GIT_TAG v1.4.321
	GIT_SHALLOW TRUE
)

FetchContent_Declare(
	VulkanMemoryAllocator
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
	GIT_TAG v3.4.0
	GIT_SHALLOW TRUE
)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.92.9
	GIT_SHALLOW TRUE
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(
	glfw
	vk-bootstrap
	VulkanMemoryAllocator
	imgui
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	glfw
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	GPUOpen::VulkanMemoryAllocator
)

target_include_directories(${PROJECT_NAME} PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_sources(${PROJECT_NAME} PRIVATE
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
set(_SHADER_BINARIES)
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")
target_compile_definitions(${PROJECT_NAME} PRIVATE OCTAHEDRON_SHADER_DIR="${SHADER_OUTPUT_DIR}")
target_compile_definitions(${PROJECT_NAME} PRIVATE LAB_VALIDATION_LOG="${CMAKE_CURRENT_BINARY_DIR}/validation.log")

foreach(SHADER_FILE IN ITEMS octahedron.vert octahedron.frag)
	set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/shaders/${SHADER_FILE})
	set(SHADER_BINARY ${SHADER_FILE}.spv)
	set(SHADER_BINARY_PATH ${SHADER_OUTPUT_DIR}/${SHADER_BINARY})

	add_custom_command(
		OUTPUT ${SHADER_BINARY_PATH}
		COMMAND "${GLSLC_EXECUTABLE}" ${SHADER_SOURCE} -o ${SHADER_BINARY_PATH}
		DEPENDS ${SHADER_SOURCE}
		COMMENT "Compiling ${SHADER_FILE} shader"
		VERBATIM
	)

	list(APPEND _SHADER_BINARIES ${SHADER_BINARY_PATH})
endforeach()
add_custom_target(shaders DEPENDS ${_SHADER_BINARIES})
add_dependencies(${PROJECT_NAME} shaders)

include(CTest)
if(BUILD_TESTING)
	add_executable(geometry-tests tests/geometry_tests.cpp)
	target_include_directories(geometry-tests PRIVATE source)
	target_compile_features(geometry-tests PRIVATE cxx_std_20)
	add_test(NAME geometry-and-projections COMMAND geometry-tests)
endif()
