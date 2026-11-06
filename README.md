# Vulkan Camera and Lighting Lab

Variant 2 uses the regular octahedron from Lab 1. This application adds a movable camera and real-time Blinn-Phong lighting with directional, point, and spot lights.

## Requirements

- Windows 10 or later
- Visual Studio 2022 with the C++ desktop workload
- CMake 3.25 or later
- Vulkan SDK with `glslc` available under `VULKAN_SDK/Bin`

The first CMake configure downloads GLFW, vk-bootstrap, VMA, and Dear ImGui with `FetchContent`. Network access is needed unless those sources are already cached.

## Build and run

From this repository directory, run PowerShell:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.321.1'
cmake --preset msvc-debug
cmake --build build-debug --config Debug --parallel
& '.\build-debug\Debug\vulkan-starter-app.exe'
```

CMake compiles the GLSL shaders to SPIR-V in the build directory. The executable can be launched from a different working directory.

## Controls

- **Camera:** use WASD while the application is focused to move, hold RMB to look around, and hold Shift to move faster. Release RMB or press Esc to restore the cursor. Projection and field of view are set in the Camera tab.
- **Octahedron:** model position, rotation, scale, vertex-color tint, animation, and the second object.
- **Material:** albedo, specular color, and Blinn-Phong shininess coefficient.
- **Ambient and directional light:** ambient color and intensity, plus directional-light direction, color, and intensity.
- **Point lights:** add or remove lights; select a light in the list and enter its X/Y/Z coordinates, color, and intensity.
- **Spot lights:** add or remove lights; select a light in the list and enter its position and direction, or use Aim at object. Color, intensity, and inner/outer cone angles are editable.
- **Visualization:** show or hide unlit light markers and spot-cone outlines.

Point and spot lights use inverse-square attenuation. Spotlights have a smooth transition between their inner and outer cone angles. Light arrays use runtime-sized storage buffers; the application respects the storage-buffer limit reported by the graphics device.

The validation layer log is written to `build-debug/validation.log` when validation is enabled.
