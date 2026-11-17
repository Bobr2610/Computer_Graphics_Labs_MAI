# Vulkan Computer Graphics Labs

Variant 2 uses the regular octahedron. The application now demonstrates textured surfaces, Blinn–Phong lighting, movable cameras, and the Lab 3 material effects.

## Requirements

- Windows 10 or later
- Visual Studio 2022 with the C++ desktop workload
- CMake 3.25 or later
- Vulkan SDK with `glslc` under `VULKAN_SDK/Bin`

The first CMake configure fetches GLFW, vk-bootstrap, VMA, and Dear ImGui if they are not already cached. `stb_image` is included in `third_party/stb_image.h`; no extra image-library installation is needed.

## Build and run

From this repository directory, run PowerShell:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.321.1'
cmake --preset msvc-debug
cmake --build build-debug --config Debug --parallel
ctest --test-dir build-debug -C Debug --output-on-failure
& '.\build-debug\Debug\vulkan-starter-app.exe'
```

CMake compiles the GLSL files to SPIR-V in the build directory. Texture files are loaded from this source tree's `assets/textures` directory, so the executable can be launched from another working directory while the source assets remain in place.

## Controls

- **Camera:** WASD moves, RMB looks around, and Shift moves faster. Release RMB or press Esc to restore the cursor. Projection and perspective FOV are in the Camera tab.
- **Objects:** edit the main octahedron transform, show or hide the second octahedron and floor, choose a texture independently for each model, and control the existing object animation.
- **Material:** change the texture tint, Blinn–Phong shininess, specular-map strength, emissive-map strength, and the animated multi-sample effect on the main octahedron.
- **Lights:** use **Turn lights off/on** to toggle all illumination while preserving light settings. Select a point or spot light and edit its coordinates, color, intensity, direction, and cone angles. Light markers remain visual helpers and do not determine whether a light illuminates the scene.
- **Reset scene:** restores camera, objects, textures, and lighting defaults.

## Included texture materials

| Material | Albedo | Specular map | Emissive map |
| --- | --- | --- | --- |
| Warm clay | `clay.bmp` | `clay_specular.bmp` | black fallback |
| Night sky | `night.bmp` | `night_specular.bmp` | `night_emissive.bmp` |
| Moss stone | `moss.bmp` | black fallback | `moss_emissive.bmp` |

Albedo and emissive images use sRGB Vulkan formats and are converted to linear values before lighting. Specular data uses a linear UNORM format. When the swapchain format is UNORM, the lit shader applies the linear-to-sRGB transfer before writing the fragment color. Specular maps mask highlights; emissive pixels are added after lighting and do not illuminate nearby objects.

Octahedron UVs are generated per face, duplicating seam vertices so triangles do not interpolate across the wrapped horizontal edge. The floor repeats its UVs. Octahedron, floor, marker, and cone vertex/index buffers are staged into device-local memory. Model and light-helper transforms share one dynamically aligned model-property uniform buffer; the scene camera and light arrays remain in the scene descriptor set.

## Demonstration route

1. In **Objects**, set the main octahedron to **Night sky**, the second octahedron to **Warm clay**, and the floor to **Moss stone**. Each selector changes only its own model.
2. In **Material**, toggle **Animated multi-sample on main object** and watch the moving color modulation on the main octahedron. The effect samples the selected albedo at three time-varying coordinate sets.
3. Return to **Objects** and select **Night sky** for the main octahedron. In **Lights**, press **Turn lights off**. Its emissive star pixels remain visible while the diffuse surface darkens. Turn the lights back on, then set **Specular strength** to zero and back up to compare the night material's mapped highlights.
4. Switch the main octahedron or floor to **Moss stone** and set specular strength above zero. Its black specular fallback removes the mapped highlight; its emissive noise remains visible in darkness, including on the floor.
5. Select **Perspective** and **Orthographic** in the Camera tab, then use WASD and RMB to inspect the UV mapping from different angles.

The validation layer log is written to `build-debug/validation.log` when validation is enabled.
