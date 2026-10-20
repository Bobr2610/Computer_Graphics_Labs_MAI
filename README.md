# Vulkan Octahedron — Computer Graphics Lab 1

The application implements variant 2: a correctly indexed octahedron rendered
with Vulkan. It demonstrates perspective and orthographic projection, model
transforms, animation, per-vertex color, color tinting, and multiple objects
with separate descriptor sets.

## Requirements

- Windows 10 or later
- Visual Studio 2022 with the C++ desktop workload
- CMake 3.25 or later
- Vulkan SDK with `glslc` available under `VULKAN_SDK/Bin`

The first CMake configure downloads GLFW, vk-bootstrap, VMA, and Dear ImGui
with `FetchContent`; network access is needed unless those sources are already
cached by CMake.

## Configure and build

From this repository directory, run PowerShell:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.321.1'
cmake --preset msvc-debug
cmake --build build-debug --config Debug --parallel
```

The preset creates a Visual Studio 2022 build in `build-debug`. Other presets
use separate directories, so switching generators does not reuse a cache made
by another generator.

## Run

```powershell
& '.\build-debug\Debug\vulkan-starter-app.exe'
```

CMake compiles the GLSL shaders into the build directory and passes their
absolute path to the application. The working directory can therefore be
different from the repository directory.

## Controls

- `Projection` selects `Perspective` or `Orthographic`.
- `Position`, `Rotation (degrees)`, and `Scale` change the first object.
- `Vertex color tint` multiplies the colors computed for its vertices.
- `Play animation`, `Path radius`, `Path height`, and `Animation speed` control
  the first object's motion and rotation.
- `Show second object` toggles the second octahedron; `Reset scene` restores
  the initial parameters.

The validation layer output is written to `build-debug/validation.log` when
validation is enabled by the application.
