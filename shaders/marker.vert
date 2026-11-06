#version 450

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition;
    vec4 ambientColorIntensity;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
} scene;

layout(push_constant) uniform MarkerData {
    mat4 model;
    vec4 color;
} marker;

void main() {
    gl_Position = scene.projection * scene.view * marker.model * vec4(inPosition, 1.0);
}
