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

layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    vec4 albedo;
    vec4 material;
} objectData;

void main() {
    gl_Position = scene.projection * scene.view * objectData.model * vec4(inPosition, 1.0);
}
