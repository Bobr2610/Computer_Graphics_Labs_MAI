#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

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

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 texCoord;

void main() {
    vec4 world = objectData.model * vec4(inPosition, 1.0);
    gl_Position = scene.projection * scene.view * world;
    worldPosition = world.xyz;
    worldNormal = normalize(transpose(inverse(mat3(objectData.model))) * inNormal);
    texCoord = inTexCoord;
}
