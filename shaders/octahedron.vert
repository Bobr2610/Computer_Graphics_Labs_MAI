#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inVertexColor;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view;
    mat4 projection;
} scene;

layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    vec4 tint;
} objectData;

layout(location = 0) out vec3 vertexColor;

void main() {
    gl_Position = scene.projection * scene.view * objectData.model * vec4(inPosition, 1.0);
    vertexColor = inVertexColor * objectData.tint.rgb;
}