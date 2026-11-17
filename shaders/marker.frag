#version 450

layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    vec4 albedo;
    vec4 material;
} objectData;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = objectData.albedo;
}
