#version 450

layout(push_constant) uniform MarkerData {
    mat4 model;
    vec4 color;
} marker;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = marker.color;
}
