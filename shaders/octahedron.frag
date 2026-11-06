#version 450

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 coneCosines;
};

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view;
    mat4 projection;
    vec4 cameraPosition;
    vec4 ambientColorIntensity;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
} scene;

layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 material;
} objectData;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

vec3 safeNormalize(vec3 value) {
    float squaredLength = dot(value, value);
    return squaredLength > 0.000001 ? value * inversesqrt(squaredLength) : vec3(0.0);
}

vec3 evaluateBlinnPhong(vec3 normal, vec3 viewDirection, vec3 lightDirection,
                        vec3 radiance, vec3 diffuseColor, vec3 specularColor,
                        float shininess) {
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    vec3 halfDirection = safeNormalize(lightDirection + viewDirection);
    float specularFactor = diffuseFactor > 0.0
        ? pow(max(dot(normal, halfDirection), 0.0), shininess)
        : 0.0;
    return radiance * (diffuseColor * diffuseFactor + specularColor * specularFactor);
}

void main() {
    vec3 normal = safeNormalize(worldNormal);
    vec3 viewDirection = safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec3 diffuseColor = objectData.albedo.rgb * mix(vec3(1.0), vertexColor, objectData.material.y);
    vec3 specularColor = objectData.specularColor.rgb;
    float shininess = max(objectData.material.x, 1.0);

    vec3 result = diffuseColor * scene.ambientColorIntensity.rgb * scene.ambientColorIntensity.a;
    vec3 directional = safeNormalize(scene.directionalDirection.xyz);
    result += evaluateBlinnPhong(normal, viewDirection, directional,
        scene.directionalColorIntensity.rgb * scene.directionalColorIntensity.a,
        diffuseColor, specularColor, shininess);

    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        vec3 toLight = pointLights[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        vec3 radiance = pointLights[i].colorIntensity.rgb
            * pointLights[i].colorIntensity.a / distanceSquared;
        result += evaluateBlinnPhong(normal, viewDirection, lightDirection,
            radiance, diffuseColor, specularColor, shininess);
    }

    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        vec3 fromLight = worldPosition - spotLights[i].position.xyz;
        float distanceSquared = max(dot(fromLight, fromLight), 0.0001);
        vec3 lightDirection = -fromLight * inversesqrt(distanceSquared);
        float coneCosine = dot(safeNormalize(fromLight), safeNormalize(spotLights[i].direction.xyz));
        float coneFactor = smoothstep(spotLights[i].coneCosines.y,
                                     spotLights[i].coneCosines.x, coneCosine);
        vec3 radiance = spotLights[i].colorIntensity.rgb
            * spotLights[i].colorIntensity.a * coneFactor / distanceSquared;
        result += evaluateBlinnPhong(normal, viewDirection, lightDirection,
            radiance, diffuseColor, specularColor, shininess);
    }

    outColor = vec4(result, 1.0);
}
