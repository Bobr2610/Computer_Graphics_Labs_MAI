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
    vec4 material;
} objectData;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D specularMap;
layout(set = 1, binding = 3) uniform sampler2D emissiveMap;

layout(push_constant) uniform SamplingEffect {
    float time;
    float strength;
    uint flags;
    float scale;
} effect;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 texCoord;
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

vec3 linearToSrgb(vec3 linearColor) {
    vec3 low = 12.92 * linearColor;
    vec3 high = 1.055 * pow(max(linearColor, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(high, low, lessThanEqual(linearColor, vec3(0.0031308)));
}

void main() {
    vec3 diffuseColor = texture(albedoMap, texCoord).rgb * objectData.albedo.rgb;
    if ((effect.flags & 1u) != 0u) {
        vec2 animatedOffset = vec2(
            sin(effect.time + texCoord.y * 8.0),
            cos(effect.time * 0.8 + texCoord.x * 10.0)) * 0.045;
        vec3 shiftedSample = texture(albedoMap, texCoord + animatedOffset).rgb;
        vec3 scaledSample = texture(albedoMap, texCoord * effect.scale - animatedOffset).rgb;
        float colorWave = 0.72 + 0.28 * sin(effect.time * 2.0 + (texCoord.x + texCoord.y) * 9.0);
        vec3 layeredColor = (diffuseColor + shiftedSample + scaledSample) / 3.0;
        diffuseColor = mix(diffuseColor, layeredColor, effect.strength) * colorWave;
    }

    vec3 normal = safeNormalize(worldNormal);
    vec3 viewDirection = safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec3 specularColor = texture(specularMap, texCoord).rgb * objectData.material.y;
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

    result += texture(emissiveMap, texCoord).rgb * objectData.material.z;
    if ((effect.flags & 2u) != 0u)
        result = linearToSrgb(result);
    outColor = vec4(result, 1.0);
}
