#include "application.hpp"
#include "graphics.hpp"
#include "math.hpp"
#include "octahedron.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <vulkan/vulkan.h>

namespace application {
namespace {

using geometry::Vertex;
using graphics::createHostBuffer;
using graphics::destroyBuffer;
using graphics::GpuBuffer;
using graphics::uploadBuffer;
using math::Mat4;
using math::Vec3;

constexpr uint32_t index_count = static_cast<uint32_t>(geometry::octahedronIndices.size());
constexpr uint32_t object_count = 3;
constexpr uint32_t main_object_index = 0;
constexpr uint32_t companion_object_index = 1;
constexpr uint32_t coneSegments = 32;

enum class ProjectionMode : int {
	perspective = 0,
	orthographic = 1,
};

struct alignas(16) SceneUniform {
	Mat4 view;
	Mat4 projection;
	float cameraPosition[4];
	float ambientColorIntensity[4];
	float directionalDirection[4];
	float directionalColorIntensity[4];
	uint32_t lightCounts[4];
};

struct alignas(16) ObjectUniform {
	Mat4 model;
	float albedo[4];
	float specularColor[4];
	float material[4];
};

struct alignas(16) PointLightGpu {
	float position[4];
	float colorIntensity[4];
};

struct alignas(16) SpotLightGpu {
	float position[4];
	float direction[4];
	float colorIntensity[4];
	float coneCosines[4];
};

struct alignas(16) MarkerPush {
	Mat4 model;
	float color[4];
};

struct MarkerVertex {
	float position[3];
};

static_assert(sizeof(SceneUniform) == 208);
static_assert(sizeof(ObjectUniform) == 112);
static_assert(sizeof(PointLightGpu) == 32);
static_assert(sizeof(SpotLightGpu) == 64);
static_assert(sizeof(MarkerPush) == 80);
static_assert(offsetof(SceneUniform, lightCounts) == 192);
static_assert(offsetof(ObjectUniform, material) == 96);

struct ObjectResources {
	GpuBuffer uniformBuffer;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct PointLight {
	std::array<float, 3> position{1.8f, 1.2f, 2.0f};
	std::array<float, 3> color{1.0f, 0.72f, 0.42f};
	float intensity = 3.0f;
};

struct SpotLight {
	std::array<float, 3> position{-1.8f, 2.2f, 2.6f};
	std::array<float, 3> direction{0.35f, -0.45f, -0.82f};
	std::array<float, 3> color{0.38f, 0.62f, 1.0f};
	float intensity = 7.0f;
	float innerAngleDegrees = 18.0f;
	float outerAngleDegrees = 28.0f;
};

struct Camera {
	ProjectionMode projection = ProjectionMode::perspective;
	std::array<float, 3> position{2.8f, 2.2f, 5.5f};
	float yaw = -27.0f;
	float pitch = -20.0f;
	float fieldOfView = 50.0f;
};

struct Model {
	std::array<float, 3> position{};
	std::array<float, 3> rotation{0.0f, 20.0f, 0.0f};
	std::array<float, 3> scale{0.9f, 0.9f, 0.9f};
	std::array<float, 3> tint{1.0f, 1.0f, 1.0f};
	bool showCompanion = true;
	bool showFloor = true;
};

struct Material {
	std::array<float, 3> albedo{0.45f, 0.55f, 0.65f};
	std::array<float, 3> specular{0.18f, 0.18f, 0.18f};
	float shininess = 48.0f;
	bool vertexColors = false;
};

struct Lighting {
	std::array<float, 3> ambientColor{1.0f, 1.0f, 1.0f};
	float ambientIntensity = 0.08f;
	std::array<float, 3> direction{0.4f, 0.75f, 0.52f};
	std::array<float, 3> color{1.0f, 0.96f, 0.88f};
	float intensity = 0.6f;
	std::vector<PointLight> points{PointLight{}};
	std::vector<SpotLight> spots{SpotLight{}};
	int mode = 0;
	bool showSources = true;
};

struct Animation {
	double time = 0.0;
	double previousTime = 0.0;
	bool hasPreviousTime = false;
	bool playing = false;
	float radius = 0.6f;
	float height = 0.25f;
	float speed = 1.0f;
};

struct SceneState {
	Camera camera;
	Model model;
	Material material;
	Lighting lighting;
	Animation animation;
};

enum class LightKind { none, point, spot };
struct LightSelection {
	LightKind kind = LightKind::none;
	size_t index = 0;
};

struct Interaction {
	LightSelection selected;
	bool looking = false;
	double cursorX = 0.0;
	double cursorY = 0.0;
	double previousTime = 0.0;
};

Interaction interaction;
GLFWwindow *inputWindow = nullptr;

SceneState scene;
float sidebarRight = 360.0f;
bool lightsDirty = true;
uint32_t maxPointLights = 0;
uint32_t maxSpotLights = 0;
std::string lightMessage;
std::vector<PointLightGpu> pointData;
std::vector<SpotLightGpu> spotData;

GpuBuffer sceneUniformBuffer;
GpuBuffer vertexBuffer;
GpuBuffer indexBuffer;
GpuBuffer pointLightBuffer;
GpuBuffer spotLightBuffer;
GpuBuffer markerVertexBuffer;
GpuBuffer markerIndexBuffer;
GpuBuffer coneVertexBuffer;
uint32_t markerIndexCount = 0;
uint32_t coneVertexCount = 0;
ObjectResources objects[object_count];

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout objectSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;
VkPipelineLayout markerPipelineLayout = VK_NULL_HANDLE;
VkPipeline markerPipeline = VK_NULL_HANDLE;
VkPipeline conePipeline = VK_NULL_HANDLE;

uint32_t sceneViewportX(uint32_t framebufferWidth) {
	const float scale = ImGui::GetIO().DisplayFramebufferScale.x;
	return std::min(static_cast<uint32_t>(sidebarRight * scale), framebufferWidth - 1);
}

Vec3 fromArray(const std::array<float, 3> &value) {
	return {value[0], value[1], value[2]};
}

Mat4 directionMatrix(const Vec3 &position, const Vec3 &direction, float radius, float length) {
	const Vec3 forward = math::normalize(direction);
	const Vec3 referenceUp = std::abs(forward.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
	const Vec3 right = math::normalize(math::cross(referenceUp, forward));
	const Vec3 up = math::cross(forward, right);
	Mat4 result{};
	result.at(0, 0) = right.x * radius;
	result.at(1, 0) = right.y * radius;
	result.at(2, 0) = right.z * radius;
	result.at(0, 1) = up.x * radius;
	result.at(1, 1) = up.y * radius;
	result.at(2, 1) = up.z * radius;
	result.at(0, 2) = forward.x * length;
	result.at(1, 2) = forward.y * length;
	result.at(2, 2) = forward.z * length;
	result.at(0, 3) = position.x;
	result.at(1, 3) = position.y;
	result.at(2, 3) = position.z;
	result.at(3, 3) = 1.0f;
	return result;
}

std::vector<MarkerVertex> makeSphereVertices(uint32_t rings, uint32_t sectors,
											 std::vector<uint16_t> &indices) {
	std::vector<MarkerVertex> vertices;
	vertices.reserve((rings + 1) * (sectors + 1));
	for (uint32_t ring = 0; ring <= rings; ++ring) {
		const float v = static_cast<float>(ring) / static_cast<float>(rings);
		const float latitude = math::pi * v;
		for (uint32_t sector = 0; sector <= sectors; ++sector) {
			const float u = static_cast<float>(sector) / static_cast<float>(sectors);
			const float longitude = 2.0f * math::pi * u;
			vertices.push_back({{
				std::sin(latitude) * std::cos(longitude),
				std::cos(latitude),
				std::sin(latitude) * std::sin(longitude),
			}});
		}
	}
	indices.reserve(rings * sectors * 6);
	for (uint32_t ring = 0; ring < rings; ++ring) {
		for (uint32_t sector = 0; sector < sectors; ++sector) {
			const uint16_t a = static_cast<uint16_t>(ring * (sectors + 1) + sector);
			const uint16_t b = static_cast<uint16_t>(a + sectors + 1);
			indices.insert(indices.end(), {a, b, static_cast<uint16_t>(a + 1), static_cast<uint16_t>(a + 1),
										   b, static_cast<uint16_t>(b + 1)});
		}
	}
	return vertices;
}

std::vector<MarkerVertex> makeConeLines() {
	std::vector<MarkerVertex> vertices;
	vertices.reserve(coneSegments * 2 + 10);
	for (uint32_t segment = 0; segment < coneSegments; ++segment) {
		const float angleA = 2.0f * math::pi * static_cast<float>(segment) / coneSegments;
		const float angleB = 2.0f * math::pi * static_cast<float>(segment + 1) / coneSegments;
		const MarkerVertex apex{{0.0f, 0.0f, 0.0f}};
		const MarkerVertex ringA{{std::cos(angleA), std::sin(angleA), 1.0f}};
		const MarkerVertex ringB{{std::cos(angleB), std::sin(angleB), 1.0f}};
		vertices.insert(vertices.end(), {ringA, ringB});
		if (segment % 8 == 0)
			vertices.insert(vertices.end(), {apex, ringA});
	}
	vertices.push_back({{0.0f, 0.0f, 0.0f}});
	vertices.push_back({{0.0f, 0.0f, 1.0f}});
	return vertices;
}

bool createMarkerGeometry() {
	std::vector<uint16_t> sphereIndices;
	const std::vector<MarkerVertex> sphereVertices = makeSphereVertices(10, 16, sphereIndices);
	const std::vector<MarkerVertex> coneVertices = makeConeLines();
	markerIndexCount = static_cast<uint32_t>(sphereIndices.size());
	coneVertexCount = static_cast<uint32_t>(coneVertices.size());
	if (!createHostBuffer(sphereVertices.size() * sizeof(MarkerVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
						  markerVertexBuffer))
		return false;
	if (!uploadBuffer(markerVertexBuffer, sphereVertices.data(),
					  sphereVertices.size() * sizeof(MarkerVertex)))
		return false;
	if (!createHostBuffer(sphereIndices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
						  markerIndexBuffer))
		return false;
	if (!uploadBuffer(markerIndexBuffer, sphereIndices.data(), sphereIndices.size() * sizeof(uint16_t)))
		return false;
	if (!createHostBuffer(coneVertices.size() * sizeof(MarkerVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
						  coneVertexBuffer))
		return false;
	return uploadBuffer(coneVertexBuffer, coneVertices.data(), coneVertices.size() * sizeof(MarkerVertex));
}

bool ensureStorageCapacity(GpuBuffer &buffer, size_t elementCount, size_t elementSize, uint32_t binding) {
	auto &context = graphics::internal::context;
	const VkDeviceSize required = static_cast<VkDeviceSize>(std::max<size_t>(elementCount, 1) * elementSize);
	if (buffer.size >= required)
		return true;
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	if (required > properties.limits.maxStorageBufferRange) {
		std::cerr << "The light array exceeds this device's storage-buffer range\n";
		return false;
	}
	const VkDeviceSize newSize =
		std::min<VkDeviceSize>(std::max(required, std::max<VkDeviceSize>(buffer.size * 2, elementSize)),
							   properties.limits.maxStorageBufferRange);
	GpuBuffer replacement{};
	if (!createHostBuffer(newSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, replacement))
		return false;
	if (sceneDescriptorSet != VK_NULL_HANDLE) {
		const VkDescriptorBufferInfo bufferInfo{replacement.buffer, 0, replacement.size};
		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = sceneDescriptorSet;
		write.dstBinding = binding;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.pBufferInfo = &bufferInfo;
		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}
	destroyBuffer(buffer);
	buffer = replacement;
	return true;
}

bool syncLightBuffers() {
	if (!lightsDirty)
		return true;
	pointData.clear();
	pointData.reserve(scene.lighting.points.size());
	for (const PointLight &light : scene.lighting.points) {
		pointData.push_back({
			.position = {light.position[0], light.position[1], light.position[2], 1.0f},
			.colorIntensity = {light.color[0], light.color[1], light.color[2], light.intensity},
		});
	}
	spotData.clear();
	spotData.reserve(scene.lighting.spots.size());
	for (const SpotLight &light : scene.lighting.spots) {
		const Vec3 direction = math::normalize(fromArray(light.direction));
		const float innerCos = std::cos(light.innerAngleDegrees * math::pi / 180.0f);
		const float outerCos = std::cos(light.outerAngleDegrees * math::pi / 180.0f);
		spotData.push_back({
			.position = {light.position[0], light.position[1], light.position[2], 1.0f},
			.direction = {direction.x, direction.y, direction.z, 0.0f},
			.colorIntensity = {light.color[0], light.color[1], light.color[2], light.intensity},
			.coneCosines = {innerCos, outerCos, 0.0f, 0.0f},
		});
	}
	if (!ensureStorageCapacity(pointLightBuffer, pointData.size(), sizeof(PointLightGpu), 1) ||
		!ensureStorageCapacity(spotLightBuffer, spotData.size(), sizeof(SpotLightGpu), 2))
		return false;
	const PointLightGpu emptyPoint{};
	const SpotLightGpu emptySpot{};
	const void *pointSource = pointData.empty() ? static_cast<const void *>(&emptyPoint) : pointData.data();
	const void *spotSource = spotData.empty() ? static_cast<const void *>(&emptySpot) : spotData.data();
	const bool uploaded =
		uploadBuffer(pointLightBuffer, pointSource,
					 pointData.empty() ? sizeof(emptyPoint) : pointData.size() * sizeof(PointLightGpu)) &&
		uploadBuffer(spotLightBuffer, spotSource,
					 spotData.empty() ? sizeof(emptySpot) : spotData.size() * sizeof(SpotLightGpu));
	if (uploaded)
		lightsDirty = false;
	return uploaded;
}

bool createDescriptorLayouts() {
	auto &context = graphics::internal::context;
	std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{};
	sceneBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
						VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
	sceneBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
	sceneBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
	sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	sceneLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
	sceneLayoutInfo.pBindings = sceneBindings.data();
	if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr, &sceneSetLayout) !=
		VK_SUCCESS) {
		std::cerr << "Failed to create the scene descriptor-set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding objectBinding{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	VkDescriptorSetLayoutCreateInfo objectLayoutInfo{};
	objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	objectLayoutInfo.bindingCount = 1;
	objectLayoutInfo.pBindings = &objectBinding;
	if (vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr, &objectSetLayout) !=
		VK_SUCCESS) {
		std::cerr << "Failed to create the model descriptor-set layout\n";
		return false;
	}

	const std::array<VkDescriptorPoolSize, 2> poolSizes{{
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + object_count},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
	}};
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1 + object_count;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		std::cerr << "Failed to create the application descriptor pool\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, 1 + object_count> layouts{};
	layouts[0] = sceneSetLayout;
	for (uint32_t i = 0; i < object_count; ++i)
		layouts[1 + i] = objectSetLayout;
	std::array<VkDescriptorSet, 1 + object_count> descriptorSets{};
	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
	allocateInfo.pSetLayouts = layouts.data();
	if (vkAllocateDescriptorSets(context.device, &allocateInfo, descriptorSets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate application descriptor sets\n";
		return false;
	}
	sceneDescriptorSet = descriptorSets[0];
	for (uint32_t i = 0; i < object_count; ++i)
		objects[i].descriptorSet = descriptorSets[1 + i];

	std::array<VkDescriptorBufferInfo, 3 + object_count> bufferInfos{};
	bufferInfos[0] = {sceneUniformBuffer.buffer, 0, sizeof(SceneUniform)};
	bufferInfos[1] = {pointLightBuffer.buffer, 0, pointLightBuffer.size};
	bufferInfos[2] = {spotLightBuffer.buffer, 0, spotLightBuffer.size};
	for (uint32_t i = 0; i < object_count; ++i) {
		bufferInfos[3 + i] = {objects[i].uniformBuffer.buffer, 0, sizeof(ObjectUniform)};
	}
	std::array<VkWriteDescriptorSet, 3 + object_count> writes{};
	for (uint32_t i = 0; i < 3; ++i) {
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = sceneDescriptorSet;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType =
			i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &bufferInfos[i];
	}
	for (uint32_t i = 0; i < object_count; ++i) {
		VkWriteDescriptorSet &write = writes[3 + i];
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = objects[i].descriptorSet;
		write.dstBinding = 0;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		write.pBufferInfo = &bufferInfos[3 + i];
	}
	vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	return true;
}

bool createPipeline(bool lit, VkPrimitiveTopology topology, VkPipeline &pipeline) {
	auto &context = graphics::internal::context;
	const std::string shaderDir = OCTAHEDRON_SHADER_DIR;
	const auto vertexCode =
		graphics::readShader(shaderDir + (lit ? "/octahedron.vert.spv" : "/marker.vert.spv"));
	const auto fragmentCode =
		graphics::readShader(shaderDir + (lit ? "/octahedron.frag.spv" : "/marker.frag.spv"));
	const VkShaderModule vertexModule = graphics::createShaderModule(vertexCode);
	const VkShaderModule fragmentModule = graphics::createShaderModule(fragmentCode);
	if (vertexModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		if (vertexModule != VK_NULL_HANDLE)
			vkDestroyShaderModule(context.device, vertexModule, nullptr);
		if (fragmentModule != VK_NULL_HANDLE)
			vkDestroyShaderModule(context.device, fragmentModule, nullptr);
		return false;
	}
	const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
		{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
		 vertexModule, "main", nullptr},
		{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
		 fragmentModule, "main", nullptr},
	}};
	const VkVertexInputBindingDescription vertexBinding{
		0, static_cast<uint32_t>(lit ? sizeof(Vertex) : sizeof(MarkerVertex)), VK_VERTEX_INPUT_RATE_VERTEX};
	const std::array<VkVertexInputAttributeDescription, 3> attributes{{
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
		{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	}};
	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &vertexBinding;
	vertexInput.vertexAttributeDescriptionCount = lit ? static_cast<uint32_t>(attributes.size()) : 1;
	vertexInput.pVertexAttributeDescriptions = attributes.data();
	VkPipelineInputAssemblyStateCreateInfo assembly{};
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = topology;
	VkPipelineViewportStateCreateInfo viewport{};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo rasterization{};
	rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization.polygonMode = VK_POLYGON_MODE_FILL;
	rasterization.cullMode = lit ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
	rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterization.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo depth{};
	depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth.depthTestEnable = VK_TRUE;
	depth.depthWriteEnable = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
									 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &blendAttachment;
	const std::array<VkDynamicState, 2> dynamicStates{{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamic.pDynamicStates = dynamicStates.data();
	VkGraphicsPipelineCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	info.stageCount = static_cast<uint32_t>(stages.size());
	info.pStages = stages.data();
	info.pVertexInputState = &vertexInput;
	info.pInputAssemblyState = &assembly;
	info.pViewportState = &viewport;
	info.pRasterizationState = &rasterization;
	info.pMultisampleState = &multisampling;
	info.pDepthStencilState = &depth;
	info.pColorBlendState = &blend;
	info.pDynamicState = &dynamic;
	info.layout = lit ? pipelineLayout : markerPipelineLayout;
	info.renderPass = context.render_pass;
	info.subpass = 0;
	const VkResult result =
		vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
	vkDestroyShaderModule(context.device, fragmentModule, nullptr);
	vkDestroyShaderModule(context.device, vertexModule, nullptr);
	return result == VK_SUCCESS;
}

bool createPipelines() {
	auto device = graphics::internal::context.device;
	const std::array<VkDescriptorSetLayout, 2> layouts{sceneSetLayout, objectSetLayout};
	VkPipelineLayoutCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	info.setLayoutCount = static_cast<uint32_t>(layouts.size());
	info.pSetLayouts = layouts.data();
	if (vkCreatePipelineLayout(device, &info, nullptr, &pipelineLayout) != VK_SUCCESS)
		return false;

	const VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
										sizeof(MarkerPush)};
	info.setLayoutCount = 1;
	info.pSetLayouts = &sceneSetLayout;
	info.pushConstantRangeCount = 1;
	info.pPushConstantRanges = &pushRange;
	if (vkCreatePipelineLayout(device, &info, nullptr, &markerPipelineLayout) != VK_SUCCESS)
		return false;
	return createPipeline(true, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, graphicsPipeline) &&
		   createPipeline(false, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, markerPipeline) &&
		   createPipeline(false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, conePipeline);
}

void releaseResources() {
	auto &context = graphics::internal::context;
	if (context.device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(context.device);
		if (graphicsPipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
		if (markerPipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(context.device, markerPipeline, nullptr);
		if (conePipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(context.device, conePipeline, nullptr);
		if (pipelineLayout != VK_NULL_HANDLE)
			vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
		if (markerPipelineLayout != VK_NULL_HANDLE)
			vkDestroyPipelineLayout(context.device, markerPipelineLayout, nullptr);
		if (descriptorPool != VK_NULL_HANDLE)
			vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
		if (objectSetLayout != VK_NULL_HANDLE)
			vkDestroyDescriptorSetLayout(context.device, objectSetLayout, nullptr);
		if (sceneSetLayout != VK_NULL_HANDLE)
			vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
	}
	graphicsPipeline = VK_NULL_HANDLE;
	markerPipeline = VK_NULL_HANDLE;
	conePipeline = VK_NULL_HANDLE;
	pipelineLayout = VK_NULL_HANDLE;
	markerPipelineLayout = VK_NULL_HANDLE;
	descriptorPool = VK_NULL_HANDLE;
	objectSetLayout = VK_NULL_HANDLE;
	sceneSetLayout = VK_NULL_HANDLE;
	sceneDescriptorSet = VK_NULL_HANDLE;
	for (ObjectResources &object : objects) {
		object.descriptorSet = VK_NULL_HANDLE;
		destroyBuffer(object.uniformBuffer);
	}
	destroyBuffer(coneVertexBuffer);
	destroyBuffer(markerIndexBuffer);
	destroyBuffer(markerVertexBuffer);
	destroyBuffer(spotLightBuffer);
	destroyBuffer(pointLightBuffer);
	destroyBuffer(indexBuffer);
	destroyBuffer(vertexBuffer);
	destroyBuffer(sceneUniformBuffer);
	markerIndexCount = 0;
	coneVertexCount = 0;
}

void updateAnimation(double time) {
	if (!scene.animation.hasPreviousTime) {
		scene.animation.previousTime = time;
		scene.animation.hasPreviousTime = true;
	}
	const double elapsed = std::clamp(time - scene.animation.previousTime, 0.0, 0.1);
	scene.animation.previousTime = time;
	if (scene.animation.playing)
		scene.animation.time += elapsed * static_cast<double>(scene.animation.speed);
}

Mat4 modelMatrixFor(size_t objectIndex);

std::array<float, 3> *lightPosition(LightSelection selection) {
	if (selection.kind == LightKind::point && selection.index < scene.lighting.points.size())
		return &scene.lighting.points[selection.index].position;
	if (selection.kind == LightKind::spot && selection.index < scene.lighting.spots.size())
		return &scene.lighting.spots[selection.index].position;
	return nullptr;
}

void selectLight(LightKind kind, size_t index) {
	interaction.selected = {kind, index};
}

Mat4 cameraView() {
	return math::viewMatrix(fromArray(scene.camera.position), scene.camera.yaw * math::pi / 180.0f,
							scene.camera.pitch * math::pi / 180.0f);
}

void setLooking(bool looking) {
	if (interaction.looking == looking)
		return;
	interaction.looking = looking;
	glfwSetInputMode(inputWindow, GLFW_CURSOR, looking ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(inputWindow, GLFW_RAW_MOUSE_MOTION, looking);
	glfwGetCursorPos(inputWindow, &interaction.cursorX, &interaction.cursorY);
}

void updateFlight(float elapsed, bool look) {
	if (look) {
		double x, y;
		glfwGetCursorPos(inputWindow, &x, &y);
		scene.camera.yaw =
			std::remainder(scene.camera.yaw + static_cast<float>(x - interaction.cursorX) * 0.15f, 360.0f);
		scene.camera.pitch = std::clamp(
			scene.camera.pitch - static_cast<float>(y - interaction.cursorY) * 0.15f, -89.0f, 89.0f);
		interaction.cursorX = x;
		interaction.cursorY = y;
	}
	const Mat4 view = cameraView();
	const Vec3 forward{-view.at(2, 0), -view.at(2, 1), -view.at(2, 2)};
	const Vec3 right{view.at(0, 0), view.at(0, 1), view.at(0, 2)};
	const auto down = [&](int key) { return glfwGetKey(inputWindow, key) == GLFW_PRESS ? 1.0f : 0.0f; };
	Vec3 direction = math::add(math::multiply(forward, down(GLFW_KEY_W) - down(GLFW_KEY_S)),
							   math::multiply(right, down(GLFW_KEY_D) - down(GLFW_KEY_A)));

	if (math::dot(direction, direction) > 1.0e-6f) {
		const float speed = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT) ? 9.0f : 3.0f;
		const Vec3 position = math::add(fromArray(scene.camera.position),
										math::multiply(math::normalize(direction), speed * elapsed));
		scene.camera.position = {position.x, position.y, position.z};
	}
}

void deleteSelectedLight() {
	if (!lightPosition(interaction.selected))
		return;
	if (interaction.selected.kind == LightKind::point)
		scene.lighting.points.erase(scene.lighting.points.begin() + interaction.selected.index);
	else
		scene.lighting.spots.erase(scene.lighting.spots.begin() + interaction.selected.index);
	interaction.selected = {};
	lightsDirty = true;
	lightMessage.clear();
}

void updateCamera(double time) {
	const float elapsed = static_cast<float>(std::clamp(time - interaction.previousTime, 0.0, 0.05));
	interaction.previousTime = time;
	if (!glfwGetWindowAttrib(inputWindow, GLFW_FOCUSED) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
		setLooking(false);
		return;
	}
	const auto &io = ImGui::GetIO();
	if (interaction.looking && glfwGetMouseButton(inputWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
		setLooking(false);
	const bool overScene = io.MousePos.x >= sidebarRight && io.MousePos.x < io.DisplaySize.x &&
						   io.MousePos.y >= 0 && io.MousePos.y < io.DisplaySize.y;
	if (!interaction.looking && overScene && !io.WantCaptureMouse && !io.WantTextInput &&
		!ImGui::IsAnyItemActive() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
		setLooking(true);
	if (!io.WantTextInput)
		updateFlight(elapsed, interaction.looking);
}

void editVector(const char *label, std::array<float, 3> &value, float step = 0.05f) {
	ImGui::PushID(label);
	ImGui::TextUnformatted(label);
	ImGui::SetNextItemWidth(-1);
	ImGui::DragFloat3("##value", value.data(), step);
	ImGui::PopID();
}

void drawCameraControls() {
	ImGui::TextWrapped("WASD moves the camera while this window is focused. Hold RMB in the scene to look "
					   "around. Shift moves faster.");
	ImGui::Text("Position: %.2f  %.2f  %.2f", scene.camera.position[0], scene.camera.position[1],
				scene.camera.position[2]);
	int projection = static_cast<int>(scene.camera.projection);
	if (ImGui::Combo("Projection", &projection, "Perspective\0Orthographic\0")) {
		scene.camera.projection = static_cast<ProjectionMode>(projection);
	}
	if (scene.camera.projection == ProjectionMode::perspective)
		ImGui::SliderFloat("Vertical FOV", &scene.camera.fieldOfView, 15.0f, 110.0f, "%.0f deg");
	if (ImGui::Button("Reset camera"))
		scene.camera = Camera{};
}

void drawObjectControls() {
	editVector("Position", scene.model.position);
	editVector("Rotation in degrees", scene.model.rotation, 0.5f);
	ImGui::TextUnformatted("Scale along X Y Z");
	ImGui::SetNextItemWidth(-1);
	ImGui::DragFloat3("##scale", scene.model.scale.data(), 0.01f, 0.05f, 3.0f, "%.2f",
					  ImGuiSliderFlags_AlwaysClamp);
	ImGui::Checkbox("Show second octahedron", &scene.model.showCompanion);
	ImGui::Checkbox("Show floor", &scene.model.showFloor);
	ImGui::SeparatorText("Animation");
	ImGui::Checkbox("Play animation", &scene.animation.playing);
	ImGui::SliderFloat("Speed", &scene.animation.speed, 0.0f, 3.0f);
	ImGui::SliderFloat("Path radius", &scene.animation.radius, 0.0f, 1.5f);
	ImGui::SliderFloat("Path height", &scene.animation.height, 0.0f, 0.8f);
	if (ImGui::Button("Restart animation"))
		scene.animation.time = 0.0;
}

void drawMaterialControls() {
	ImGui::ColorEdit3("Albedo", scene.material.albedo.data());
	ImGui::ColorEdit3("Specular", scene.material.specular.data());
	ImGui::SliderFloat("Shininess", &scene.material.shininess, 1.0f, 128.0f, "%.0f");
	ImGui::Checkbox("Use vertex colors", &scene.material.vertexColors);
	ImGui::ColorEdit3("Vertex tint", scene.model.tint.data());
	ImGui::TextWrapped(
		"Higher shininess makes the highlight narrower. The floor has its own matte material.");
}

void drawSelectedLightControls() {
	auto *position = lightPosition(interaction.selected);
	if (!position) {
		ImGui::TextWrapped("Choose a light from the list to edit its coordinates.");
		return;
	}
	ImGui::SeparatorText("Selected light");
	if (ImGui::Button("Delete")) {
		deleteSelectedLight();
		return;
	}
	ImGui::TextUnformatted("Position X / Y / Z");
	ImGui::SetNextItemWidth(-1);
	lightsDirty |= ImGui::InputFloat3("##position", position->data(), "%.3f");
	if (interaction.selected.kind == LightKind::point) {
		auto &light = scene.lighting.points[interaction.selected.index];
		lightsDirty |= ImGui::ColorEdit3("Color", light.color.data());
		lightsDirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
	} else {
		auto &light = scene.lighting.spots[interaction.selected.index];
		if (ImGui::Button("Aim at object")) {
			const Vec3 target = math::transformPoint(modelMatrixFor(0), {0, 0, 0});
			const Vec3 direction = math::normalize(math::subtract(target, fromArray(light.position)));
			light.direction = {direction.x, direction.y, direction.z};
			lightsDirty = true;
		}
		ImGui::TextUnformatted("Ray direction X / Y / Z");
		ImGui::SetNextItemWidth(-1);
		lightsDirty |= ImGui::InputFloat3("##direction", light.direction.data(), "%.3f");
		lightsDirty |= ImGui::ColorEdit3("Color", light.color.data());
		lightsDirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
		lightsDirty |= ImGui::SliderFloat("Inner angle", &light.innerAngleDegrees, 1.0f, 88.0f, "%.0f deg");
		const float minimumOuter = light.innerAngleDegrees + 1.0f;
		if (light.outerAngleDegrees < minimumOuter) {
			light.outerAngleDegrees = minimumOuter;
			lightsDirty = true;
		}
		lightsDirty |=
			ImGui::SliderFloat("Outer angle", &light.outerAngleDegrees, minimumOuter, 89.0f, "%.0f deg");
	}
}

void drawLightingControls() {
	ImGui::Combo("Lighting mode", &scene.lighting.mode,
				 "Combined\0Directional only\0Point only\0Spot only\0Ambient only\0");
	if (ImGui::CollapsingHeader("Ambient and directional")) {
		ImGui::ColorEdit3("Ambient color", scene.lighting.ambientColor.data());
		ImGui::SliderFloat("Ambient intensity", &scene.lighting.ambientIntensity, 0.0f, 1.0f);
		ImGui::DragFloat3("Toward light", scene.lighting.direction.data(), 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Directional color", scene.lighting.color.data());
		ImGui::SliderFloat("Directional intensity", &scene.lighting.intensity, 0.0f, 3.0f);
	}
	if (ImGui::Button("Add point light")) {
		if (scene.lighting.points.size() < maxPointLights) {
			PointLight light;
			const float angle = static_cast<float>(scene.lighting.points.size()) * 2.4f;
			light.position = {2.2f * std::cos(angle), 1.2f, 2.2f * std::sin(angle)};
			scene.lighting.points.push_back(light);
			selectLight(LightKind::point, scene.lighting.points.size() - 1);
			lightsDirty = true;
			lightMessage.clear();
		} else
			lightMessage = "The point-light buffer limit has been reached.";
	}
	ImGui::SameLine();
	if (ImGui::Button("Add spotlight")) {
		if (scene.lighting.spots.size() < maxSpotLights) {
			SpotLight light;
			const float angle = static_cast<float>(scene.lighting.spots.size()) * 2.4f;
			light.position = {2.5f * std::cos(angle), 2.8f, 2.5f * std::sin(angle)};
			const Vec3 direction = math::normalize(math::subtract(
				math::transformPoint(modelMatrixFor(0), {0, 0, 0}), fromArray(light.position)));
			light.direction = {direction.x, direction.y, direction.z};
			scene.lighting.spots.push_back(light);
			selectLight(LightKind::spot, scene.lighting.spots.size() - 1);
			lightsDirty = true;
			lightMessage.clear();
		} else
			lightMessage = "The spotlight buffer limit has been reached.";
	}
	if (ImGui::BeginListBox("##lights", ImVec2(-1, 110))) {
		for (size_t i = 0; i < scene.lighting.points.size(); ++i) {
			const std::string name = "Point " + std::to_string(i + 1);
			if (ImGui::Selectable(name.c_str(), interaction.selected.kind == LightKind::point &&
													interaction.selected.index == i))
				selectLight(LightKind::point, i);
		}
		for (size_t i = 0; i < scene.lighting.spots.size(); ++i) {
			const std::string name = "Spot " + std::to_string(i + 1);
			if (ImGui::Selectable(name.c_str(), interaction.selected.kind == LightKind::spot &&
													interaction.selected.index == i))
				selectLight(LightKind::spot, i);
		}
		ImGui::EndListBox();
	}
	drawSelectedLightControls();
	if (!lightMessage.empty())
		ImGui::TextWrapped("%s", lightMessage.c_str());
}

void drawControls() {
	const ImVec2 display = ImGui::GetIO().DisplaySize;
	const float width = std::clamp(display.x * 0.3f, 300.0f, 380.0f);
	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(width, display.y), ImGuiCond_Always);
	ImGui::Begin("Lab 2 controls", nullptr,
				 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar |
					 ImGuiWindowFlags_NoSavedSettings);
	sidebarRight = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
	ImGui::SeparatorText("CAMERA AND LIGHTING");
	const float settingsHeight = std::max(120.0f, ImGui::GetContentRegionAvail().y - 100.0f);
	ImGui::BeginChild("Settings", ImVec2(0, settingsHeight), false);
	ImGui::PushItemWidth(std::max(100.0f, width * 0.48f));
	if (ImGui::BeginTabBar("Settings tabs")) {
		if (ImGui::BeginTabItem("Camera")) {
			drawCameraControls();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Objects")) {
			drawObjectControls();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Material")) {
			drawMaterialControls();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights")) {
			drawLightingControls();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
	ImGui::PopItemWidth();
	ImGui::EndChild();
	ImGui::Separator();
	ImGui::Checkbox("Show light helpers", &scene.lighting.showSources);
	if (ImGui::Button("Reset scene")) {
		setLooking(false);
		interaction = Interaction{};
		scene = SceneState{};
		lightsDirty = true;
		lightMessage.clear();
	}
	ImGui::SameLine();
	ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);
	ImGui::End();
}

Mat4 modelMatrixFor(size_t objectIndex) {
	if (objectIndex == 2)
		return math::translationMatrix(0.0f, -1.1f, 0.0f);
	const double fullTurn = 2.0 * static_cast<double>(math::pi);
	const float t = static_cast<float>(std::remainder(scene.animation.time, fullTurn));
	const float mainSpin = static_cast<float>(std::remainder(0.65 * scene.animation.time, fullTurn));
	const float companionSpin = static_cast<float>(std::remainder(-0.45 * scene.animation.time, fullTurn));
	const Vec3 animationOffset{
		scene.animation.radius * (std::cos(t) - 1.0f),
		scene.animation.height * std::sin(2.0f * t),
		scene.animation.radius * 0.7f * std::sin(t),
	};
	const Vec3 mainPosition{
		scene.model.position[0] + animationOffset.x,
		scene.model.position[1] + animationOffset.y,
		scene.model.position[2] + animationOffset.z,
	};
	const Vec3 baseRotation = math::radiansFromDegrees(scene.model.rotation);
	const Vec3 mainRotation{
		baseRotation.x + 0.35f * std::sin(t),
		baseRotation.y + mainSpin,
		baseRotation.z + 0.25f * std::sin(2.0f * t),
	};

	if (objectIndex == 1) {
		return math::modelMatrix({-1.65f, 0.0f, -0.15f}, {-0.35f, -0.6f + companionSpin, 0.35f},
								 {0.52f, 0.52f, 0.52f});
	}
	return math::modelMatrix(mainPosition, mainRotation, fromArray(scene.model.scale));
}

bool updateUniforms() {
	auto &context = graphics::internal::context;
	if (!syncLightBuffers())
		return false;
	const uint32_t viewportX = sceneViewportX(context.swapchain_extent.width);
	const float aspect = static_cast<float>(context.swapchain_extent.width - viewportX) /
						 static_cast<float>(context.swapchain_extent.height);
	const Vec3 cameraPosition = fromArray(scene.camera.position);
	SceneUniform sceneData{};
	sceneData.view = math::viewMatrix(cameraPosition, scene.camera.yaw * math::pi / 180.0f,
									  scene.camera.pitch * math::pi / 180.0f);
	sceneData.projection = scene.camera.projection == ProjectionMode::perspective
							   ? math::perspectiveMatrix(aspect, scene.camera.fieldOfView)
							   : math::orthographicMatrix(aspect);
	sceneData.cameraPosition[0] = cameraPosition.x;
	sceneData.cameraPosition[1] = cameraPosition.y;
	sceneData.cameraPosition[2] = cameraPosition.z;
	sceneData.cameraPosition[3] = 1.0f;
	for (size_t i = 0; i < 3; ++i) {
		sceneData.ambientColorIntensity[i] = scene.lighting.ambientColor[i];
		sceneData.directionalColorIntensity[i] = scene.lighting.color[i];
	}
	sceneData.ambientColorIntensity[3] = scene.lighting.ambientIntensity;
	sceneData.directionalColorIntensity[3] = scene.lighting.mode <= 1 ? scene.lighting.intensity : 0.0f;
	const Vec3 directionalDirection = math::normalize(fromArray(scene.lighting.direction));
	sceneData.directionalDirection[0] = directionalDirection.x;
	sceneData.directionalDirection[1] = directionalDirection.y;
	sceneData.directionalDirection[2] = directionalDirection.z;
	sceneData.lightCounts[0] = (scene.lighting.mode == 0 || scene.lighting.mode == 2)
								   ? static_cast<uint32_t>(scene.lighting.points.size())
								   : 0;
	sceneData.lightCounts[1] = (scene.lighting.mode == 0 || scene.lighting.mode == 3)
								   ? static_cast<uint32_t>(scene.lighting.spots.size())
								   : 0;
	if (!uploadBuffer(sceneUniformBuffer, &sceneData, sizeof(sceneData)))
		return false;

	ObjectUniform mainUniform{};
	mainUniform.model = modelMatrixFor(0);
	for (size_t i = 0; i < 3; ++i) {
		mainUniform.albedo[i] = scene.material.albedo[i] * scene.model.tint[i];
		mainUniform.specularColor[i] = scene.material.specular[i];
	}
	mainUniform.albedo[3] = 1.0f;
	mainUniform.specularColor[3] = 1.0f;
	mainUniform.material[0] = scene.material.shininess;
	mainUniform.material[1] = scene.material.vertexColors ? 1.0f : 0.0f;
	if (!uploadBuffer(objects[main_object_index].uniformBuffer, &mainUniform, sizeof(mainUniform)))
		return false;

	ObjectUniform companionUniform{};
	companionUniform.model = modelMatrixFor(1);
	companionUniform.albedo[0] = 0.95f * scene.material.albedo[0];
	companionUniform.albedo[1] = 0.6f * scene.material.albedo[1];
	companionUniform.albedo[2] = 0.25f * scene.material.albedo[2];
	companionUniform.albedo[3] = 1.0f;
	for (size_t i = 0; i < 3; ++i)
		companionUniform.specularColor[i] = scene.material.specular[i];
	companionUniform.specularColor[3] = 1.0f;
	companionUniform.material[0] = scene.material.shininess;
	companionUniform.material[1] = scene.material.vertexColors ? 1.0f : 0.0f;
	if (!uploadBuffer(objects[companion_object_index].uniformBuffer, &companionUniform,
					  sizeof(companionUniform)))
		return false;
	ObjectUniform floor{};
	floor.model = math::translationMatrix(0.0f, -1.1f, 0.0f);
	floor.albedo[0] = 0.3f;
	floor.albedo[1] = 0.33f;
	floor.albedo[2] = 0.38f;
	floor.specularColor[0] = floor.specularColor[1] = floor.specularColor[2] = 0.04f;
	floor.material[0] = 16.0f;
	return uploadBuffer(objects[2].uniformBuffer, &floor, sizeof(floor));
}

void pushMarker(VkCommandBuffer commandBuffer, const Mat4 &model, const std::array<float, 3> &color) {
	MarkerPush push{};
	push.model = model;
	push.color[0] = std::clamp(color[0], 0.0f, 1.0f);
	push.color[1] = std::clamp(color[1], 0.0f, 1.0f);
	push.color[2] = std::clamp(color[2], 0.0f, 1.0f);
	push.color[3] = 1.0f;
	vkCmdPushConstants(commandBuffer, markerPipelineLayout,
					   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
}

void drawLightVisualization(VkCommandBuffer commandBuffer) {
	if (!scene.lighting.showSources)
		return;
	const VkDeviceSize offset = 0;
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipelineLayout, 0, 1,
							&sceneDescriptorSet, 0, nullptr);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipeline);
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &markerVertexBuffer.buffer, &offset);
	vkCmdBindIndexBuffer(commandBuffer, markerIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	for (const PointLight &light : scene.lighting.points) {
		pushMarker(commandBuffer,
				   math::modelMatrix(fromArray(light.position), {0.0f, 0.0f, 0.0f}, {0.13f, 0.13f, 0.13f}),
				   light.color);
		vkCmdDrawIndexed(commandBuffer, markerIndexCount, 1, 0, 0, 0);
	}
	for (const SpotLight &light : scene.lighting.spots) {
		const Vec3 position = fromArray(light.position);
		const Vec3 direction = math::normalize(fromArray(light.direction));
		pushMarker(commandBuffer, math::modelMatrix(position, {0.0f, 0.0f, 0.0f}, {0.15f, 0.15f, 0.15f}),
				   light.color);
		vkCmdDrawIndexed(commandBuffer, markerIndexCount, 1, 0, 0, 0);
		if (interaction.selected.kind != LightKind::spot ||
			interaction.selected.index >= scene.lighting.spots.size() ||
			&light != &scene.lighting.spots[interaction.selected.index])
			continue;
		const float coneLength = 1.2f;
		const float coneRadius = coneLength * std::tan(light.outerAngleDegrees * math::pi / 180.0f);
		const Mat4 coneModel = directionMatrix(position, direction, coneRadius, coneLength);
		pushMarker(commandBuffer, coneModel, {0.9f, 0.9f, 0.9f});
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, conePipeline);
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &coneVertexBuffer.buffer, &offset);
		vkCmdDraw(commandBuffer, coneVertexCount, 1, 0, 0);
		const float innerRadius = coneLength * std::tan(light.innerAngleDegrees * math::pi / 180.0f);
		pushMarker(commandBuffer, directionMatrix(position, direction, innerRadius, coneLength), light.color);
		vkCmdDraw(commandBuffer, coneVertexCount, 1, 0, 0);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipeline);
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &markerVertexBuffer.buffer, &offset);
		vkCmdBindIndexBuffer(commandBuffer, markerIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	}
}

} // namespace

bool initialize(GLFWwindow *window) {
	inputWindow = window;
	interaction = Interaction{};
	const auto fail = []() {
		releaseResources();
		return false;
	};
	scene = SceneState{};
	lightsDirty = true;
	const auto octahedron = geometry::makeOctahedronVertices();
	std::vector<Vertex> vertices(octahedron.begin(), octahedron.end());
	for (const Vec3 &p : {Vec3{-5, 0, -5}, Vec3{5, 0, -5}, Vec3{5, 0, 5}, Vec3{-5, 0, 5}}) {
		vertices.push_back({{p.x, p.y, p.z}, {1, 1, 1}, {0, 1, 0}});
	}
	std::vector<uint16_t> indices(geometry::octahedronIndices.begin(), geometry::octahedronIndices.end());
	indices.insert(indices.end(), {0, 2, 1, 0, 3, 2});
	if (!createHostBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer))
		return fail();
	if (!uploadBuffer(vertexBuffer, vertices.data(), vertices.size() * sizeof(Vertex)))
		return fail();
	if (!createHostBuffer(indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer))
		return fail();
	if (!uploadBuffer(indexBuffer, indices.data(), indices.size() * sizeof(uint16_t)))
		return fail();
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
	maxPointLights = properties.limits.maxStorageBufferRange / sizeof(PointLightGpu);
	maxSpotLights = properties.limits.maxStorageBufferRange / sizeof(SpotLightGpu);
	if (!createHostBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUniformBuffer))
		return fail();
	for (ObjectResources &object : objects) {
		if (!createHostBuffer(sizeof(ObjectUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							  object.uniformBuffer))
			return fail();
	}
	if (!createMarkerGeometry())
		return fail();
	if (!ensureStorageCapacity(pointLightBuffer, scene.lighting.points.size(), sizeof(PointLightGpu), 1) ||
		!ensureStorageCapacity(spotLightBuffer, scene.lighting.spots.size(), sizeof(SpotLightGpu), 2))
		return fail();
	if (!createDescriptorLayouts())
		return fail();
	if (!createPipelines())
		return fail();
	return true;
}

void shutdown() {
	if (inputWindow)
		glfwSetInputMode(inputWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	releaseResources();
}

void update(double time) {
	updateAnimation(time);
	drawControls();
	updateCamera(time);
}

bool render(const graphics::internal::FrameData &fd) {
	auto &context = graphics::internal::context;
	if (graphicsPipeline == VK_NULL_HANDLE || fd.command_buffer == VK_NULL_HANDLE ||
		context.swapchain_extent.width == 0 || context.swapchain_extent.height == 0)
		return false;
	if (vkResetCommandBuffer(fd.command_buffer, 0) != VK_SUCCESS) {
		std::cerr << "Failed to reset the drawing command buffer\n";
		return false;
	}
	if (!updateUniforms())
		return false;
	VkCommandBufferBeginInfo commandBegin{};
	commandBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	if (vkBeginCommandBuffer(fd.command_buffer, &commandBegin) != VK_SUCCESS) {
		std::cerr << "Failed to begin the application command buffer\n";
		return false;
	}
	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color.float32[0] = 0.012f;
	clearValues[0].color.float32[1] = 0.017f;
	clearValues[0].color.float32[2] = 0.025f;
	clearValues[0].color.float32[3] = 1.0f;
	clearValues[1].depthStencil.depth = 1.0f;
	VkRenderPassBeginInfo renderBegin{};
	renderBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderBegin.renderPass = context.render_pass;
	renderBegin.framebuffer = fd.framebuffer;
	renderBegin.renderArea.extent = context.swapchain_extent;
	renderBegin.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderBegin.pClearValues = clearValues.data();
	vkCmdBeginRenderPass(fd.command_buffer, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
	const uint32_t viewportX = sceneViewportX(context.swapchain_extent.width);
	const VkViewport viewport{
		.x = static_cast<float>(viewportX),
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width - viewportX),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor{{static_cast<int32_t>(viewportX), 0},
						   {context.swapchain_extent.width - viewportX, context.swapchain_extent.height}};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
	const VkDeviceSize vertexOffset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer.buffer, &vertexOffset);
	vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	const std::array<VkDescriptorSet, 2> mainSets{
		{sceneDescriptorSet, objects[main_object_index].descriptorSet}};
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
							static_cast<uint32_t>(mainSets.size()), mainSets.data(), 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	if (scene.model.showCompanion) {
		const std::array<VkDescriptorSet, 2> companionSets{
			{sceneDescriptorSet, objects[companion_object_index].descriptorSet}};
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
								static_cast<uint32_t>(companionSets.size()), companionSets.data(), 0,
								nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}
	if (scene.model.showFloor) {
		const std::array<VkDescriptorSet, 2> floorSets{sceneDescriptorSet, objects[2].descriptorSet};
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2,
								floorSets.data(), 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, 6, 1, index_count, 24, 0);
	}
	drawLightVisualization(fd.command_buffer);
	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to end the application command buffer\n";
		return false;
	}
	return true;
}

} // namespace application
