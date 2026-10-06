#include "application.hpp"
#include "math.hpp"
#include "octahedron.hpp"
#include "graphics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>
#include <vulkan/vulkan.h>

namespace application {
namespace {

using math::Mat4;
using math::Vec3;
using geometry::Vertex;
using geometry::makeOctahedronVertices;
using geometry::octahedronIndices;
using graphics::GpuBuffer;
using graphics::createHostBuffer;
using graphics::destroyBuffer;
using graphics::uploadBuffer;

constexpr uint32_t index_count = static_cast<uint32_t>(geometry::octahedronIndices.size());
constexpr float panelWidth = 340.0f;
constexpr float viewportOffset = panelWidth + 32.0f;
constexpr uint32_t object_count = 2;

enum class ProjectionMode : int {
	perspective = 0,
	orthographic = 1,
};

struct alignas(16) SceneUniform {
	Mat4 view;
	Mat4 projection;
};

struct alignas(16) ObjectUniform {
	Mat4 model;
	float tint[4];
};

struct ObjectResources {
	GpuBuffer uniformBuffer;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

static_assert(sizeof(SceneUniform) == 128);
static_assert(sizeof(ObjectUniform) == 80);
static_assert(offsetof(ObjectUniform, tint) == 64);

struct Controls {
	ProjectionMode projection = ProjectionMode::perspective;
	std::array<float, 3> position{0.0f, 0.0f, 0.0f};
	std::array<float, 3> rotationDegrees{0.0f, 0.0f, 0.0f};
	std::array<float, 3> scale{0.85f, 0.85f, 0.85f};
	std::array<float, 3> tint{1.0f, 1.0f, 1.0f};
	float pathRadius = 0.72f;
	float pathHeight = 0.38f;
	float animationSpeed = 1.0f;
	bool animationPlaying = true;
	bool showCompanion = true;
};

Controls controls;

double animationTime = 0.0;
double previousUpdateTime = 0.0;
bool hasPreviousUpdateTime = false;

GpuBuffer sceneUniformBuffer;
GpuBuffer vertexBuffer;
GpuBuffer indexBuffer;
ObjectResources objects[object_count];

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout objectSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;

bool createDescriptorLayouts() {
	auto& context = graphics::internal::context;
	VkDescriptorSetLayoutBinding sceneBinding{};
	sceneBinding.binding = 0;
	sceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	sceneBinding.descriptorCount = 1;
	sceneBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
	sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	sceneLayoutInfo.bindingCount = 1;
	sceneLayoutInfo.pBindings = &sceneBinding;
	if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr,
										&sceneSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create the scene descriptor-set layout\n";
		return false;
	}

	VkDescriptorSetLayoutBinding objectBinding{};
	objectBinding.binding = 0;
	objectBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	objectBinding.descriptorCount = 1;
	objectBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo objectLayoutInfo{};
	objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	objectLayoutInfo.bindingCount = 1;
	objectLayoutInfo.pBindings = &objectBinding;
	if (vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr,
										&objectSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create the object descriptor-set layout\n";
		return false;
	}

	const VkDescriptorPoolSize poolSize{
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1 + object_count,
	};
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1 + object_count;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		std::cerr << "Failed to create the application descriptor pool\n";
		return false;
	}

	const std::array<VkDescriptorSetLayout, 1 + object_count> layouts{{
		sceneSetLayout, objectSetLayout, objectSetLayout,
	}};
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
	objects[0].descriptorSet = descriptorSets[1];
	objects[1].descriptorSet = descriptorSets[2];

	const std::array<VkDescriptorBufferInfo, 1 + object_count> bufferInfos{{
		{sceneUniformBuffer.buffer, 0, sizeof(SceneUniform)},
		{objects[0].uniformBuffer.buffer, 0, sizeof(ObjectUniform)},
		{objects[1].uniformBuffer.buffer, 0, sizeof(ObjectUniform)},
	}};
	std::array<VkWriteDescriptorSet, 1 + object_count> writes{};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descriptorSets[i];
		writes[i].dstBinding = 0;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[i].pBufferInfo = &bufferInfos[i];
	}
	vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	return true;
}

bool createGraphicsPipeline() {
	auto& context = graphics::internal::context;
	const std::string shaderDir = OCTAHEDRON_SHADER_DIR;
	const std::vector<uint32_t> vertexCode = graphics::readShader(shaderDir + "/octahedron.vert.spv");
	const std::vector<uint32_t> fragmentCode = graphics::readShader(shaderDir + "/octahedron.frag.spv");
	const VkShaderModule vertexModule = graphics::createShaderModule(vertexCode);
	const VkShaderModule fragmentModule = graphics::createShaderModule(fragmentCode);
	if (vertexModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		if (vertexModule != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vertexModule, nullptr);
		if (fragmentModule != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragmentModule, nullptr);
		return false;
	}

	const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertexModule,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentModule,
			.pName = "main",
		},
	}};

	const VkVertexInputBindingDescription vertexBinding{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const std::array<VkVertexInputAttributeDescription, 2> vertexAttributes{{
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
	}};
	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &vertexBinding;
	vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
	vertexInput.pVertexAttributeDescriptions = vertexAttributes.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterization{};
	rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization.polygonMode = VK_POLYGON_MODE_FILL;
	rasterization.cullMode = VK_CULL_MODE_NONE;
	rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterization.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.minDepthBounds = 0.0f;
	depthStencil.maxDepthBounds = 1.0f;

	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
									VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo colorBlend{};
	colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlend.attachmentCount = 1;
	colorBlend.pAttachments = &blendAttachment;

	const std::array<VkDynamicState, 2> dynamicStates{{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	}};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	const std::array<VkDescriptorSetLayout, 2> setLayouts{{sceneSetLayout, objectSetLayout}};
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
	layoutInfo.pSetLayouts = setLayouts.data();
	if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create the graphics pipeline layout\n";
		vkDestroyShaderModule(context.device, fragmentModule, nullptr);
		vkDestroyShaderModule(context.device, vertexModule, nullptr);
		return false;
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterization;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlend;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = context.render_pass;
	pipelineInfo.subpass = 0;

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
												 &pipelineInfo, nullptr, &graphicsPipeline);
	vkDestroyShaderModule(context.device, fragmentModule, nullptr);
	vkDestroyShaderModule(context.device, vertexModule, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create the octahedron graphics pipeline\n";
		return false;
	}
	return true;
}

void releaseResources() {
	auto& context = graphics::internal::context;
	if (context.device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(context.device);
		if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
		if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
		if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
		if (objectSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, objectSetLayout, nullptr);
		if (sceneSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
	}
	graphicsPipeline = VK_NULL_HANDLE;
	pipelineLayout = VK_NULL_HANDLE;
	descriptorPool = VK_NULL_HANDLE;
	objectSetLayout = VK_NULL_HANDLE;
	sceneSetLayout = VK_NULL_HANDLE;
	sceneDescriptorSet = VK_NULL_HANDLE;
	for (ObjectResources& object : objects) {
		object.descriptorSet = VK_NULL_HANDLE;
		destroyBuffer(object.uniformBuffer);
	}
	destroyBuffer(indexBuffer);
	destroyBuffer(vertexBuffer);
	destroyBuffer(sceneUniformBuffer);
}

void updateAnimation(double time) {
	if (!hasPreviousUpdateTime) {
		previousUpdateTime = time;
		hasPreviousUpdateTime = true;
	}
	const double elapsed = std::clamp(time - previousUpdateTime, 0.0, 0.1);
	previousUpdateTime = time;
	if (controls.animationPlaying) {
		animationTime += elapsed * static_cast<double>(controls.animationSpeed);
	}
}

void drawControls() {
	ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(panelWidth, 0.0f), ImGuiCond_Always);
	ImGui::Begin("Octahedron", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
	ImGui::PushItemWidth(180.0f);

	static const char* projectionNames[] = {"Perspective", "Orthographic"};
	int projection = static_cast<int>(controls.projection);
	if (ImGui::Combo("Projection", &projection, projectionNames, 2)) {
		controls.projection = static_cast<ProjectionMode>(projection);
	}

	ImGui::Text("Main octahedron");
	ImGui::SliderFloat3("Position", controls.position.data(), -2.0f, 2.0f);
	ImGui::SliderFloat3("Rotation (degrees)", controls.rotationDegrees.data(), -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", controls.scale.data(), 0.1f, 2.0f);
	ImGui::ColorEdit3("Vertex color tint", controls.tint.data());

	ImGui::Separator();
	ImGui::Text("Animation path");
	ImGui::Checkbox("Play animation", &controls.animationPlaying);
	ImGui::SliderFloat("Path radius", &controls.pathRadius, 0.0f, 1.6f);
	ImGui::SliderFloat("Path height", &controls.pathHeight, 0.0f, 1.0f);
	ImGui::SliderFloat("Animation speed", &controls.animationSpeed, 0.0f, 3.0f);

	ImGui::Separator();
	ImGui::Checkbox("Show second object", &controls.showCompanion);
	if (ImGui::Button("Reset scene")) {
		controls = {};
		animationTime = 0.0;
	}
	ImGui::PopItemWidth();
	ImGui::End();
}

bool updateUniforms() {
	auto& context = graphics::internal::context;
	const float aspect = std::max(1.0f, static_cast<float>(context.swapchain_extent.width) - viewportOffset) /
					 static_cast<float>(context.swapchain_extent.height);
	SceneUniform scene{};
	scene.view = math::viewMatrix();
	scene.projection = controls.projection == ProjectionMode::perspective
			? math::perspectiveMatrix(aspect)
			: math::orthographicMatrix(aspect);
	if (!uploadBuffer(sceneUniformBuffer, &scene, sizeof(scene))) return false;

	const float t = static_cast<float>(animationTime);
	const Vec3 animationOffset{
		controls.pathRadius * std::cos(t),
		controls.pathHeight * std::sin(2.0f * t),
		controls.pathRadius * 0.7f * std::sin(t),
	};
	const Vec3 mainPosition{
		controls.position[0] + animationOffset.x,
		controls.position[1] + animationOffset.y,
		controls.position[2] + animationOffset.z,
	};
	const Vec3 baseRotation = math::radiansFromDegrees(controls.rotationDegrees);
	const Vec3 mainRotation{
		baseRotation.x + 0.35f * std::sin(t),
		baseRotation.y + 0.65f * t,
		baseRotation.z + 0.25f * std::sin(2.0f * t),
	};
	const Vec3 mainScale{controls.scale[0], controls.scale[1], controls.scale[2]};

	ObjectUniform mainUniform{};
	mainUniform.model = math::modelMatrix(mainPosition, mainRotation, mainScale);
	mainUniform.tint[0] = controls.tint[0];
	mainUniform.tint[1] = controls.tint[1];
	mainUniform.tint[2] = controls.tint[2];
	mainUniform.tint[3] = 1.0f;
	if (!uploadBuffer(objects[0].uniformBuffer, &mainUniform, sizeof(mainUniform))) return false;

	const Vec3 companionPosition{-1.65f, 0.0f, -0.15f};
	const Vec3 companionRotation{0.0f, -0.45f * t, 0.35f};
	const Vec3 companionScale{0.52f, 0.52f, 0.52f};
	ObjectUniform companionUniform{};
	companionUniform.model = math::modelMatrix(companionPosition, companionRotation, companionScale);
	companionUniform.tint[0] = 0.45f;
	companionUniform.tint[1] = 0.8f;
	companionUniform.tint[2] = 1.0f;
	companionUniform.tint[3] = 1.0f;
	return uploadBuffer(objects[1].uniformBuffer, &companionUniform, sizeof(companionUniform));
}

} // namespace

bool initialize() {
	const auto fail = []() {
		releaseResources();
		return false;
	};

	const auto vertices = makeOctahedronVertices();
	if (!createHostBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer)) return fail();
	if (!uploadBuffer(vertexBuffer, vertices.data(), sizeof(vertices))) return fail();
	if (!createHostBuffer(sizeof(octahedronIndices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer)) return fail();
	if (!uploadBuffer(indexBuffer, octahedronIndices.data(), sizeof(octahedronIndices))) return fail();

	if (!createHostBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUniformBuffer)) return fail();
	for (ObjectResources& object : objects) {
		if (!createHostBuffer(sizeof(ObjectUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						  object.uniformBuffer)) return fail();
	}
	if (!createDescriptorLayouts()) return fail();
	if (!createGraphicsPipeline()) return fail();

	return true;
}

void shutdown() {
	releaseResources();
}

void update(double time) {
	updateAnimation(time);
	drawControls();
}

bool render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (graphicsPipeline == VK_NULL_HANDLE || fd.command_buffer == VK_NULL_HANDLE ||
		context.swapchain_extent.width <= static_cast<uint32_t>(viewportOffset) || context.swapchain_extent.height == 0) {
		return false;
	}

	if (!updateUniforms()) return false;

	if (vkResetCommandBuffer(fd.command_buffer, 0) != VK_SUCCESS) {
		std::cerr << "Failed to reset the drawing command buffer\n";
		return false;
	}

	VkCommandBufferBeginInfo commandBegin{};
	commandBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	if (vkBeginCommandBuffer(fd.command_buffer, &commandBegin) != VK_SUCCESS) {
		std::cerr << "Failed to begin the application command buffer\n";
		return false;
	}

	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color.float32[0] = 0.025f;
	clearValues[0].color.float32[1] = 0.035f;
	clearValues[0].color.float32[2] = 0.055f;
	clearValues[0].color.float32[3] = 1.0f;
	clearValues[1].depthStencil.depth = 1.0f;
	clearValues[1].depthStencil.stencil = 0;
	VkRenderPassBeginInfo renderBegin{};
	renderBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderBegin.renderPass = context.render_pass;
	renderBegin.framebuffer = fd.framebuffer;
	renderBegin.renderArea.extent = context.swapchain_extent;
	renderBegin.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderBegin.pClearValues = clearValues.data();
	vkCmdBeginRenderPass(fd.command_buffer, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
	const VkViewport viewport{
		.x = viewportOffset,
		.y = 0.0f,
		.width = std::max(1.0f, static_cast<float>(context.swapchain_extent.width) - viewportOffset),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor{
		{static_cast<int32_t>(viewportOffset), 0},
		{context.swapchain_extent.width - static_cast<uint32_t>(viewportOffset), context.swapchain_extent.height},
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	const VkDeviceSize vertexOffset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer.buffer, &vertexOffset);
	vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	const std::array<VkDescriptorSet, 2> mainSets{{sceneDescriptorSet, objects[0].descriptorSet}};
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
						0, static_cast<uint32_t>(mainSets.size()), mainSets.data(), 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

	if (controls.showCompanion) {
		const std::array<VkDescriptorSet, 2> companionSets{{sceneDescriptorSet, objects[1].descriptorSet}};
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
							0, static_cast<uint32_t>(companionSets.size()), companionSets.data(), 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to end the application command buffer\n";
		return false;
	}
	return true;
}

} // namespace application
