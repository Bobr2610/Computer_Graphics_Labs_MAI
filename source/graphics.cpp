#include "graphics.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace graphics {

bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, GpuBuffer& result) {
	auto& context = graphics::internal::context;
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						  VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo allocationResult{};
	if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
						&result.buffer, &result.allocation, &allocationResult) != VK_SUCCESS) {
		std::cerr << "Failed to allocate a Vulkan buffer\n";
		return false;
	}
	result.mapped = allocationResult.pMappedData;
	result.size = size;
	if (result.mapped == nullptr) {
		std::cerr << "Vulkan buffer was not persistently mapped\n";
		vmaDestroyBuffer(context.allocator, result.buffer, result.allocation);
		result = {};
		return false;
	}
	return true;
}

void destroyBuffer(GpuBuffer& buffer) {
	auto& context = graphics::internal::context;
	if (buffer.buffer != VK_NULL_HANDLE && context.allocator != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

bool uploadBuffer(const GpuBuffer& buffer, const void* data, size_t size) {
	if (buffer.mapped == nullptr || size > buffer.size) {
		std::cerr << "Invalid mapped-buffer write\n";
		return false;
	}
	std::memcpy(buffer.mapped, data, size);
	if (vmaFlushAllocation(internal::context.allocator, buffer.allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to flush mapped-buffer memory\n";
		return false;
	}
	return true;
}

bool immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
	auto& context = internal::context;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	poolInfo.queueFamilyIndex = context.graphics_queue_index;
	if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
		return false;
	VkCommandBuffer command = VK_NULL_HANDLE;
	VkCommandBufferAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.commandPool = pool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;
	VkResult result = vkAllocateCommandBuffers(context.device, &allocateInfo, &command);
	if (result == VK_SUCCESS) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		result = vkBeginCommandBuffer(command, &beginInfo);
		if (result == VK_SUCCESS) {
			record(command);
			result = vkEndCommandBuffer(command);
		}
	}
	if (result == VK_SUCCESS) {
		VkSubmitInfo submit{};
		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &command;
		result = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
		if (result == VK_SUCCESS)
			result = vkQueueWaitIdle(context.graphics_queue);
	}
	vkDestroyCommandPool(context.device, pool, nullptr);
	if (result != VK_SUCCESS)
		std::cerr << "Failed to submit a one-time GPU transfer\n";
	return result == VK_SUCCESS;
}

bool createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
							 GpuBuffer& result) {
	if (data == nullptr || size == 0 || size > std::numeric_limits<size_t>::max())
		return false;
	GpuBuffer staging{};
	if (!createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging))
		return false;
	if (!uploadBuffer(staging, data, static_cast<size_t>(size))) {
		destroyBuffer(staging);
		return false;
	}
	auto& context = internal::context;
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo, &result.buffer,
					&result.allocation, nullptr) != VK_SUCCESS) {
		destroyBuffer(staging);
		std::cerr << "Failed to allocate a device-local buffer\n";
		return false;
	}
	result.size = size;
	const bool copied = immediateSubmit([&](VkCommandBuffer command) {
		const VkBufferCopy region{0, 0, size};
		vkCmdCopyBuffer(command, staging.buffer, result.buffer, 1, &region);
		VkBufferMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = 0;
		if (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
			barrier.dstAccessMask |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		if (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
			barrier.dstAccessMask |= VK_ACCESS_INDEX_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = result.buffer;
		barrier.offset = 0;
		barrier.size = size;
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
							 0, 0, nullptr, 1, &barrier, 0, nullptr);
	});
	destroyBuffer(staging);
	if (!copied)
		destroyBuffer(result);
	return copied;
}

std::vector<uint32_t> readShader(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Cannot open " << filename << '\n';
		return {};
	}
	const std::streamsize byteCount = file.tellg();
	if (byteCount <= 0 || byteCount % 4 != 0) {
		std::cerr << "Invalid SPIR-V file size: " << filename << '\n';
		return {};
	}
	std::vector<uint32_t> words(static_cast<size_t>(byteCount) / sizeof(uint32_t));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(words.data()), byteCount)) {
		std::cerr << "Could not read shader: " << filename << '\n';
		return {};
	}
	return words;
}

VkShaderModule createShaderModule(const std::vector<uint32_t>& words) {
	if (words.empty()) {
		return VK_NULL_HANDLE;
	}
	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = words.size() * sizeof(uint32_t);
	info.pCode = words.data();
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create a Vulkan shader module\n";
	}
	return module;
}

} // namespace graphics
