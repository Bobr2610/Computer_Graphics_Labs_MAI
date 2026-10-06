#include "graphics.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

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
