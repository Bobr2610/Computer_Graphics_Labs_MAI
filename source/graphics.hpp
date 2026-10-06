#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "graphics_internal.hpp"

namespace graphics {

struct GpuBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, GpuBuffer& result);
void destroyBuffer(GpuBuffer& buffer);
bool uploadBuffer(const GpuBuffer& buffer, const void* data, size_t size);

std::vector<uint32_t> readShader(const std::string& filename);
VkShaderModule createShaderModule(const std::vector<uint32_t>& words);

} // namespace graphics
