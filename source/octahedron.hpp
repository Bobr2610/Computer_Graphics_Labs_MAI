#pragma once

#include <array>
#include <cstdint>

#include "math.hpp"

namespace geometry {

using math::Vec3;

struct Vertex {
	float position[3];
	float color[3];
};

inline std::array<Vertex, 6> makeOctahedronVertices() {
	const std::array<Vec3, 6> positions{{
		{0.0f, 1.0f, 0.0f},
		{1.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, 1.0f},
		{-1.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, -1.0f},
		{0.0f, -1.0f, 0.0f},
	}};

	std::array<Vertex, 6> vertices{};
	for (size_t i = 0; i < positions.size(); ++i) {
		vertices[i] = {
			.position = {positions[i].x, positions[i].y, positions[i].z},
			.color = {0.5f + 0.5f * positions[i].x,
					  0.5f + 0.5f * positions[i].y,
					  0.5f + 0.5f * positions[i].z},
		};
	}
	return vertices;
}

constexpr std::array<uint16_t, 24> octahedronIndices{{
	0, 2, 1,  0, 3, 2,  0, 4, 3,  0, 1, 4,
	5, 1, 2,  5, 2, 3,  5, 3, 4,  5, 4, 1,
}};

} // namespace geometry
