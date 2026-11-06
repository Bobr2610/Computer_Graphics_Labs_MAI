#pragma once

#include "math.hpp"
#include <array>
#include <cstdint>

namespace geometry {

struct Vertex {
	float position[3];
	float color[3];
	float normal[3];
};

inline constexpr std::array<math::Vec3, 6> octahedronPositions{{
	{0, 1, 0},
	{1, 0, 0},
	{0, 0, 1},
	{-1, 0, 0},
	{0, 0, -1},
	{0, -1, 0},
}};

inline constexpr std::array<uint16_t, 24> octahedronFaces{{
	0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1,
}};

inline std::array<Vertex, 24> makeOctahedronVertices() {
	std::array<Vertex, 24> vertices{};
	for (size_t face = 0; face < octahedronFaces.size(); face += 3) {
		const auto a = octahedronPositions[octahedronFaces[face]];
		const auto b = octahedronPositions[octahedronFaces[face + 1]];
		const auto c = octahedronPositions[octahedronFaces[face + 2]];
		const auto normal = math::normalize(math::cross(math::subtract(b, a), math::subtract(c, a)));
		for (size_t corner = 0; corner < 3; ++corner) {
			const auto p = octahedronPositions[octahedronFaces[face + corner]];
			vertices[face + corner] = {
				{p.x, p.y, p.z},
				{0.5f + 0.5f * p.x, 0.5f + 0.5f * p.y, 0.5f + 0.5f * p.z},
				{normal.x, normal.y, normal.z},
			};
		}
	}
	return vertices;
}

inline constexpr std::array<uint16_t, 24> octahedronIndices = [] {
	std::array<uint16_t, 24> indices{};
	for (uint16_t i = 0; i < indices.size(); ++i)
		indices[i] = i;
	return indices;
}();

} // namespace geometry
