#include "math.hpp"
#include "octahedron.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
	if (!condition) throw std::runtime_error(message);
}

bool close(float actual, float expected) {
	return std::abs(actual - expected) < 1e-5f;
}

math::Vec3 transform(const math::Mat4& matrix, math::Vec3 point) {
	return {
		matrix.at(0, 0) * point.x + matrix.at(0, 1) * point.y + matrix.at(0, 2) * point.z + matrix.at(0, 3),
		matrix.at(1, 0) * point.x + matrix.at(1, 1) * point.y + matrix.at(1, 2) * point.z + matrix.at(1, 3),
		matrix.at(2, 0) * point.x + matrix.at(2, 1) * point.y + matrix.at(2, 2) * point.z + matrix.at(2, 3),
	};
}

void checkOctahedron() {
	const auto vertices = geometry::makeOctahedronVertices();
	const auto& indices = geometry::octahedronIndices;
	std::map<std::pair<uint16_t, uint16_t>, int> edgeUses;
	for (size_t i = 0; i < indices.size(); i += 3) {
		for (size_t side = 0; side < 3; ++side) {
			const auto a = indices[i + side];
			const auto b = indices[i + (side + 1) % 3];
			require(a < vertices.size() && b < vertices.size(), "Invalid vertex index");
			edgeUses[std::minmax(a, b)]++;
			float squaredLength = 0.0f;
			for (int axis = 0; axis < 3; ++axis) {
				const float difference = vertices[a].position[axis] - vertices[b].position[axis];
				squaredLength += difference * difference;
			}
			require(close(squaredLength, 2.0f), "Octahedron edges have unequal lengths");
		}
	}
	require(vertices.size() == 6 && indices.size() == 24 && edgeUses.size() == 12,
		"Incorrect octahedron topology");
	for (const auto& [edge, uses] : edgeUses) {
		require(uses == 2, "Octahedron is not a closed surface");
	}
	for (const auto& vertex : vertices) {
		for (float channel : vertex.color) require(channel >= 0.0f && channel <= 1.0f, "Color outside RGB range");
	}
}

void checkTransforms() {
	const auto model = math::modelMatrix({3.0f, 4.0f, 5.0f}, {0.0f, 0.0f, math::pi / 2.0f}, {2.0f, 3.0f, 4.0f});
	const auto point = transform(model, {1.0f, 0.0f, 0.0f});
	require(close(point.x, 3.0f) && close(point.y, 6.0f) && close(point.z, 5.0f),
		"Model matrix must scale, rotate, then translate");
	const auto origin = transform(math::viewMatrix(), {0.0f, 0.0f, 0.0f});
	require(close(origin.z, -5.0f), "Incorrect camera position");
	for (const auto& projection : {math::perspectiveMatrix(16.0f / 9.0f), math::orthographicMatrix(16.0f / 9.0f)}) {
		for (const auto& [z, expected] : {std::pair{-0.1f, 0.0f}, std::pair{-100.0f, 1.0f}}) {
			const float clipZ = projection.at(2, 2) * z + projection.at(2, 3);
			const float clipW = projection.at(3, 2) * z + projection.at(3, 3);
			require(close(clipZ / clipW, expected), "Projection does not map depth to Vulkan [0, 1]");
		}
		require(projection.at(1, 1) < 0.0f, "Vulkan framebuffer Y axis must be inverted");
	}
}

} // namespace

int main() {
	try {
		checkOctahedron();
		checkTransforms();
		std::cout << "Geometry and projection checks passed\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
