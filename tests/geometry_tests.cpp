#include "math.hpp"
#include "octahedron.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}

bool close(float actual, float expected) {
	return std::abs(actual - expected) < 1e-5f;
}

math::Vec3 transform(const math::Mat4 &matrix, math::Vec3 point) {
	return {
		matrix.at(0, 0) * point.x + matrix.at(0, 1) * point.y + matrix.at(0, 2) * point.z + matrix.at(0, 3),
		matrix.at(1, 0) * point.x + matrix.at(1, 1) * point.y + matrix.at(1, 2) * point.z + matrix.at(1, 3),
		matrix.at(2, 0) * point.x + matrix.at(2, 1) * point.y + matrix.at(2, 2) * point.z + matrix.at(2, 3),
	};
}

math::Vec3 projectToNdc(const math::Mat4 &view, const math::Mat4 &projection, const math::Vec3 &point) {
	const math::Vec3 eye = transform(view, point);
	const math::Vec3 clip = transform(projection, eye);
	const float w = projection.at(3, 2) * eye.z + projection.at(3, 3);
	return {clip.x / w, clip.y / w, clip.z / w};
}

void checkOctahedron() {
	const auto &positions = geometry::octahedronPositions;
	const auto &indices = geometry::octahedronFaces;
	std::map<std::pair<uint16_t, uint16_t>, int> edgeUses;
	for (size_t i = 0; i < indices.size(); i += 3) {
		for (size_t side = 0; side < 3; ++side) {
			const auto a = indices[i + side];
			const auto b = indices[i + (side + 1) % 3];
			require(a < positions.size() && b < positions.size(), "Invalid vertex index");
			edgeUses[std::minmax(a, b)]++;
			float squaredLength = 0.0f;
			for (int axis = 0; axis < 3; ++axis) {
				const float av[] = {positions[a].x, positions[a].y, positions[a].z};
				const float bv[] = {positions[b].x, positions[b].y, positions[b].z};
				const float difference = av[axis] - bv[axis];
				squaredLength += difference * difference;
			}
			require(close(squaredLength, 2.0f), "Octahedron edges have unequal lengths");
		}
	}
	require(positions.size() == 6 && indices.size() == 24 && edgeUses.size() == 12,
			"Incorrect octahedron topology");
	for (const auto &[edge, uses] : edgeUses) {
		require(uses == 2, "Octahedron is not a closed surface");
	}
	const auto vertices = geometry::makeOctahedronVertices();
	for (const auto &vertex : vertices) {
		const math::Vec3 normal{vertex.normal[0], vertex.normal[1], vertex.normal[2]};
		const math::Vec3 position{vertex.position[0], vertex.position[1], vertex.position[2]};
		require(close(math::dot(normal, normal), 1.0f), "Non-unit face normal");
		require(math::dot(normal, position) > 0.0f, "Face normal must point outward");
		require(vertex.texCoord[0] >= 0.0f && vertex.texCoord[0] <= 1.0f &&
				vertex.texCoord[1] >= 0.0f && vertex.texCoord[1] <= 1.0f,
				"Generated texture coordinates are outside the wrapped octahedron map");
	}
	require(math::dynamicUniformStride(96, 256) == 256 &&
			math::dynamicUniformStride(96, 16) == 96,
			"Dynamic uniform records must respect the device alignment");
	for (size_t face = 0; face < vertices.size(); face += 3) {
		const auto& a = vertices[face].texCoord;
		const auto& b = vertices[face + 1].texCoord;
		const auto& c = vertices[face + 2].texCoord;
		const float uvArea = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
		require(std::abs(uvArea) > 0.5f,
				"An octahedron face needs a non-degenerate, independent UV island");
	}
}

void checkTransforms() {
	const auto model =
		math::modelMatrix({3.0f, 4.0f, 5.0f}, {0.0f, 0.0f, math::pi / 2.0f}, {2.0f, 3.0f, 4.0f});
	const auto point = transform(model, {1.0f, 0.0f, 0.0f});
	require(close(point.x, 3.0f) && close(point.y, 6.0f) && close(point.z, 5.0f),
			"Model matrix must scale, rotate, then translate");
	const auto movedView = math::viewMatrix({3.0f, 2.0f, 5.0f}, 0.4f, -0.2f);
	const auto cameraOrigin = transform(movedView, {3.0f, 2.0f, 5.0f});
	require(close(cameraOrigin.x, 0) && close(cameraOrigin.y, 0) && close(cameraOrigin.z, 0),
			"View matrix must map camera position to origin");
	const math::Vec3 forward{std::cos(-0.2f) * std::sin(0.4f), std::sin(-0.2f),
							 -std::cos(-0.2f) * std::cos(0.4f)};
	const auto inFront = transform(movedView, math::add({3.0f, 2.0f, 5.0f}, forward));
	require(close(inFront.x, 0) && close(inFront.y, 0) && close(inFront.z, -1),
			"Camera forward axis is wrong");
	const auto origin = transform(math::viewMatrix(), {0.0f, 0.0f, 0.0f});
	require(close(origin.z, -5.0f), "Incorrect camera position");
	for (const auto &projection :
		 {math::perspectiveMatrix(16.0f / 9.0f), math::orthographicMatrix(16.0f / 9.0f)}) {
		for (const auto &[z, expected] : {std::pair{-0.1f, 0.0f}, std::pair{-100.0f, 1.0f}}) {
			const float clipZ = projection.at(2, 2) * z + projection.at(2, 3);
			const float clipW = projection.at(3, 2) * z + projection.at(3, 3);
			require(close(clipZ / clipW, expected), "Projection does not map depth to Vulkan [0, 1]");
		}
		require(projection.at(1, 1) < 0.0f, "Vulkan framebuffer Y axis must be inverted");
	}
}

// Both projections flip the y axis for the y-down screen space, so the faces
// turned to the camera have a negative signed area in framebuffer coordinates,
// which Vulkan classifies as counterclockwise when selecting front faces.
void checkWinding() {
	const auto vertices = geometry::makeOctahedronVertices();
	const auto &indices = geometry::octahedronIndices;
	const math::Mat4 view = math::viewMatrix();
	for (const auto &projection :
		 {math::perspectiveMatrix(16.0f / 9.0f), math::orthographicMatrix(16.0f / 9.0f)}) {
		const auto ndc = [&](int corner) {
			const auto &position = vertices[indices[corner]].position;
			return projectToNdc(view, projection, math::Vec3{position[0], position[1], position[2]});
		};
		const math::Vec3 a = ndc(0);
		const math::Vec3 b = ndc(1);
		const math::Vec3 c = ndc(2);
		const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
		require(area < 0.0f, "The visible faces must be clockwise in framebuffer coordinates");
	}
}

} // namespace

int main() {
	try {
		checkOctahedron();
		checkTransforms();
		checkWinding();
		std::cout << "Geometry and projection checks passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
