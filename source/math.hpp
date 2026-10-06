#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace math {

inline constexpr float pi = std::numbers::pi_v<float>;

struct Vec3 {
	float x;
	float y;
	float z;
};

struct Mat4 {
	// Columns are contiguous, matching GLSL mat4 in a uniform buffer.
	std::array<float, 16> value{};

	float& at(size_t row, size_t column) {
		return value[column * 4 + row];
	}

	const float& at(size_t row, size_t column) const {
		return value[column * 4 + row];
	}
};

inline Mat4 identityMatrix() {
	Mat4 result{};
	result.at(0, 0) = 1.0f;
	result.at(1, 1) = 1.0f;
	result.at(2, 2) = 1.0f;
	result.at(3, 3) = 1.0f;
	return result;
}

inline Mat4 multiply(const Mat4& left, const Mat4& right) {
	Mat4 result{};
	for (size_t column = 0; column < 4; ++column) {
		for (size_t row = 0; row < 4; ++row) {
			for (size_t k = 0; k < 4; ++k) {
				result.at(row, column) += left.at(row, k) * right.at(k, column);
			}
		}
	}
	return result;
}

inline Mat4 translationMatrix(float x, float y, float z) {
	Mat4 result = identityMatrix();
	result.at(0, 3) = x;
	result.at(1, 3) = y;
	result.at(2, 3) = z;
	return result;
}

inline Mat4 scaleMatrix(float x, float y, float z) {
	Mat4 result{};
	result.at(0, 0) = x;
	result.at(1, 1) = y;
	result.at(2, 2) = z;
	result.at(3, 3) = 1.0f;
	return result;
}

inline Mat4 rotationX(float radians) {
	Mat4 result = identityMatrix();
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	result.at(1, 1) = c;
	result.at(2, 1) = s;
	result.at(1, 2) = -s;
	result.at(2, 2) = c;
	return result;
}

inline Mat4 rotationY(float radians) {
	Mat4 result = identityMatrix();
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	result.at(0, 0) = c;
	result.at(2, 0) = -s;
	result.at(0, 2) = s;
	result.at(2, 2) = c;
	return result;
}

inline Mat4 rotationZ(float radians) {
	Mat4 result = identityMatrix();
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	result.at(0, 0) = c;
	result.at(1, 0) = s;
	result.at(0, 1) = -s;
	result.at(1, 1) = c;
	return result;
}

inline Mat4 viewMatrix() {
	// The camera is at (0, 0, 5) and looks along the negative Z axis.
	return translationMatrix(0.0f, 0.0f, -5.0f);
}

inline Mat4 perspectiveMatrix(float aspect) {
	const float nearPlane = 0.1f;
	const float farPlane = 100.0f;
	const float focal = std::min(1.0f, aspect) / std::tan(0.5f * (50.0f * pi / 180.0f));
	Mat4 result{};
	result.at(0, 0) = focal / aspect;
	result.at(1, 1) = -focal;
	result.at(2, 2) = farPlane / (nearPlane - farPlane);
	result.at(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
	result.at(3, 2) = -1.0f;
	return result;
}

inline Mat4 orthographicMatrix(float aspect) {
	const float halfHeight = 2.6f / std::min(1.0f, aspect);
	const float halfWidth = halfHeight * aspect;
	const float nearPlane = 0.1f;
	const float farPlane = 100.0f;
	Mat4 result = identityMatrix();
	result.at(0, 0) = 1.0f / halfWidth;
	result.at(1, 1) = -1.0f / halfHeight;
	result.at(2, 2) = 1.0f / (nearPlane - farPlane);
	result.at(2, 3) = nearPlane / (nearPlane - farPlane);
	return result;
}

inline Mat4 modelMatrix(const Vec3& position, const Vec3& rotationRadians, const Vec3& scale) {
	const Mat4 translation = translationMatrix(position.x, position.y, position.z);
	const Mat4 rz = rotationZ(rotationRadians.z);
	const Mat4 ry = rotationY(rotationRadians.y);
	const Mat4 rx = rotationX(rotationRadians.x);
	const Mat4 scaleTransform = scaleMatrix(scale.x, scale.y, scale.z);
	return multiply(multiply(multiply(multiply(translation, rz), ry), rx), scaleTransform);
}

inline Vec3 radiansFromDegrees(const std::array<float, 3>& degrees) {
	return {
		degrees[0] * pi / 180.0f,
		degrees[1] * pi / 180.0f,
		degrees[2] * pi / 180.0f,
	};
}

} // namespace math
