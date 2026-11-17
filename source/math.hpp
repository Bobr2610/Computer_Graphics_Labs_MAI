#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace math {

inline constexpr float pi = std::numbers::pi_v<float>;

inline constexpr size_t dynamicUniformStride(size_t recordSize, size_t alignment) {
	return alignment == 0 ? recordSize : ((recordSize + alignment - 1) / alignment) * alignment;
}

struct Vec3 {
	float x;
	float y;
	float z;
};

inline Vec3 add(const Vec3 &left, const Vec3 &right) {
	return {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline Vec3 subtract(const Vec3 &left, const Vec3 &right) {
	return {left.x - right.x, left.y - right.y, left.z - right.z};
}

inline Vec3 multiply(const Vec3 &value, float scalar) {
	return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline float dot(const Vec3 &left, const Vec3 &right) {
	return left.x * right.x + left.y * right.y + left.z * right.z;
}

inline Vec3 cross(const Vec3 &left, const Vec3 &right) {
	return {
		left.y * right.z - left.z * right.y,
		left.z * right.x - left.x * right.z,
		left.x * right.y - left.y * right.x,
	};
}

inline Vec3 normalize(const Vec3 &value) {
	const float length = std::sqrt(dot(value, value));
	return length > 1.0e-6f ? multiply(value, 1.0f / length) : Vec3{0.0f, 0.0f, -1.0f};
}

struct Mat4 {
	// Columns are contiguous, matching GLSL mat4 in a uniform buffer.
	std::array<float, 16> value{};

	float &at(size_t row, size_t column) {
		return value[column * 4 + row];
	}

	const float &at(size_t row, size_t column) const {
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

inline Mat4 multiply(const Mat4 &left, const Mat4 &right) {
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
	return translationMatrix(0.0f, 0.0f, -5.0f);
}

inline Mat4 viewMatrix(const Vec3 &cameraPosition, float yawRadians, float pitchRadians) {
	const float cosinePitch = std::cos(pitchRadians);
	const Vec3 forward = normalize({
		cosinePitch * std::sin(yawRadians),
		std::sin(pitchRadians),
		-cosinePitch * std::cos(yawRadians),
	});
	const Vec3 worldUp{0.0f, 1.0f, 0.0f};
	const Vec3 right = normalize(cross(forward, worldUp));
	const Vec3 up = cross(right, forward);

	Mat4 result = identityMatrix();
	result.at(0, 0) = right.x;
	result.at(0, 1) = right.y;
	result.at(0, 2) = right.z;
	result.at(0, 3) = -dot(right, cameraPosition);
	result.at(1, 0) = up.x;
	result.at(1, 1) = up.y;
	result.at(1, 2) = up.z;
	result.at(1, 3) = -dot(up, cameraPosition);
	result.at(2, 0) = -forward.x;
	result.at(2, 1) = -forward.y;
	result.at(2, 2) = -forward.z;
	result.at(2, 3) = dot(forward, cameraPosition);
	return result;
}

inline Mat4 perspectiveMatrix(float aspect, float fieldOfViewDegrees = 50.0f) {
	const float nearPlane = 0.1f;
	const float farPlane = 100.0f;
	const float fieldOfView = std::clamp(fieldOfViewDegrees, 15.0f, 110.0f) * pi / 180.0f;
	const float focal = 1.0f / std::tan(0.5f * fieldOfView);
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

inline Mat4 modelMatrix(const Vec3 &position, const Vec3 &rotationRadians, const Vec3 &scale) {
	const Mat4 translation = translationMatrix(position.x, position.y, position.z);
	const Mat4 rz = rotationZ(rotationRadians.z);
	const Mat4 ry = rotationY(rotationRadians.y);
	const Mat4 rx = rotationX(rotationRadians.x);
	const Mat4 scaleTransform = scaleMatrix(scale.x, scale.y, scale.z);
	return multiply(multiply(multiply(multiply(translation, rz), ry), rx), scaleTransform);
}

inline Vec3 radiansFromDegrees(const std::array<float, 3> &degrees) {
	return {
		degrees[0] * pi / 180.0f,
		degrees[1] * pi / 180.0f,
		degrees[2] * pi / 180.0f,
	};
}

inline Vec3 transformPoint(const Mat4 &matrix, const Vec3 &point) {
	return {
		matrix.at(0, 0) * point.x + matrix.at(0, 1) * point.y + matrix.at(0, 2) * point.z + matrix.at(0, 3),
		matrix.at(1, 0) * point.x + matrix.at(1, 1) * point.y + matrix.at(1, 2) * point.z + matrix.at(1, 3),
		matrix.at(2, 0) * point.x + matrix.at(2, 1) * point.y + matrix.at(2, 2) * point.z + matrix.at(2, 3),
	};
}

} // namespace math
