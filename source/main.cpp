#include <climits>
#include <cstdint>
#include <filesystem>

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <imgui.h>

#include "application.hpp"
#include "graphics_internal.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "Lab 2 - Camera and Lighting";

GLFWwindow *glfw_window;

} // namespace

int main() {
	int status = EXIT_SUCCESS;

	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window =
		glfwCreateWindow(default_window_width, default_window_height, default_window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	glfwSetWindowSizeLimits(glfw_window, 800, 600, GLFW_DONT_CARE, GLFW_DONT_CARE);

	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow *, int width, int height) {
		if (width == 0 || height == 0) {
			return;
		}

		graphics::internal::resize(width, height);
	});

	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}

	{
		ImFontConfig fontConfig{};
		fontConfig.SizePixels = 18.0f;
		if (std::filesystem::exists("C:/Windows/Fonts/segoeui.ttf")) {
			ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 18.0f);
		} else {
			ImGui::GetIO().Fonts->AddFontDefault(&fontConfig);
		}
		ImGui::StyleColorsDark();
		ImGuiStyle &style = ImGui::GetStyle();
		style.FramePadding = ImVec2(6, 5);
		style.ItemSpacing = ImVec2(8, 10);
		style.WindowPadding = ImVec2(12, 12);
		style.FrameRounding = 3.0f;
		style.Colors[ImGuiCol_WindowBg] = ImVec4(0.065f, 0.075f, 0.095f, 1.0f);
		style.Colors[ImGuiCol_ChildBg] = style.Colors[ImGuiCol_WindowBg];
		style.Colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.15f, 0.19f, 1.0f);
		style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.17f, 0.23f, 0.29f, 1.0f);
		style.Colors[ImGuiCol_Header] = ImVec4(0.13f, 0.25f, 0.34f, 1.0f);
		style.Colors[ImGuiCol_Button] = ImVec4(0.13f, 0.25f, 0.34f, 1.0f);
	}

	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}

	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}

	if (!application::initialize(glfw_window)) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

	while (!glfwWindowShouldClose(glfw_window)) {
		const double time = glfwGetTime();

		glfwPollEvents();
		int width = 0;
		int height = 0;
		glfwGetFramebufferSize(glfw_window, &width, &height);
		if (width == 0 || height == 0) {
			glfwWaitEvents();
			continue;
		}
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
		application::update(time);
		ImGui::Render();

		graphics::internal::FrameData fd = graphics::internal::prepare();
		if (!application::render(fd) || !graphics::internal::submitAndPresent()) {
			status = EXIT_FAILURE;
			break;
		}
	}

	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();

	return status;
}
