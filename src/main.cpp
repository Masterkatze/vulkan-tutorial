#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include "stb_image.h"
#include "tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <array>
#include <optional>
#include <cstring>
#include <chrono>
#include <unordered_map>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::string MODEL_PATH = "resources/models/viking_room.obj";
const std::string TEXTURE_PATH = "resources/textures/viking_room.png";

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

#define CHECK_VKRESULT(RESULT, MESSAGE)                                             \
if(RESULT != VK_SUCCESS)                                                            \
{                                                                                   \
	throw std::runtime_error(MESSAGE" (VkResult: " + std::to_string(RESULT) + ")"); \
}

const std::vector<const char*> validation_layers =
{
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> device_extensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *create_info,
									  const VkAllocationCallbacks *allocator, VkDebugUtilsMessengerEXT *debug_messenger)
{
	auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	return func ? func(instance, create_info, allocator, debug_messenger) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks *allocator)
{
	auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	if(func != nullptr)
	{
		func(instance, debug_messenger, allocator);
	}
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool IsComplete()
	{
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities{};
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
	glm::vec2 texture_coordinates;

	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription binding_description{};
		binding_description.binding = 0;
		binding_description.stride = sizeof(Vertex);
		binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return binding_description;
	}

	static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions{};

		attribute_descriptions[0].binding = 0;
		attribute_descriptions[0].location = 0;
		attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[0].offset = offsetof(Vertex, position);

		attribute_descriptions[1].binding = 0;
		attribute_descriptions[1].location = 1;
		attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[1].offset = offsetof(Vertex, color);

		attribute_descriptions[2].binding = 0;
		attribute_descriptions[2].location = 2;
		attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attribute_descriptions[2].offset = offsetof(Vertex, texture_coordinates);

		return attribute_descriptions;
	}

	bool operator==(const Vertex& other) const
	{
		return position == other.position && color == other.color && texture_coordinates == other.texture_coordinates;
	}
};

namespace std {
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texture_coordinates) << 1);
		}
	};
}

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 projection;
};

class Application
{
public:
	Application() = default;

	void Run()
	{
		InitWindow();
		InitVulkan();
		MainLoop();
		Cleanup();
	}

private:
	GLFWwindow *window;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;
	VkDevice device;

	VkQueue graphics_queue;
	VkQueue present_queue;

	VkSwapchainKHR swap_chain;
	std::vector<VkImage> swap_chain_images;
	VkFormat swap_chain_image_format;
	VkExtent2D swap_chain_extent;
	std::vector<VkImageView> swap_chain_image_views;
	std::vector<VkFramebuffer> swap_chain_framebuffers;

	VkRenderPass render_pass;
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;

	uint32_t mip_levels;
	VkImage texture_image;
	VkDeviceMemory texture_image_memory;
	VkImageView texture_image_view;
	VkSampler texture_sampler;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertex_buffer;
	VkDeviceMemory vertex_buffer_memory;
	VkBuffer index_buffer;
	VkDeviceMemory index_buffer_memory;

	VkImage depth_image;
	VkDeviceMemory depth_image_memory;
	VkImageView depth_image_view;

	VkImage color_image;
	VkDeviceMemory color_image_memory;
	VkImageView color_image_view;

	std::vector<VkBuffer> uniform_buffers;
	std::vector<VkDeviceMemory> uniform_buffers_memory;

	VkDescriptorPool descriptor_pool;
	std::vector<VkDescriptorSet> descriptor_sets;

	VkCommandPool command_pool;
	std::vector<VkCommandBuffer> command_buffers;

	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;
	std::vector<VkFence> images_in_flight;
	size_t current_frame = 0;

	bool framebuffer_resized = false;

	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		#pragma unused(width)
		#pragma unused(height)

		auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		app->framebuffer_resized = true;
	}

	void InitWindow()
	{
		glfwInit();

		// Don't create an OpenGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, FramebufferResizeCallback);
		glfwSetWindowSizeLimits(window, 256, 256, INT32_MAX, INT32_MAX);
	}

	void InitVulkan()
	{
		CreateInstance();
		SetupDebugMessenger();
		CreateSurface();
		PickPhysicalDevice();
		CreateLogicalDevice();
		CreateSwapChain();
		CreateImageViews();
		CreateRenderPass();
		CreateDescriptorSetLayout();
		CreateGraphicsPipeline();
		CreateCommandPool();
		CreateColorResources();
		CreateDepthResources();
		CreateFramebuffers();
		CreateTextureImage();
		CreateTextureImageView();
		CreateTextureSampler();
		LoadModel();
		CreateVertexBuffer();
		CreateIndexBuffer();
		CreateUniformBuffers();
		CreateDescriptorPool();
		CreateDescriptorSets();
		CreateCommandBuffers();
		CreateSyncObjects();
	}

	void MainLoop()
	{
		while(!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			DrawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	void CleanupSwapChain()
	{
		vkDestroyImageView(device, depth_image_view, nullptr);
		vkDestroyImage(device, depth_image, nullptr);
		vkFreeMemory(device, depth_image_memory, nullptr);

		vkDestroyImageView(device, color_image_view, nullptr);
		vkDestroyImage(device, color_image, nullptr);
		vkFreeMemory(device, color_image_memory, nullptr);

		for(const auto& framebuffer : swap_chain_framebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		vkFreeCommandBuffers(device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

		vkDestroyPipeline(device, graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyRenderPass(device, render_pass, nullptr);

		for(const auto& image_view : swap_chain_image_views)
		{
			vkDestroyImageView(device, image_view, nullptr);
		}

		vkDestroySwapchainKHR(device, swap_chain, nullptr);

		for(size_t i = 0; i < swap_chain_images.size(); i++)
		{
			vkDestroyBuffer(device, uniform_buffers[i], nullptr);
			vkFreeMemory(device, uniform_buffers_memory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	}

	void Cleanup()
	{
		CleanupSwapChain();

		vkDestroySampler(device, texture_sampler, nullptr);
		vkDestroyImageView(device, texture_image_view, nullptr);

		vkDestroyImage(device, texture_image, nullptr);
		vkFreeMemory(device, texture_image_memory, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);

		vkDestroyBuffer(device, index_buffer, nullptr);
		vkFreeMemory(device, index_buffer_memory, nullptr);

		vkDestroyBuffer(device, vertex_buffer, nullptr);
		vkFreeMemory(device, vertex_buffer_memory, nullptr);

		for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
			vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
			vkDestroyFence(device, in_flight_fences[i], nullptr);
		}

		vkDestroyCommandPool(device, command_pool, nullptr);

		vkDestroyDevice(device, nullptr);

		if(enable_validation_layers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void RecreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while(width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		CleanupSwapChain();

		CreateSwapChain();
		CreateImageViews();
		CreateRenderPass();
		CreateGraphicsPipeline();
		CreateColorResources();
		CreateDepthResources();
		CreateFramebuffers();
		CreateUniformBuffers();
		CreateDescriptorPool();
		CreateDescriptorSets();
		CreateCommandBuffers();
	}

	void CreateInstance()
	{
		if(enable_validation_layers && !CheckValidationLayerSupport())
		{
			throw std::runtime_error("Validation layers requested, but not available!");
		}

		VkApplicationInfo app_info{};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "Vulkan Tutorial";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "No Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;

		auto required_extensions = GetRequiredExtensions();
		create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
		create_info.ppEnabledExtensionNames = required_extensions.data();

		std::cout << "Required extensions:" << std::endl;
		for(const auto& extension : required_extensions)
		{
			std::cout << '\t' << extension << std::endl;
		}
		std::cout << std::endl;

		auto available_extensions = GetAvailableExtensions();

		std::cout << "Available extensions:" << std::endl;
		for(const auto& extension : available_extensions)
		{
			std::cout << '\t' << extension.extensionName << " (version " << extension.specVersion << ")" << std::endl;
		}
		std::cout << std::endl;

		VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
		if(enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();

			PopulateDebugMessengerCreateInfo(debug_create_info);
			create_info.pNext = reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&debug_create_info);
		}
		else
		{
			create_info.enabledLayerCount = 0;
			create_info.pNext = nullptr;
		}

		auto result = vkCreateInstance(&create_info, nullptr, &instance);
		CHECK_VKRESULT(result, "Failed to create instance");
	}

	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info)
	{
		create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		create_info.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = DebugCallback;
	}

	void SetupDebugMessenger()
	{
		if(!enable_validation_layers) return;

		VkDebugUtilsMessengerCreateInfoEXT create_info;
		PopulateDebugMessengerCreateInfo(create_info);

		auto result = CreateDebugUtilsMessengerEXT(instance, &create_info, nullptr, &debug_messenger);
		CHECK_VKRESULT(result, "Failed to set up debug messenger");
	}

	void CreateSurface()
	{
		auto result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
		CHECK_VKRESULT(result, "Failed to create window surface");
	}

	void PickPhysicalDevice()
	{
		uint32_t device_count = 0;
		auto result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
		CHECK_VKRESULT(result, "Failed to enumerate physucal devices");

		if(device_count == 0)
		{
			throw std::runtime_error("Failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> physical_devices(device_count);
		result = vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
		CHECK_VKRESULT(result, "Failed to enumerate physucal devices");

		auto it = std::find_if(physical_devices.begin(), physical_devices.end(), [this](const auto& d)
		{
			return IsDeviceSuitable(d);
		});

		if(it == physical_devices.end())
		{
			throw std::runtime_error("Failed to find a suitable GPU!");
		}

		physical_device = *it;
		msaa_samples = GetMaxUsableSampleCount();
	}

	bool IsDeviceSuitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices queue_family_indices = FindQueueFamilies(device);
		bool extensions_supported = CheckDeviceExtensionSupport(device);

		bool swap_chain_adequate = false;
		if(extensions_supported)
		{
			SwapChainSupportDetails swap_chain_support = QuerySwapChainSupport(device);
			swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
		}

		VkPhysicalDeviceFeatures supported_features;
		vkGetPhysicalDeviceFeatures(device, &supported_features);

		return queue_family_indices.IsComplete() &&
				extensions_supported &&
				swap_chain_adequate &&
				supported_features.samplerAnisotropy;
	}

	bool CheckDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extension_count;
		auto result = vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
		CHECK_VKRESULT(result, "Failed to enumerate device extension properties");

		std::vector<VkExtensionProperties> available_extensions(extension_count);
		result = vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());
		CHECK_VKRESULT(result, "Failed to enumerate device extension properties");

		std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

		for(const auto& extension : available_extensions)
		{
			required_extensions.erase(extension.extensionName);
		}

		return required_extensions.empty();
	}

	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices queue_family_indices;

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

		std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

		int i = 0;
		for(const auto& queue_family : queue_families)
		{
			if(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				queue_family_indices.graphics_family = i;
			}

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

			if(present_support)
			{
				queue_family_indices.present_family = i;
			}

			if(queue_family_indices.IsComplete())
			{
				break;
			}

			i++;
		}

		return queue_family_indices;
	}

	void CreateLogicalDevice()
	{
		QueueFamilyIndices queue_family_indices = FindQueueFamilies(physical_device);

		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<uint32_t> unique_queue_families =
		{
			queue_family_indices.graphics_family.value(),
			queue_family_indices.present_family.value()
		};

		float queue_priority = 1.0f;
		for(auto queue_family : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queue_family;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queue_priority;
			queue_create_infos.emplace_back(std::move(queueCreateInfo));
		}

		VkPhysicalDeviceFeatures device_features{};
		device_features.samplerAnisotropy = VK_TRUE;
		device_features.sampleRateShading = VK_TRUE;

		VkDeviceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		create_info.pQueueCreateInfos = queue_create_infos.data();
		create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());

		create_info.pEnabledFeatures = &device_features;

		create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		create_info.ppEnabledExtensionNames = device_extensions.data();

		if(enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			create_info.enabledLayerCount = 0;
		}

		auto result = vkCreateDevice(physical_device, &create_info, nullptr, &device);
		CHECK_VKRESULT(result, "Failed to create logical device");

		vkGetDeviceQueue(device, queue_family_indices.graphics_family.value(), 0, &graphics_queue);
		vkGetDeviceQueue(device, queue_family_indices.present_family.value(), 0, &present_queue);
	}

	void CreateSwapChain()
	{
		SwapChainSupportDetails swap_chain_support = QuerySwapChainSupport(physical_device);

		VkSurfaceFormatKHR surface_format = ChooseSwapSurfaceFormat(swap_chain_support.formats);
		VkPresentModeKHR present_mode = ChooseSwapPresentMode(swap_chain_support.present_modes);
		VkExtent2D extent = ChooseSwapExtent(swap_chain_support.capabilities);

		uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
		if(swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
		{
			image_count = swap_chain_support.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = surface;

		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices queue_family_indices = FindQueueFamilies(physical_device);
		uint32_t queue_family_indices_array[] = {queue_family_indices.graphics_family.value(), queue_family_indices.present_family.value()};

		if(queue_family_indices.graphics_family != queue_family_indices.present_family)
		{
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices = queue_family_indices_array;
		}
		else
		{
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		create_info.preTransform = swap_chain_support.capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;
		create_info.oldSwapchain = VK_NULL_HANDLE;

		auto result = vkCreateSwapchainKHR(device, &create_info, nullptr, &swap_chain);
		CHECK_VKRESULT(result, "Failed to create swap chain");

		vkGetSwapchainImagesKHR(device, swap_chain, &image_count, nullptr);
		swap_chain_images.resize(image_count);
		vkGetSwapchainImagesKHR(device, swap_chain, &image_count, swap_chain_images.data());

		swap_chain_image_format = surface_format.format;
		swap_chain_extent = extent;
	}

	void CreateImageViews()
	{
		swap_chain_image_views.resize(swap_chain_images.size());

		for(size_t i = 0; i < swap_chain_images.size(); i++)
		{
			swap_chain_image_views[i] = CreateImageView(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription color_attachment{};
		color_attachment.format = swap_chain_image_format;
		color_attachment.samples = msaa_samples;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference color_attachment_ref{};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depth_attachment{};
		depth_attachment.format = FindDepthFormat();
		depth_attachment.samples = msaa_samples;
		depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depth_attachment_ref{};
		depth_attachment_ref.attachment = 1;
		depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription color_attachment_resolve{};
		color_attachment_resolve.format = swap_chain_image_format;
		color_attachment_resolve.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment_resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment_resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment_resolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment_resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment_resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment_resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_resolve_ref{};
		color_attachment_resolve_ref.attachment = 2;
		color_attachment_resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;
		subpass.pResolveAttachments = &color_attachment_resolve_ref;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 3> attachments = {color_attachment, depth_attachment, color_attachment_resolve};
		VkRenderPassCreateInfo render_pass_info{};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
		render_pass_info.pAttachments = attachments.data();
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;

		auto result = vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass);
		CHECK_VKRESULT(result, "Failed to create render pass");
	}

	void CreateDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding ubo_layout_binding{};
		ubo_layout_binding.binding = 0;
		ubo_layout_binding.descriptorCount = 1;
		ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		ubo_layout_binding.pImmutableSamplers = nullptr;
		ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding sampler_layout_binding{};
		sampler_layout_binding.binding = 1;
		sampler_layout_binding.descriptorCount = 1;
		sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		sampler_layout_binding.pImmutableSamplers = nullptr;
		sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ubo_layout_binding, sampler_layout_binding};
		VkDescriptorSetLayoutCreateInfo layout_info{};
		layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
		layout_info.pBindings = bindings.data();

		auto result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout);
		CHECK_VKRESULT(result, "Failed to create descriptor set layout");
	}

	void CreateGraphicsPipeline()
	{
		auto vert_shader_code = ReadFile("shaders/shader.vert.spv");
		auto frag_shader_code = ReadFile("shaders/shader.frag.spv");

		VkShaderModule vert_shader_module = CreateShaderModule(vert_shader_code);
		VkShaderModule frag_shader_module = CreateShaderModule(frag_shader_code);

		VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
		vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_stage_info.module = vert_shader_module;
		vert_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
		frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_stage_info.module = frag_shader_module;
		frag_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

		auto binding_description = Vertex::GetBindingDescription();
		auto attribute_descriptions = Vertex::GetAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertex_input_info{};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 1;
		vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
		vertex_input_info.pVertexBindingDescriptions = &binding_description;
		vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

		VkPipelineInputAssemblyStateCreateInfo input_assembly{};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swap_chain_extent.width);
		viewport.height = static_cast<float>(swap_chain_extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swap_chain_extent;

		VkPipelineViewportStateCreateInfo viewport_state{};
		viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_TRUE;
		multisampling.minSampleShading = 0.2f; // min fraction for sample shading; closer to 1 is smoother
		multisampling.rasterizationSamples = msaa_samples;

		VkPipelineDepthStencilStateCreateInfo depth_stencil{};
		depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil.depthTestEnable = VK_TRUE;
		depth_stencil.depthWriteEnable = VK_TRUE;
		depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil.depthBoundsTestEnable = VK_FALSE;
		depth_stencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState color_blend_attachment{};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
												VK_COLOR_COMPONENT_G_BIT |
												VK_COLOR_COMPONENT_B_BIT |
												VK_COLOR_COMPONENT_A_BIT;
		color_blend_attachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo color_blending{};
		color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blending.logicOpEnable = VK_FALSE;
		color_blending.logicOp = VK_LOGIC_OP_COPY;
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &color_blend_attachment;
		color_blending.blendConstants[0] = 0.0f;
		color_blending.blendConstants[1] = 0.0f;
		color_blending.blendConstants[2] = 0.0f;
		color_blending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipeline_layout_info{};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 1;
		pipeline_layout_info.pSetLayouts = &descriptor_set_layout;

		auto result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout);
		CHECK_VKRESULT(result, "Failed to create pipeline layout");

		VkGraphicsPipelineCreateInfo pipeline_info{};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pDepthStencilState = &depth_stencil;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.layout = pipeline_layout;
		pipeline_info.renderPass = render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);
		CHECK_VKRESULT(result, "Failed to create graphics pipeline");

		vkDestroyShaderModule(device, frag_shader_module, nullptr);
		vkDestroyShaderModule(device, vert_shader_module, nullptr);
	}

	void CreateFramebuffers()
	{
		swap_chain_framebuffers.resize(swap_chain_image_views.size());

		for(size_t i = 0; i < swap_chain_image_views.size(); i++)
		{
			std::array<VkImageView, 3> attachments =
			{
				color_image_view,
				depth_image_view,
				swap_chain_image_views[i]
			};

			VkFramebufferCreateInfo framebuffer_info{};
			framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer_info.renderPass = render_pass;
			framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebuffer_info.pAttachments = attachments.data();
			framebuffer_info.width = swap_chain_extent.width;
			framebuffer_info.height = swap_chain_extent.height;
			framebuffer_info.layers = 1;

			auto result = vkCreateFramebuffer(device, &framebuffer_info, nullptr, &swap_chain_framebuffers[i]);
			CHECK_VKRESULT(result, "Failed to create framebuffer");
		}
	}

	void CreateCommandPool()
	{
		QueueFamilyIndices queue_family_indices = FindQueueFamilies(physical_device);

		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();

		auto result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
		CHECK_VKRESULT(result, "Failed to create command pool");
	}

	void CreateColorResources()
	{
		VkFormat color_format = swap_chain_image_format;

		CreateImage(swap_chain_extent.width, swap_chain_extent.height, 1, msaa_samples, color_format, VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, color_image, color_image_memory);

		color_image_view = CreateImageView(color_image, color_format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}

	void CreateDepthResources()
	{
		VkFormat depth_format = FindDepthFormat();

		CreateImage(swap_chain_extent.width, swap_chain_extent.height, 1, msaa_samples, depth_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image, depth_image_memory);
		depth_image_view = CreateImageView(depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	}

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for(VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

			if((tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) ||
			   (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features))
			{
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	VkFormat FindDepthFormat()
	{
		return FindSupportedFormat(
					{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
					VK_IMAGE_TILING_OPTIMAL,
					VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool HasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void CreateTextureImage()
	{
		int tex_width, tex_height, tex_channels;

		stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
		if(!pixels)
		{
			throw std::runtime_error("Failed to load texture image \"" + TEXTURE_PATH + "\"");
		}

		VkDeviceSize image_size = tex_width * tex_height * 4;
		mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(tex_width, tex_height)))) + 1;

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;
		CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void *data;
		vkMapMemory(device, staging_buffer_memory, 0, image_size, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(image_size));
		vkUnmapMemory(device, staging_buffer_memory);

		stbi_image_free(pixels);

		CreateImage(tex_width, tex_height, mip_levels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image, texture_image_memory);

		TransitionImageLayout(texture_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mip_levels);
		CopyBufferToImage(staging_buffer, texture_image, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height));
		//transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

		vkDestroyBuffer(device, staging_buffer, nullptr);
		vkFreeMemory(device, staging_buffer_memory, nullptr);

		GenerateMipmaps(texture_image, VK_FORMAT_R8G8B8A8_SRGB, tex_width, tex_height, mip_levels);
	}

	void GenerateMipmaps(VkImage image, VkFormat image_format, int32_t tex_width, int32_t tex_height, uint32_t mip_levels)
	{
		// Check if image format supports linear blitting
		VkFormatProperties format_properties;
		vkGetPhysicalDeviceFormatProperties(physical_device, image_format, &format_properties);

		if(!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		{
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer command_buffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mip_width = tex_width;
		int32_t mip_height = tex_height;

		for(uint32_t i = 1; i < mip_levels; i++)
		{
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(command_buffer,
								 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
								 0, nullptr,
								 0, nullptr,
								 1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = {0, 0, 0};
			blit.srcOffsets[1] = {mip_width, mip_height, 1};
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = {0, 0, 0};
			blit.dstOffsets[1] = { mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(command_buffer,
						   image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						   image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						   1, &blit,
						   VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(command_buffer,
								 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
								 0, nullptr,
								 0, nullptr,
								 1, &barrier);

			if(mip_width > 1) mip_width /= 2;
			if(mip_height > 1) mip_height /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mip_levels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(command_buffer,
							 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
							 0, nullptr,
							 0, nullptr,
							 1, &barrier);

		EndSingleTimeCommands(command_buffer);
	}

	VkSampleCountFlagBits GetMaxUsableSampleCount()
	{
		VkPhysicalDeviceProperties physical_device_properties;
		vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

		VkSampleCountFlags counts = physical_device_properties.limits.framebufferColorSampleCounts &
									physical_device_properties.limits.framebufferDepthSampleCounts;
		if(counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if(counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if(counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if(counts & VK_SAMPLE_COUNT_8_BIT)  { return VK_SAMPLE_COUNT_8_BIT; }
		if(counts & VK_SAMPLE_COUNT_4_BIT)  { return VK_SAMPLE_COUNT_4_BIT; }
		if(counts & VK_SAMPLE_COUNT_2_BIT)  { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	void CreateTextureImageView()
	{
		texture_image_view = CreateImageView(texture_image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mip_levels);
	}

	void CreateTextureSampler()
	{
		VkSamplerCreateInfo sampler_info{};
		sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		sampler_info.magFilter = VK_FILTER_LINEAR;
		sampler_info.minFilter = VK_FILTER_LINEAR;
		sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.anisotropyEnable = VK_TRUE;
		sampler_info.maxAnisotropy = 16.0f;
		sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		sampler_info.unnormalizedCoordinates = VK_FALSE;
		sampler_info.compareEnable = VK_FALSE;
		sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
		sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler_info.minLod = 0.0f;
		sampler_info.maxLod = static_cast<float>(mip_levels);
		sampler_info.mipLodBias = 0.0f;
		auto result = vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler);
		CHECK_VKRESULT(result, "Failed to create texture sampler");
	}

	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags, int32_t mip_levels)
	{
		VkImageViewCreateInfo view_info{};
		view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_info.image = image;
		view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_info.format = format;
		view_info.subresourceRange.aspectMask = aspect_flags;
		view_info.subresourceRange.baseMipLevel = 0;
		view_info.subresourceRange.levelCount = mip_levels;
		view_info.subresourceRange.baseArrayLayer = 0;
		view_info.subresourceRange.layerCount = 1;

		VkImageView image_view;
		auto result = vkCreateImageView(device, &view_info, nullptr, &image_view);
		CHECK_VKRESULT(result, "Failed to create image views");

		return image_view;
	}

	void CreateImage(uint32_t width, uint32_t height, uint32_t mip_levels, VkSampleCountFlagBits num_samples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
					 VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory)
	{
		VkImageCreateInfo image_info{};
		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.extent.width = width;
		image_info.extent.height = height;
		image_info.extent.depth = 1;
		image_info.mipLevels = mip_levels;
		image_info.arrayLayers = 1;
		image_info.format = format;
		image_info.tiling = tiling;
		image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		image_info.usage = usage;
		image_info.samples = num_samples;
		image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		auto result = vkCreateImage(device, &image_info, nullptr, &image);
		CHECK_VKRESULT(result, "Failed to create image");

		VkMemoryRequirements mem_requirements;
		vkGetImageMemoryRequirements(device, image, &mem_requirements);

		VkMemoryAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, properties);

		result = vkAllocateMemory(device, &alloc_info, nullptr, &image_memory);
		CHECK_VKRESULT(result, "Failed to allocate image memory");

		vkBindImageMemory(device, image, image_memory, 0);
	}

	void TransitionImageLayout(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, uint32_t mip_levels)
	{
		VkCommandBuffer command_buffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = old_layout;
		barrier.newLayout = new_layout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mip_levels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags source_stage;
		VkPipelineStageFlags destination_stage;

		if(old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if(old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
				new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		EndSingleTimeCommands(command_buffer);
	}

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer command_buffer = BeginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = {0, 0, 0};
		region.imageExtent = {width, height, 1};

		vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		EndSingleTimeCommands(command_buffer);
	}

	void LoadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if(!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}

		std::unordered_map<Vertex, uint32_t> unique_vertices{};

		for(const auto& shape : shapes)
		{
			for(const auto& index : shape.mesh.indices)
			{
				Vertex vertex{};

				vertex.position =
				{
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texture_coordinates =
				{
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vertex.color = {1.0f, 1.0f, 1.0f};
				if(!unique_vertices.contains(vertex))
				{
					unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				indices.push_back(unique_vertices[vertex]);
			}
		}
	}


	template <typename T>
	void CreateVkBuffer(const std::vector<T>& input, VkBuffer& buffer,
						VkDeviceMemory& buffer_memory, VkBufferUsageFlags usage_flags)
	{
		VkDeviceSize buffer_size = sizeof(input[0]) * input.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;
		CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					 staging_buffer, staging_buffer_memory);

		void *data;
		vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, input.data(), static_cast<size_t>(buffer_size));
		vkUnmapMemory(device, staging_buffer_memory);

		CreateBuffer(buffer_size, usage_flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, buffer_memory);

		CopyBuffer(staging_buffer, buffer, buffer_size);

		vkDestroyBuffer(device, staging_buffer, nullptr);
		vkFreeMemory(device, staging_buffer_memory, nullptr);
	}

	void CreateVertexBuffer()
	{
		CreateVkBuffer(vertices, vertex_buffer, vertex_buffer_memory, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	void CreateIndexBuffer()
	{
		CreateVkBuffer(indices, index_buffer, index_buffer_memory, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	}

	void CreateUniformBuffers()
	{
		VkDeviceSize buffer_size = sizeof(UniformBufferObject);

		uniform_buffers.resize(swap_chain_images.size());
		uniform_buffers_memory.resize(swap_chain_images.size());

		for(size_t i = 0; i < swap_chain_images.size(); i++)
		{
			CreateBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						 uniform_buffers[i], uniform_buffers_memory[i]);
		}
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
					  VkBuffer& buffer, VkDeviceMemory& buffer_memory)
	{
		VkBufferCreateInfo buffer_info{};
		buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_info.size = size;
		buffer_info.usage = usage;
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		auto result = vkCreateBuffer(device, &buffer_info, nullptr, &buffer);
		CHECK_VKRESULT(result, "Failed to create vertex buffer");

		VkMemoryRequirements mempry_requirements;
		vkGetBufferMemoryRequirements(device, buffer, &mempry_requirements);

		VkMemoryAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mempry_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(mempry_requirements.memoryTypeBits, properties);

		result = vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory);
		CHECK_VKRESULT(result, "Failed to allocate vertex buffer memory");

		vkBindBufferMemory(device, buffer, buffer_memory, 0);
	}

	VkCommandBuffer BeginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandPool = command_pool;
		alloc_info.commandBufferCount = 1;

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(command_buffer, &begin_info);

		return command_buffer;
	}

	void EndSingleTimeCommands(VkCommandBuffer command_buffer)
	{
		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);

		vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
	}

	void CopyBuffer(VkBuffer source, VkBuffer destination, VkDeviceSize size)
	{
		VkCommandBuffer command_buffer = BeginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(command_buffer, source, destination, 1, &copyRegion);

		EndSingleTimeCommands(command_buffer);
	}

	void CreateDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> pool_sizes{};
		pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		pool_sizes[0].descriptorCount = static_cast<uint32_t>(swap_chain_images.size());
		pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		pool_sizes[1].descriptorCount = static_cast<uint32_t>(swap_chain_images.size());

		VkDescriptorPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
		pool_info.pPoolSizes = pool_sizes.data();
		pool_info.maxSets = static_cast<uint32_t>(swap_chain_images.size());

		auto result = vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool);
		CHECK_VKRESULT(result, "Failed to create descriptor pool");
	}

	void CreateDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(swap_chain_images.size(), descriptor_set_layout);
		VkDescriptorSetAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = descriptor_pool;
		alloc_info.descriptorSetCount = static_cast<uint32_t>(swap_chain_images.size());
		alloc_info.pSetLayouts = layouts.data();

		descriptor_sets.resize(swap_chain_images.size());
		auto result = vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data());
		CHECK_VKRESULT(result, "Failed to allocate descriptor sets");

		for(size_t i = 0; i < swap_chain_images.size(); i++)
		{
			VkDescriptorBufferInfo buffer_info{};
			buffer_info.buffer = uniform_buffers[i];
			buffer_info.offset = 0;
			buffer_info.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo image_info{};
			image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			image_info.imageView = texture_image_view;
			image_info.sampler = texture_sampler;

			std::array<VkWriteDescriptorSet, 2> descriptor_writes{};

			descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptor_writes[0].dstSet = descriptor_sets[i];
			descriptor_writes[0].dstBinding = 0;
			descriptor_writes[0].dstArrayElement = 0;
			descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptor_writes[0].descriptorCount = 1;
			descriptor_writes[0].pBufferInfo = &buffer_info;

			descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptor_writes[1].dstSet = descriptor_sets[i];
			descriptor_writes[1].dstBinding = 1;
			descriptor_writes[1].dstArrayElement = 0;
			descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptor_writes[1].descriptorCount = 1;
			descriptor_writes[1].pImageInfo = &image_info;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
		}
	}

	uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

		for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
		{
			if((type_filter & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void CreateCommandBuffers()
	{
		command_buffers.resize(swap_chain_framebuffers.size());

		VkCommandBufferAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = command_pool;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

		auto result = vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data());
		CHECK_VKRESULT(result, "Failed to allocate command buffers");

		for(size_t i = 0; i < command_buffers.size(); i++)
		{
			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			result = vkBeginCommandBuffer(command_buffers[i], &begin_info);
			CHECK_VKRESULT(result, "Failed to begin recording command buffer");

			std::array<VkClearValue, 2> clear_values{};
			clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
			clear_values[1].depthStencil = {1.0f, 0};

			VkRenderPassBeginInfo render_pass_info{};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			render_pass_info.renderPass = render_pass;
			render_pass_info.framebuffer = swap_chain_framebuffers[i];
			render_pass_info.renderArea.offset = {0, 0};
			render_pass_info.renderArea.extent = swap_chain_extent;
			render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
			render_pass_info.pClearValues = clear_values.data();

			vkCmdBeginRenderPass(command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

			VkBuffer vertex_buffers[] = {vertex_buffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertex_buffers, offsets);
			vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[i], 0, nullptr);
			vkCmdDrawIndexed(command_buffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

			vkCmdEndRenderPass(command_buffers[i]);

			result = vkEndCommandBuffer(command_buffers[i]);
			CHECK_VKRESULT(result, "Failed to record command buffer");
		}
	}

	void CreateSyncObjects()
	{
		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
		images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphore_info{};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info{};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			auto result = vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_semaphores[i]);
			CHECK_VKRESULT(result, "Failed to create semaphore");

			result = vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphores[i]);
			CHECK_VKRESULT(result, "Failed to create semaphore");

			result = vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]);
			CHECK_VKRESULT(result, "Failed to create fence");
		}
	}

	void UpdateUniformBuffer(uint32_t current_image)
	{
		static auto start_time = std::chrono::high_resolution_clock::now();

		auto current_time = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.projection = glm::perspective(glm::radians(45.0f), swap_chain_extent.width / static_cast<float>(swap_chain_extent.height), 0.1f, 10.0f);
		ubo.projection[1][1] *= -1;

		void *data;
		vkMapMemory(device, uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniform_buffers_memory[current_image]);
	}

	void DrawFrame()
	{
		vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		auto result = vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);

		if(result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			RecreateSwapChain();
			return;
		}
		else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("Failed to acquire swap chain image (VkResult: " + std::to_string(result) + ")");
		}

		UpdateUniformBuffer(image_index);

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if(images_in_flight[image_index] != VK_NULL_HANDLE)
		{
			vkWaitForFences(device, 1, &images_in_flight[image_index], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		images_in_flight[image_index] = in_flight_fences[current_frame];

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore wait_semaphores[] = {image_available_semaphores[current_frame]};
		VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;

		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[image_index];

		VkSemaphore signal_semaphores[] = {render_finished_semaphores[current_frame]};
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		vkResetFences(device, 1, &in_flight_fences[current_frame]);

		result = vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]);
		CHECK_VKRESULT(result, "Failed to submit draw command buffer");

		VkPresentInfoKHR present_info{};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		VkSwapchainKHR swapChains[] = {swap_chain};
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapChains;

		present_info.pImageIndices = &image_index;

		result = vkQueuePresentKHR(present_queue, &present_info);

		if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
		{
			framebuffer_resized = false;
			RecreateSwapChain();
		}
		else if(result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present swap chain image (VkResult: " + std::to_string(result) + ")");
		}

		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shader_module;
		auto result = vkCreateShaderModule(device, &create_info, nullptr, &shader_module);
		CHECK_VKRESULT(result, "Failed to create shader module");

		return shader_module;
	}

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats)
	{
		auto it = std::find_if(available_formats.begin(), available_formats.end(), [](const auto& available_format)
		{
			return available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		});

		return it != available_formats.end() ? *it : available_formats[0];
	}

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available_present_modes)
	{
		auto it = std::find_if(available_present_modes.begin(), available_present_modes.end(), [](const auto& available_present_mode)
		{
			return available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR;
		});

		return it != available_present_modes.end() ? *it : VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if(capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actual_extent =
			{
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actual_extent.width = std::max(capabilities.minImageExtent.width,
										   std::min(capabilities.maxImageExtent.width, actual_extent.width));
			actual_extent.height = std::max(capabilities.minImageExtent.height,
										   std::min(capabilities.maxImageExtent.height, actual_extent.height));

			return actual_extent;
		}
	}

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

		if(format_count != 0)
		{
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);

		if(present_mode_count != 0)
		{
			details.present_modes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
		}

		return details;
	}

	std::vector<const char*> GetRequiredExtensions()
	{
		uint32_t glfw_extension_count = 0;
		const char **glfw_extensions;
		glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

		if(enable_validation_layers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	std::vector<VkExtensionProperties> GetAvailableExtensions()
	{
		uint32_t extension_count = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
		std::vector<VkExtensionProperties> extensions(extension_count);
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

		return extensions;
	}

	bool CheckValidationLayerSupport()
	{
		uint32_t layer_count;
		vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

		std::vector<VkLayerProperties> availableLayers(layer_count);
		vkEnumerateInstanceLayerProperties(&layer_count, availableLayers.data());

		return std::all_of(validation_layers.begin(), validation_layers.end(), [&availableLayers](auto name){
			return std::find_if(availableLayers.begin(), availableLayers.end(), [&name](auto lhs){
				return std::strcmp(name, lhs.layerName) == 0;
			}) != availableLayers.end();
		});
	}

	static std::vector<char> ReadFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if(!file.is_open())
		{
			throw std::runtime_error("Failed to open file \"" + filename + "\"");
		}

		auto fileSize = static_cast<size_t>(file.tellg());
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
														VkDebugUtilsMessageTypeFlagsEXT message_type,
														const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
														void *user_data)
	{
		#pragma unused(message_severity)
		#pragma unused(message_type)
		#pragma unused(user_data)

		std::cerr << "validation layer: " << callback_data->pMessage << std::endl;

		return VK_FALSE;
	}
};

int main()
{
	Application app;

	try
	{
		app.Run();
	}
	catch(const std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
