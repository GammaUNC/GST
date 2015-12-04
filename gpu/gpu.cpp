#include "gpu.h"

#include <cassert>
#include <iostream>

#ifdef __APPLE__
#  include <OpenGL/opengl.h>
#elif defined (_WIN32)
#  define NOMINMAX
#  include "Windows.h"
#else
#  include <GL/glx.h>
#endif

#include "kernel_cache.h"

namespace gpu {

static void ContextErrorCallback(const char *errinfo, const void *, size_t, void *) {
  fprintf(stderr, "Context error: %s", errinfo);
  assert(false);
  exit(1);
}

static void PrintDeviceInfo(cl_device_id device_id) {

  size_t strLen;
  const size_t kStrBufSz = 1024;
  union {
    char strBuf[kStrBufSz];
    cl_uint intBuf[kStrBufSz / sizeof(cl_uint)];
    size_t sizeBuf[kStrBufSz / sizeof(size_t)];
  };

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_NAME, kStrBufSz, strBuf, &strLen);
  std::cout << "Device name: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_PROFILE, kStrBufSz, strBuf, &strLen);
  std::cout << "Device profile: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_VENDOR, kStrBufSz, strBuf, &strLen);
  std::cout << "Device vendor: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_VERSION, kStrBufSz, strBuf, &strLen);
  std::cout << "Device version: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DRIVER_VERSION, kStrBufSz, strBuf, &strLen);
  std::cout << "Device driver version: " << strBuf << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_ADDRESS_BITS, kStrBufSz, strBuf,
           &strLen);
  std::cout << "Device driver address bits: " << intBuf[0] << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, kStrBufSz,
           strBuf, &strLen);
  std::cout << "Max work group size: " << sizeBuf[0] << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, kStrBufSz,
           strBuf, &strLen);
  std::cout << "Max work item dimensions: " << intBuf[0] << std::endl;
  assert(*(reinterpret_cast<cl_uint *>(strBuf)) >= 2);

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_GLOBAL_MEM_SIZE, kStrBufSz,
           strBuf, &strLen);
  std::cout << "Total global bytes available: " << intBuf[0] << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_LOCAL_MEM_SIZE, kStrBufSz,
           strBuf, &strLen);
  std::cout << "Total local bytes available: " << intBuf[0] << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, kStrBufSz,
           strBuf, &strLen);
  std::cout << "Total size of memory allocatable: " << intBuf[0] << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, kStrBufSz, strBuf, &strLen);
  size_t nSizeElements = strLen / sizeof(size_t);
  std::cout << "Max work item sizes: (";
  size_t *dimSizes = (size_t *)(strBuf);
  for(size_t j = 0; j < nSizeElements; j++) {
    std::cout << dimSizes[j] << ((j == nSizeElements - 1)? "" : ", ");
  }
  std::cout << ")" << std::endl;

  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_EXTENSIONS, kStrBufSz, strBuf, &strLen);
  std::cout << "Device extensions: " << std::endl;
  for (size_t k = 0; k < strLen; ++k) {
    if (strBuf[k] == ',' || strBuf[k] == ' ') {
      strBuf[k] = '\0';
    }
  }

  std::cout << "  " << strBuf << std::endl;
  for (size_t k = 1; k < strLen; ++k) {
    if (strBuf[k] == '\0' && k < strLen - 1) {
      std::cout << "  " << (strBuf + k + 1) << std::endl;
    }
  }

  cl_device_type deviceType;
  size_t deviceTypeSz;
  CHECK_CL(clGetDeviceInfo, device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, &deviceTypeSz);
  if(deviceType & CL_DEVICE_TYPE_CPU) {
    std::cout << "Device driver type: CPU" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_GPU) {
    std::cout << "Device driver type: GPU" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
    std::cout << "Device driver type: ACCELERATOR" << std::endl;
  }
  if(deviceType & CL_DEVICE_TYPE_DEFAULT) {
    std::cout << "Device driver type: DEFAULT" << std::endl;
  }
}

static std::vector<cl_context_properties> GetSharedCLGLProps() {
  std::vector<cl_context_properties> props;
#ifdef __APPLE__
  // Get current CGL Context and CGL Share group
  CGLContextObj kCGLContext = CGLGetCurrentContext();
  CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

  props.push_back(CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
  props.push_back((cl_context_properties)kCGLShareGroup);
  props.push_back(0);

#elif defined (_WIN32)

  // OpenGL context
  props.push_back(CL_GL_CONTEXT_KHR);
  props.push_back((cl_context_properties)wglGetCurrentContext());

  // HDC used to create the OpenGL context
  props.push_back(CL_WGL_HDC_KHR);
  props.push_back((cl_context_properties)wglGetCurrentDC());

  props.push_back(0);

#else  // Linux??

  props.push_back(CL_GL_CONTEXT_KHR);
  props.push_back((cl_context_properties) glXGetCurrentContext());
  props.push_back(CL_GLX_DISPLAY_KHR);
  props.push_back((cl_context_properties) glXGetCurrentDisplay());
  props.push_back(0);

#endif

  return std::move(props);
}

static void CreateCLContext(cl_context *result, const cl_context_properties *props,
                            cl_device_id device) {
  cl_int errCreateContext;
  *result = clCreateContext(props, 1, &device, ContextErrorCallback, NULL,
                            &errCreateContext);
  CHECK_CL((cl_int), errCreateContext);
}

static cl_device_id GetDeviceForSharedContext(cl_context ctx) {
  size_t device_id_size_bytes;
  cl_device_id device;

#ifndef __APPLE__
  std::vector<cl_context_properties> props = GetSharedCLGLProps();

  typedef CL_API_ENTRY cl_int(CL_API_CALL *CtxInfoFunc)
    (const cl_context_properties *properties, cl_gl_context_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);
  static CtxInfoFunc getCtxInfoFunc = NULL;
#ifndef CL_VERSION_1_1
  getCtxInfoFunc = reinterpret_cast<CtxInfoFunc>(
    clGetExtensionFunctionAddress("clGetGLContextInfoKHR"));
#else
  cl_platform_id platform;
  CHECK_CL(clGetPlatformIDs, 1, &platform, NULL);
  std::vector<cl_context_properties> extra_props = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform };
  extra_props.insert(extra_props.end(), props.begin(), props.end());
  props = extra_props;

  getCtxInfoFunc = reinterpret_cast<CtxInfoFunc>(
    clGetExtensionFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR"));
#endif  // CL_VERSION_1_1

  assert(getCtxInfoFunc);
  CHECK_CL(getCtxInfoFunc, props.data(), CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
    sizeof(device), &device, &device_id_size_bytes);
#else
  // Get current CGL Context and CGL Share group
  CGLContextObj kCGLContext = CGLGetCurrentContext();
  // And now we can ask OpenCL which particular device is being used by
  // OpenGL to do the rendering, currently:
  clGetGLContextInfoAPPLE(ctx, kCGLContext,
    CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE,
    sizeof(device), &device, &device_id_size_bytes);
#endif  // __APPLE
  // If we're sharing an openGL context, there should really only
  // be one device ID...
  assert(device_id_size_bytes == sizeof(cl_device_id));
  return device;
}

static std::vector<cl_device_id> GetAllDevicesForContext(cl_context ctx) {
  std::vector<cl_device_id> devices(16);
  size_t nDeviceIds;
  CHECK_CL(clGetContextInfo, ctx, CL_CONTEXT_DEVICES, devices.size() * sizeof(cl_device_id),
    devices.data(), &nDeviceIds);
  nDeviceIds /= sizeof(cl_device_id);
  devices.resize(nDeviceIds);
  return std::move(devices);
}


GPUContext::~GPUContext() {
  GPUKernelCache::Instance(_ctx, _device)->Clear();
  CHECK_CL(clReleaseCommandQueue, _command_queue);
  CHECK_CL(clReleaseContext, _ctx);
}

std::unique_ptr<GPUContext> GPUContext::InitializeOpenCL(bool share_opengl) {
  const cl_uint kMaxPlatforms = 8;
  cl_platform_id platforms[kMaxPlatforms];
  cl_uint nPlatforms;
  CHECK_CL(clGetPlatformIDs, kMaxPlatforms, platforms, &nPlatforms);

#ifndef NDEBUG
  size_t strLen;
  fprintf(stdout, "\n");
  fprintf(stdout, "Found %d OpenCL platform%s.\n", nPlatforms, nPlatforms == 1 ? "" : "s");

  for (cl_uint i = 0; i < nPlatforms; i++) {
    char strBuf[256];

    fprintf(stdout, "\n");
    fprintf(stdout, "Platform %d info:\n", i);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_PROFILE, 256, strBuf, &strLen);
    fprintf(stdout, "Platform profile: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VERSION, 256, strBuf, &strLen);
    fprintf(stdout, "Platform version: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_NAME, 256, strBuf, &strLen);
    fprintf(stdout, "Platform name: %s\n", strBuf);

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VENDOR, 256, strBuf, &strLen);
    fprintf(stdout, "Platform vendor: %s\n", strBuf);
  }
#endif

  cl_platform_id platform = platforms[0];

  const cl_uint kMaxDevices = 8;
  cl_device_id devices[kMaxDevices];
  cl_uint nDevices;
  CHECK_CL(clGetDeviceIDs, platform, CL_DEVICE_TYPE_GPU, kMaxDevices, devices, &nDevices);

#ifndef NDEBUG
  fprintf(stdout, "\n");
  fprintf(stdout, "Found %d device%s on platform 0.\n", nDevices, nDevices == 1 ? "" : "s");

  for (cl_uint i = 0; i < nDevices; i++) {
    gpu::PrintDeviceInfo(devices[i]);
  }

  std::cout << std::endl;
#endif

  // Create OpenCL context...
  cl_context ctx;
  std::vector<cl_context_properties> props = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform
  };

  if (share_opengl) {
    std::vector<cl_context_properties> clgl_props = std::move(GetSharedCLGLProps());
    props.insert(props.end(), clgl_props.begin(), clgl_props.end());
  }
  else {
    props.push_back(0);
  }

  CreateCLContext(&ctx, props.data(), devices[0]);

  // We got the context...
  std::unique_ptr<GPUContext> gpu_ctx =
    std::unique_ptr<GPUContext>(new GPUContext);
  gpu_ctx->_ctx = ctx;

  // The device...
  if (share_opengl) {
    gpu_ctx->_device = GetDeviceForSharedContext(ctx);
  } else {
    std::vector<cl_device_id> devices = std::move(GetAllDevicesForContext(ctx));
    assert(devices.size() > 0);
    gpu_ctx->_device = devices[0];
  }

  // And the command queue...
  cl_int errCreateCommandQueue;
#ifndef CL_VERSION_2_0
  gpu_ctx->_command_queue = clCreateCommandQueue(ctx, gpu_ctx->_device, 0, &errCreateCommandQueue);
#else
  gpu_ctx->_command_queue = clCreateCommandQueueWithProperties(ctx, gpu_ctx->_device, 0, &errCreateCommandQueue);
#endif
  CHECK_CL((cl_int), errCreateCommandQueue);

  return std::move(gpu_ctx);
}

void GPUContext::PrintDeviceInfo() const {
  gpu::PrintDeviceInfo(_device);
}

cl_kernel GPUContext::GetOpenCLKernel(const std::string &filename, const std::string &kernel) const {
  GPUKernelCache *cache = GPUKernelCache::Instance(_ctx, _device);
  return cache->GetKernel(filename, kernel);
}

}  // namespace gpu
