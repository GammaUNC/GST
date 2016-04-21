#include "gpu.h"

#include <cassert>
#include <cstring>
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
  fprintf(stderr, "Context error: %s\n", errinfo);
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

static cl_platform_id GetCLPlatform() {
  const cl_uint kMaxPlatforms = 8;
  cl_platform_id platforms[kMaxPlatforms];
  cl_uint nPlatforms;
  static int platform_idx = -1;

  CHECK_CL(clGetPlatformIDs, kMaxPlatforms, platforms, &nPlatforms);
#ifdef NDEBUG
  platform_idx = 0;
#else
  if (platform_idx < 0) {
    std::cout << "OpenCL has " << nPlatforms << " platform"
      << ((nPlatforms != 1) ? "s" : "")
      << " available. Querying... " << std::endl;
  }
  else {
    return platforms[platform_idx];
  }

  platform_idx = 0;

  size_t strLen;
  static const size_t kStrBufSz = 1024;
  for (cl_uint i = 0; i < nPlatforms; i++) {
    char strBuf[kStrBufSz];

    std::cout << std::endl;
    std::cout << "Platform " << i << " info:" << std::endl;

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_PROFILE, kStrBufSz, strBuf, &strLen);
    std::cout << "Platform profile: " << strBuf << std::endl;

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VERSION, kStrBufSz, strBuf, &strLen);
    std::cout << "Platform version: " << strBuf << std::endl;

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_NAME, kStrBufSz, strBuf, &strLen);
    std::cout << "Platform name: " << strBuf << std::endl;

    bool isCPUOnly = false;
    if (strstr(strBuf, "CPU Only")) {
      isCPUOnly = true;
    }

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_VENDOR, kStrBufSz, strBuf, &strLen);
    std::cout << "Platform vendor: " << strBuf << std::endl;

    bool isIntel = false;
    if (strstr(strBuf, "Intel")) {
      isIntel = true;
    }

    CHECK_CL(clGetPlatformInfo, platforms[i], CL_PLATFORM_EXTENSIONS, kStrBufSz, strBuf, &strLen);
    std::cout << "Platform extensions: " << std::endl;
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

    if (isCPUOnly && isIntel) {
      platform_idx = i;
    }
  }
  std::cout << "Using platform " << platform_idx << std::endl;
#endif

  return platforms[platform_idx];
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
  cl_platform_id platform = GetCLPlatform();
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
  GPUKernelCache::Instance(_ctx, _type, _device)->Clear();
  CHECK_CL(clReleaseCommandQueue, _command_queue);
  CHECK_CL(clReleaseContext, _ctx);
}

std::unique_ptr<GPUContext> GPUContext::InitializeOpenCL(bool share_opengl) {
  const cl_uint kMaxDevices = 8;
  cl_device_id devices[kMaxDevices];
  cl_uint nDevices;

  cl_platform_id platform = GetCLPlatform();
  CHECK_CL(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL, kMaxDevices, devices, &nDevices);

  cl_device_type device_type;
  CHECK_CL(clGetDeviceInfo, devices[0], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

  if (device_type == CL_DEVICE_TYPE_CPU) {
    std::cout << "========================================";
    std::cout << "========================================" << std::endl;
    std::cout << "========================================";
    std::cout << "========================================" << std::endl;
    std::cout << "WARNING: Running on the CPU" << std::endl;
    std::cout << "========================================";
    std::cout << "========================================" << std::endl;
    std::cout << "========================================";
    std::cout << "========================================" << std::endl;
  }

#ifndef NDEBUG
  std::cout << std::endl;
  std::cout << "Found " << nDevices << " device" << (nDevices == 1 ? "" : "s") << std::endl;

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

  if (device_type == CL_DEVICE_TYPE_CPU) {
    gpu_ctx->_type = eContextType_IntelCPU;
  } else {
    gpu_ctx->_type = eContextType_GenericGPU;
  }

  // The device...
  if (share_opengl) {
    gpu_ctx->_device = GetDeviceForSharedContext(ctx);
    assert(gpu_ctx->_device == devices[0]);
  } else {
    std::vector<cl_device_id> ds = std::move(GetAllDevicesForContext(ctx));
    assert(ds.size() > 0);
    assert(ds[0] == devices[0]);
    gpu_ctx->_device = ds[0];
  }

  // And the command queue...
  cl_int errCreateCommandQueue;
  cl_command_queue_properties cq_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#ifndef CL_VERSION_2_0
  gpu_ctx->_command_queue = clCreateCommandQueue(ctx, gpu_ctx->_device, cq_props, &errCreateCommandQueue);
#else
  cl_queue_properties cq_props_list[] = {
    CL_QUEUE_PROPERTIES,
    cq_props,
    0
  };

  gpu_ctx->_command_queue = clCreateCommandQueueWithProperties(ctx, gpu_ctx->_device, cq_props_list, &errCreateCommandQueue);
#endif
  CHECK_CL((cl_int), errCreateCommandQueue);

  return std::move(gpu_ctx);
}

void GPUContext::PrintDeviceInfo() const {
  gpu::PrintDeviceInfo(_device);
}

cl_kernel GPUContext::GetOpenCLKernel(const std::string &filename, const std::string &kernel) const {
  GPUKernelCache *cache = GPUKernelCache::Instance(_ctx, _type, _device);
  return cache->GetKernel(filename, kernel);
}

}  // namespace gpu
