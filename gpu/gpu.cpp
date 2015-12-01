#include "gpu.h"

#include <cassert>
#include <iostream>
#include <fstream>

#ifdef __APPLE__
#  include <OpenGL/opengl.h>
#elif defined (_WIN32)
#  define NOMINMAX
#  include "Windows.h"
#  include <CL/cl_gl.h>
#else
#  include <CL/cl_gl.h>
#endif


static const size_t kBlockWidth = 4;
static const size_t kBlockHeight = 4;
static const size_t kNumComponents = 4;
static const size_t kNumBlockPixels = kBlockWidth * kBlockHeight;
static const size_t kBlockSize = kNumBlockPixels * kNumComponents;

static size_t kLocalWorkSizeX = 16;
static size_t kLocalWorkSizeY = 16;
static inline size_t GetTotalWorkItems() { return kLocalWorkSizeX * kLocalWorkSizeY; }

// We have 16 pixels per work item and 4 bytes per pixel, so each work
// group will need this much local memory.
static inline size_t GetPixelBufferBytes() { return GetTotalWorkItems() * kBlockSize; }

// Thirty-two bytes for the kernel arguments...
static inline size_t GetTotalLocalMemory() { return GetPixelBufferBytes() + 32; }

#ifndef CL_VERSION_1_2
static cl_int clUnloadCompiler11(cl_platform_id) {
  return clUnloadCompiler();
}
#endif

typedef cl_int (*clUnloadCompilerFunc)(cl_platform_id);

#ifdef CL_VERSION_1_2
static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadPlatformCompiler;
#else
static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadCompiler11;
#endif

static cl_platform_id GetPlatformForContext(cl_context ctx) {
  size_t num_props;
  cl_context_properties props[128];
  CHECK_CL(clGetContextInfo, ctx, CL_CONTEXT_PROPERTIES, sizeof(props), &props, &num_props);
  num_props /= sizeof(cl_context_properties);

  for (int i = 0; i < num_props; i += 2) {
    if (props[i] == CL_CONTEXT_PLATFORM) {
      return (cl_platform_id)(props[i + 1]);
    }
  }

  assert(!"Context has no platform??");
  return (cl_platform_id)(-1);
}

namespace gpu {

void ContextErrorCallback(const char *errinfo, const void *, size_t, void *) {
  fprintf(stderr, "Context error: %s", errinfo);
  assert(false);
  exit(1);
}

void PrintDeviceInfo(cl_device_id device_id) {

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
  assert(*((size_t *)strBuf) >= GetTotalWorkItems());

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
    assert(j != 0 || dimSizes[j] > kLocalWorkSizeX);
    assert(j != 1 || dimSizes[j] > kLocalWorkSizeY);
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

std::vector<cl_context_properties> GetSharedCLGLProps() {
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

cl_context InitializeOpenCL(bool share_opengl) {
  const cl_uint kMaxPlatforms = 8;
  cl_platform_id platforms[kMaxPlatforms];
  cl_uint nPlatforms;
  CHECK_CL(clGetPlatformIDs, kMaxPlatforms, platforms, &nPlatforms);

#ifndef NDEBUG
  size_t strLen;
  fprintf(stdout, "\n");
  fprintf(stdout, "Found %d OpenCL platform%s.\n", nPlatforms, nPlatforms == 1? "" : "s");

  for(cl_uint i = 0; i < nPlatforms; i++) {
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
  fprintf(stdout, "Found %d device%s on platform 0.\n", nDevices, nDevices == 1? "" : "s");

  for(cl_uint i = 0; i < nDevices; i++) {
    PrintDeviceInfo(devices[i]);
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
  } else {
    props.push_back(0);
  }

  CreateCLContext(&ctx, props.data(), devices[0]);

  return ctx;
}

cl_device_id GetDeviceForSharedContext(cl_context ctx) {
  size_t device_id_size_bytes;
  cl_device_id device;

#ifndef __APPLE__
  std::vector<cl_context_properties> props = GetSharedCLGLProps();

  typedef CL_API_ENTRY cl_int (CL_API_CALL *CtxInfoFunc)
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

  assert (getCtxInfoFunc);
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
  assert (device_id_size_bytes == sizeof(cl_device_id));
  return device;
}

std::vector<cl_device_id> GetAllDevicesForContext(cl_context ctx) {
  std::vector<cl_device_id> devices(16);
  size_t nDeviceIds;
  CHECK_CL(clGetContextInfo, ctx, CL_CONTEXT_DEVICES, devices.size() * sizeof(cl_device_id),
           devices.data(), &nDeviceIds);
  nDeviceIds /= sizeof(cl_device_id);
  devices.resize(nDeviceIds);
  return std::move(devices);
}

LoadedCLKernel InitializeOpenCLKernel(const char *source_filename, const char *kernel_name,
                                      cl_context ctx, cl_device_id device) {
#ifndef NDEBUG
  // If the total local memory required is greater than the minimum specified.. check!
  size_t strLen;
  cl_ulong totalLocalMemory;
  CHECK_CL(clGetDeviceInfo, device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
           &totalLocalMemory, &strLen);
  assert(strLen == sizeof(cl_ulong));
  assert(totalLocalMemory >= 16384);
  while(GetTotalLocalMemory() > totalLocalMemory) {
    kLocalWorkSizeX >>= 1;
    kLocalWorkSizeY >>= 1;
  }
#endif

  std::ifstream progfs(source_filename);
  std::string progStr((std::istreambuf_iterator<char>(progfs)),
                      std::istreambuf_iterator<char>());
  const char *progCStr = progStr.c_str();

  cl_int errCreateProgram;
  cl_program program;
  program = clCreateProgramWithSource(ctx, 1, &progCStr, NULL, &errCreateProgram);
  CHECK_CL((cl_int), errCreateProgram);

  if(clBuildProgram(program, 1, &device, "-Werror", NULL, NULL) != CL_SUCCESS) {
    size_t bufferSz;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(size_t), NULL, &bufferSz);

    char *buffer = new char[bufferSz + 1];
    buffer[bufferSz] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          bufferSz, buffer, NULL);

    std::cerr << "CL Compilation failed:" << std::endl;
    std::cerr << buffer + 1 << std::endl;
    abort();
  }
#ifndef NDEBUG
  else {
    std::cerr << "CL Kernel compiled successfully!" << std::endl;
  }
#endif
  CHECK_CL(gUnloadCompilerFunc, GetPlatformForContext(ctx));

  LoadedCLKernel kernel;

  // !FIXME! Need to share the command queue across devices...
  cl_int errCreateCommandQueue;
#ifndef CL_VERSION_2_0
  kernel._command_queue = clCreateCommandQueue(ctx, device, 0, &errCreateCommandQueue);
#else
  kernel._command_queue = clCreateCommandQueueWithProperties(ctx, device, 0, &errCreateCommandQueue);
#endif
  CHECK_CL((cl_int), errCreateCommandQueue);

  cl_int errCreateKernel;
  kernel._kernel = clCreateKernel(program, kernel_name, &errCreateKernel);
  CHECK_CL((cl_int), errCreateKernel);

  CHECK_CL(clReleaseProgram, program);

  return kernel;
}

void DestroyOpenCLKernel(const LoadedCLKernel &kernel) {
  CHECK_CL(clReleaseKernel, kernel._kernel);
  CHECK_CL(clReleaseCommandQueue, kernel._command_queue);
}

void ShutdownOpenCL(cl_context ctx) {
  CHECK_CL(clReleaseContext, ctx);
}

}  // namespace gpu
