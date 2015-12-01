#include "kernel_cache.h"

#include <iostream>
#include <fstream>

#ifndef CL_VERSION_1_2
static cl_int clUnloadCompiler11(cl_platform_id) {
  return clUnloadCompiler();
}
#endif

typedef cl_int(*clUnloadCompilerFunc)(cl_platform_id);

#ifdef CL_VERSION_1_2
static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadPlatformCompiler;
#else
static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadCompiler11;
#endif

namespace gpu {

GPUKernelCache *gKernelCache = nullptr;

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

static cl_kernel LoadKernel(const char *source_filename, const char *kernel_name,
                            cl_context ctx, cl_device_id device) {
  std::ifstream progfs(source_filename);
  std::string progStr((std::istreambuf_iterator<char>(progfs)),
    std::istreambuf_iterator<char>());
  const char *progCStr = progStr.c_str();

  cl_program program;
  cl_int errCreateProgram;
  program = clCreateProgramWithSource(ctx, 1, &progCStr, NULL, &errCreateProgram);
  CHECK_CL((cl_int), errCreateProgram);

  if (clBuildProgram(program, 1, &device, "-Werror", NULL, NULL) != CL_SUCCESS) {
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

  cl_int errCreateKernel;
  cl_kernel kernel = clCreateKernel(program, kernel_name, &errCreateKernel);
  CHECK_CL((cl_int), errCreateKernel);

  CHECK_CL(clReleaseProgram, program);

  return kernel;
}

GPUKernelCache *GPUKernelCache::Instance(cl_context ctx, cl_device_id device) {
  if (gKernelCache == nullptr) {
    gKernelCache = new GPUKernelCache(ctx, device);
  }

  // !FIXME! This comparison might not be cross-platform...
  if (ctx == gKernelCache->_ctx && device == gKernelCache->_device) {
    return gKernelCache;
  }

  Clear();
  gKernelCache = new GPUKernelCache(ctx, device);
  return gKernelCache;
}

void GPUKernelCache::Clear() {
  if (gKernelCache == nullptr) {
    return;
  }

  for (auto kernel_pair : gKernelCache->_kernels) {
    CHECK_CL(clReleaseKernel, kernel_pair.second);
  }

  delete gKernelCache;
  gKernelCache = nullptr;
}

cl_kernel GPUKernelCache::GetKernel(const std::string &filename, const std::string &kernel) {
  std::string kernel_name = filename + ":" + kernel;

  if (_kernels.find(kernel_name) == _kernels.end()) {
    _kernels[kernel_name] = LoadKernel(filename.c_str(), kernel.c_str(), _ctx, _device);
  }

  return _kernels[kernel_name];
}

}  // namespace gpu