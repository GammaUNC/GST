#include "kernel_cache.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <mutex>

#ifndef CL_VERSION_1_2
static cl_int clUnloadCompiler11(cl_platform_id) {
  return clUnloadCompiler();
}
#endif

typedef cl_int(STDCALL *clUnloadCompilerFunc)(cl_platform_id);

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

  for (size_t i = 0; i < num_props; i += 2) {
    if (props[i] == CL_CONTEXT_PLATFORM) {
      return (cl_platform_id)(props[i + 1]);
    }
  }

  assert(!"Context has no platform??");
  return (cl_platform_id)(-1);
}

static cl_program CompileProgram(const char *source_filename, cl_context ctx,
                                 EContextType ctx_ty, EOpenCLVersion ver,
                                 cl_device_id device) {
  std::ifstream progfs(source_filename, std::ifstream::in);
  if (!progfs) {
    assert(!"Error opening file!");
    abort();
  }

  std::string progStr((std::istreambuf_iterator<char>(progfs)),
                       std::istreambuf_iterator<char>());

  // Internal error! We should never call this function without exactly knowing
  // what we're getting ourselves into...
  if (progStr.empty()) {
    assert(false);
    abort();
  }

  const char *progCStr = progStr.c_str();

  cl_int errCreateProgram;
  cl_program program = clCreateProgramWithSource(ctx, 1, &progCStr, NULL, &errCreateProgram);
  CHECK_CL((cl_int), errCreateProgram);

  std::string args("-Werror ");

  if (ctx_ty == eContextType_IntelCPU && ver >= eOpenCLVersion_20) {
    // !FIXME! Currently crashes build_table kernel
    if (!strstr(source_filename, "build_table.cl"))
      args += std::string("-g ");
    args += std::string("-s \"");
    args += std::string(source_filename);
    args += std::string("\" ");
  }

  cl_int build_program_result = clBuildProgram(program, 1, &device, args.c_str(), NULL, NULL);
  if (build_program_result == CL_BUILD_PROGRAM_FAILURE) {
    size_t bufferSz;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(size_t), NULL, &bufferSz);

    char *buffer = new char[bufferSz + 1];
    buffer[bufferSz] = '\0';
    CHECK_CL(clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG,
                                    bufferSz, buffer, NULL);

    std::cerr << "CL Compilation failed:" << std::endl;
    std::cerr << buffer + 1 << std::endl;
    abort();
  }
#ifndef NDEBUG
  else if (build_program_result == CL_SUCCESS) {
    std::cerr << "CL Program " << source_filename << " compiled successfully!" << std::endl;
  }
#endif
  CHECK_CL((cl_int), build_program_result);
  CHECK_CL(gUnloadCompilerFunc, GetPlatformForContext(ctx));

  return program;
}

static std::mutex gKernelCacheMutex;

GPUKernelCache *GPUKernelCache::Instance(cl_context ctx, EContextType ctx_ty,
                                         EOpenCLVersion ctx_ver, cl_device_id device) {
  std::unique_lock<std::mutex> lock(gKernelCacheMutex);

  if (gKernelCache == nullptr) {
    gKernelCache = new GPUKernelCache(ctx, ctx_ty, ctx_ver, device);
  }

  // !FIXME! This comparison might not be cross-platform...
  if (ctx == gKernelCache->_ctx && device == gKernelCache->_device) {
    return gKernelCache;
  }

  Clear();
  gKernelCache = new GPUKernelCache(ctx, ctx_ty, ctx_ver, device);
  return gKernelCache;
}

void GPUKernelCache::Clear() {
  if (gKernelCache == nullptr) {
    return;
  }

  for (auto pgm : gKernelCache->_programs) {
    GPUProgram *program = &(pgm.second);
    for (auto krnl : program->_kernels) {
      CHECK_CL(clReleaseKernel, krnl.second);
    }
    CHECK_CL(clReleaseProgram, program->_prog);
  }

  delete gKernelCache;
  gKernelCache = nullptr;
}

cl_kernel GPUKernelCache::GetKernel(const std::string &filename, const std::string &kernel) {
  std::unique_lock<std::mutex> lock(gKernelCacheMutex);
  if (_programs.find(filename) == _programs.end()) {
    _programs[filename]._prog =
      CompileProgram(filename.c_str(), _ctx, _ctx_ty, _ctx_ver, _device);
  }

  GPUProgram *program = &(_programs[filename]);
  if (program->_kernels.find(kernel) == program->_kernels.end()) {
    cl_int errCreateKernel;
    program->_kernels[kernel] = clCreateKernel(program->_prog, kernel.c_str(), &errCreateKernel);
    CHECK_CL((cl_int), errCreateKernel);

#ifndef NDEBUG
    std::cout << "Loaded CL Kernel " << kernel << "..." << std::endl;
#endif
  }

  return program->_kernels[kernel];
}

}  // namespace gpu
