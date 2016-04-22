#ifndef __GPU_KERNEL_CACHE_H__
#define __GPU_KERNEL_CACHE_H__

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu.h"

namespace gpu {

class GPUKernelCache {
public:
  static GPUKernelCache *Instance(cl_context ctx, EContextType ctx_ty,
                                  EOpenCLVersion ctx_ver, cl_device_id device);
  static void Clear();

  cl_kernel GetKernel(const std::string &filename,
                      const std::string &kernel);
private:
  // disallow copying...
  GPUKernelCache(cl_context ctx, EContextType ctx_ty, EOpenCLVersion ctx_ver, cl_device_id device)
    : _ctx(ctx), _ctx_ty(ctx_ty), _ctx_ver(ctx_ver), _device(device) { }
  GPUKernelCache(const GPUKernelCache&);

  cl_context _ctx;
  EContextType _ctx_ty;
  EOpenCLVersion _ctx_ver;

  cl_device_id _device;

  std::mutex _kernel_creation_mutex;

  struct GPUProgram {
    cl_program _prog;
    std::unordered_map<std::string, cl_kernel> _kernels;
  };
  std::unordered_map<std::string, GPUProgram> _programs;
};

}

#endif  // __GPU_KERNEL_CACHE_H__
