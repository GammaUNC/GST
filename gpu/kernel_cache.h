#ifndef __GPU_KERNEL_CACHE_H__
#define __GPU_KERNEL_CACHE_H__

#include <string>
#include <unordered_map>
#include <vector>

#include "gpu.h"

namespace gpu {

class GPUKernelCache {
public:
  static GPUKernelCache *Instance(cl_context ctx, cl_device_id device);
  static void Clear();

  cl_kernel GetKernel(const std::string &filename,
                      const std::string &kernel);
private:
  // disallow copying...
  GPUKernelCache(cl_context ctx, cl_device_id device)
    : _ctx(ctx), _device(device) { }
  GPUKernelCache(const GPUKernelCache&);
  GPUKernelCache &operator=(const GPUKernelCache &);

  cl_context _ctx;
  cl_device_id _device;

  std::unordered_map<std::string, cl_kernel> _kernels;
};

}

#endif  // __GPU_KERNEL_CACHE_H__