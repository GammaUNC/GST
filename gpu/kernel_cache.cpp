#include "kernel_cache.h"

namespace gpu {

GPUKernelCache *gKernelCache = nullptr;

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
    DestroyOpenCLKernel(kernel_pair.second);
  }

  delete gKernelCache;
  gKernelCache = nullptr;
}

const LoadedCLKernel *GPUKernelCache::GetKernel(const std::string &filename, const std::string &kernel) {
  std::string kernel_name = filename + ":" + kernel;

  if (_kernels.find(kernel_name) == _kernels.end()) {
    _kernels[kernel_name] = gpu::InitializeOpenCLKernel(filename.c_str(), kernel.c_str(), _ctx, _device);
  }

  return &_kernels[kernel_name];
}

}  // namespace gpu