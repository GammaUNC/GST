#ifndef __TCAR_DECODER_H__
#define __TCAR_DECODER_H__

#include <cstdint>
#include <functional>
#include <vector>

#include "dxt_image.h"
#include "gpu.h"
#include "codec_base.h"

namespace GenTC {
  // Optional to compile kernels so that we don't have to do it at runtime.
  // Returns true if our platform meets all of the expectations...
  bool InitializeDecoder(const std::unique_ptr<gpu::GPUContext> &gpu_ctx);
  DXTImage DecompressDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                         const std::vector<uint8_t> &cmp_data);

  cl_event LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                             const GenTCHeader &hdr, cl_command_queue queue,
                             cl_mem cmp_data, cl_mem output, cl_uint num_init, const cl_event *init);

  cl_event LoadCompressedDXTs(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                              const std::vector<GenTCHeader> &hdr, cl_command_queue queue,
                              cl_mem cmp_data, cl_mem output, cl_uint num_init, const cl_event *init);

  cl_event LoadRGB(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                   const GenTCHeader &hdr, cl_command_queue queue,
                   cl_mem cmp_data, cl_mem output, cl_uint num_init, const cl_event *init);

  cl_event LoadRGBs(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                    const std::vector<GenTCHeader> &hdr, cl_command_queue queue,
                    cl_mem cmp_data, cl_mem output, cl_uint num_init, const cl_event *init);

  size_t RequiredScratchMem(const GenTCHeader &hdr);
  void PreallocateDecompressor(const std::unique_ptr<gpu::GPUContext> &gpu_ctx, size_t req_sz);
  void FreeDecompressor();
}  // namespace GenTC

#endif  // __TCAR_DECODER_H__
