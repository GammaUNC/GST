#ifndef __TCAR_CODEC_H__
#define __TCAR_CODEC_H__

#include <cstdint>
#include <functional>
#include <vector>

#include "dxt_image.h"
#include "gpu.h"

namespace GenTC {
  // Compresses the DXT texture with the given width and height into a
  // GPU decompressible stream.
  std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn);
  std::vector<uint8_t> CompressDXT(int width, int height,
                                   const std::vector<uint8_t> &rgb_data,
                                   const std::vector<uint8_t> &dxt_data);
  std::vector<uint8_t> CompressDXT(const DXTImage &dxt_img);

  // Optional to compile kernels so that we don't have to do it at runtime.
  // Returns true if our platform meets all of the expectations...
  bool InitializeDecoder(const std::unique_ptr<gpu::GPUContext> &gpu_ctx);
  DXTImage DecompressDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                         const std::vector<uint8_t> &cmp_data);

  struct GenTCHeader {
    uint32_t width;
    uint32_t height;
    uint32_t palette_bytes;
    uint32_t y_cmp_sz;
    uint32_t chroma_cmp_sz;
    uint32_t palette_sz;
    uint32_t indices_sz;

    size_t RequiredScratchMem() const;
    void Print() const;
    void LoadFrom(const uint8_t *buf);
  };

  cl_event LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                             const GenTCHeader &hdr, cl_command_queue queue,
                             cl_mem cmp_data, cl_mem output, cl_event init);

  cl_event LoadCompressedDXTs(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                              const std::vector<GenTCHeader> &hdr, cl_command_queue queue,
                              cl_mem cmp_data, cl_mem output, cl_event init);

  cl_event LoadRGB(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                   const GenTCHeader &hdr, cl_command_queue queue,
                   cl_mem cmp_data, cl_mem output, cl_event init);

  cl_event LoadRGBs(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                    const std::vector<GenTCHeader> &hdr, cl_command_queue queue,
                    cl_mem cmp_data, cl_mem output, cl_event init);

  void PreallocateDecompressor(const std::unique_ptr<gpu::GPUContext> &gpu_ctx, size_t req_sz);
  void FreeDecompressor();

  bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
               const char *filename, const char *cmp_fn);

}  // namespace GenTC

#endif  // __TCAR_CODEC_H__
