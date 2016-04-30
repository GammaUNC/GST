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

  // Optional to compile kernels so that we don't have to do it at runtime.
  // Returns true if our platform meets all of the expectations...
  bool InitializeDecoder(const std::unique_ptr<gpu::GPUContext> &gpu_ctx);
  DXTImage DecompressDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                         const std::vector<uint8_t> &cmp_data);

  struct GenTCHeader {
    struct rANSInfo {
      uint32_t sz;
      uint32_t num_offsets;
    };

    uint32_t width;
    uint32_t height;
    rANSInfo ep1_y;
    rANSInfo ep1_co;
    rANSInfo ep1_cg;
    rANSInfo ep2_y;
    rANSInfo ep2_co;
    rANSInfo ep2_cg;
    rANSInfo palette;
    rANSInfo indices;

    void Print() const;
    void LoadFrom(const uint8_t *buf);
  };

  std::vector<cl_event> LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                                          const GenTCHeader &hdr, cl_command_queue queue,
                                          cl_mem cmp_data, cl_mem output, cl_event init);

  bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
               const char *filename, const char *cmp_fn);

}  // namespace GenTC

#endif  // __TCAR_CODEC_H__
