#include "fast_dct.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// #define VERBOSE
#include "encoder.h"
#include "decoder.h"
#include "gpu.h"
#include "kernel_cache.h"

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#else
#pragma warning(disable: 4996)
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#else
#pragma warning(default: 4996)
#endif

int main(int argc, char **argv) {
  // Make sure that we have the proper number of arguments...
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << "<original> [compressed]" << std::endl;
    return 1;
  }

  std::unique_ptr<gpu::GPUContext> ctx = gpu::GPUContext::InitializeOpenCL(false);

  const char *orig_fn = argv[1];
  const char *cmp_fn = (argc == 2) ? NULL : argv[2];

#if 0
  GenTC::DXTImage dxt_img = GenTC::DXTImage(orig_fn, cmp_fn);
#else
  std::vector<uint8_t> cmp_img = std::move(GenTC::CompressDXT(orig_fn, cmp_fn));
  GenTC::DXTImage dxt_img = GenTC::DecompressDXT(ctx, cmp_img);
#endif

  // Decompress into image...
  std::vector<uint8_t> decomp_rgba = std::move(dxt_img.DecompressedImage()->Pack());

  stbi_write_png("img_dxt.png", dxt_img.Width(), dxt_img.Height(), 4, decomp_rgba.data(), 4 * dxt_img.Width());

  // Visualize interpolation data...
  std::vector<uint8_t> interp_img_data = std::move(dxt_img.InterpolationImage());
  stbi_write_png("img_dxt_interp.png", dxt_img.Width(), dxt_img.Height(), 1, interp_img_data.data(), dxt_img.Width());

  clFlush(ctx->GetDefaultCommandQueue());
  clFinish(ctx->GetDefaultCommandQueue());
  gpu::GPUKernelCache::Clear();

  return 0;
}
