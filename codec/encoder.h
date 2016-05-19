#ifndef __TCAR_ENCODER_H__
#define __TCAR_ENCODER_H__

#include <cstdint>
#include <functional>
#include <vector>

#include "dxt_image.h"

namespace GenTC {
  // Compresses the DXT texture with the given width and height into a
  // GPU decompressible stream.
  std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn);
  std::vector<uint8_t> CompressDXT(int width, int height,
                                   const std::vector<uint8_t> &rgb_data,
                                   const std::vector<uint8_t> &dxt_data);
  std::vector<uint8_t> CompressDXT(const DXTImage &dxt_img);
}  // namespace GenTC

#endif  // __TCAR_ENCODER_H__
