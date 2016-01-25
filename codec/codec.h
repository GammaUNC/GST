#ifndef __TCAR_CODEC_H__
#define __TCAR_CODEC_H__

#include <cstdint>
#include <vector>

namespace GenTC {

  // Compresses the DXT texture with the given width and height into a
  // GPU decompressible stream.
  std::vector<uint8_t> CompressDXT(const uint8_t *dxt, int width, int height);

}  // namespace GenTC

#endif  // __TCAR_CODEC_H__