#ifndef __TCAR_CODEC_BASE_H__
#define __TCAR_CODEC_BASE_H__

#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace GenTC {
  struct GenTCHeader {
    uint32_t width;
    uint32_t height;
    uint32_t palette_bytes;
    uint32_t y_cmp_sz;
    uint32_t chroma_cmp_sz;
    uint32_t palette_sz;
    uint32_t indices_sz;

    void Print() const;
    void LoadFrom(const uint8_t *buf);
  };

  static const size_t kWaveletBlockDim = 32;
  static_assert((kWaveletBlockDim % 2) == 0, "Wavelet dimension must be power of two!");
}

#endif  // __TCAR_CODEC_BASE_H__
