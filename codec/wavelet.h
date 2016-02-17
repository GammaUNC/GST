#ifndef __TCAR_WAVELET_H__
#define __TCAR_WAVELET_H__

#include <cstdint>

namespace GenTC {

// Performs a 1D reversible integre wavelet transform using the
// 5/3 Daubechies wavelet used in JPEG 2000. Returns the split
// position in the return array where the low-frequency coefficients
// begin
extern size_t ForwardWavelet1D(const int16_t *src, int16_t *dst, size_t len);

extern void InverseWavelet1D(const int16_t *src, int16_t *dst, size_t len);

extern void Wavelet2D(
  const int16_t *src, size_t width, size_t height, size_t src_rowbytes,
  int16_t *dst, size_t dst_rowbytes);

}  // namespace GenTC

#endif  // __TCAR_WAVELET_H__