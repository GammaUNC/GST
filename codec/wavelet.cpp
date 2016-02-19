#include "wavelet.h"

#include <vector>

// Returns a normalized index in the given range by ping-ponging
// across boundaries...
//
// e.g. for an array of size 5, ABCDE, we interpret 'idx' as an index
// into the infinite array .....CBABCDEDCBABCDE... such that idx 0 is
// at the start of ABCDE. Hence, f(-1, 5) = B, and f(-6, 5) = C, etc
static int NormalizeIndex(int idx, size_t range) {
  int r = static_cast<int>(range - 1);
  int pidx = std::abs(idx);
  int idx_bucket = pidx / r;

  int next_lowest = idx_bucket * r;

  if (idx_bucket % 2 == 0) {
    return pidx - next_lowest;
  } else {
    return next_lowest + r - pidx;
  }
}

static void Transpose(int16_t *img, size_t dim, size_t rowbytes) {
  uint8_t *bytes = reinterpret_cast<uint8_t *>(img);
  for (size_t y = 0; y < dim; ++y) {
    for (size_t x = y + 1; x < dim; ++x) {
      int16_t *v1 = reinterpret_cast<int16_t *>(bytes + y * rowbytes);
      int16_t *v2 = reinterpret_cast<int16_t *>(bytes + x * rowbytes);
      std::swap(v1[x], v2[y]);
    }
  }
}

namespace GenTC {

size_t ForwardWavelet1D(const int16_t *src, int16_t *dst, size_t len) {
  if (len == 0) {
    return 0;
  }

  if (len == 1) {
    dst[0] = src[0];
    return 0;
  }

  // First set the output buffer to zero:
  memset(dst, 0, len * sizeof(dst[0]));

  // Deinterleave everything
  const size_t mid_pt = len - (len / 2);

  // Do the odd coefficients first
  for (int i = 1; i < len; i += 2) {
    int next = NormalizeIndex(i + 1, len);
    int prev = NormalizeIndex(i - 1, len);
    dst[mid_pt + i / 2] = src[i] - (src[prev] + src[next]) / 2;
  }

  // Do the even coefficients second
  for (int i = 0; i < len; i += 2) {
    int next = static_cast<int>(mid_pt) + NormalizeIndex(i + 1, len) / 2;
    int prev = static_cast<int>(mid_pt) + NormalizeIndex(i - 1, len) / 2;
    dst[i / 2] = src[i] + (dst[prev] + dst[next] + 2) / 4;
  }

  return mid_pt;
}

void InverseWavelet1D(const int16_t *src, int16_t *dst, size_t len) {
  if (len == 0) {
    return;
  }

  if (len == 1) {
    dst[0] = src[0];
    return;
  }

  // First set the output buffer to zero:
  memset(dst, 0, len * sizeof(dst[0]));

  // Interleave everything
  const size_t mid_pt = len - (len / 2);

  // Do the even coefficients first
  for (int i = 0; i < len; i += 2) {
    int prev = static_cast<int>(mid_pt) + NormalizeIndex(i - 1, len) / 2;
    int next = static_cast<int>(mid_pt) + NormalizeIndex(i + 1, len) / 2;
    dst[i] = src[i / 2] - (src[prev] + src[next] + 2) / 4;
  }

  // Do the odd coefficients second
  for (int i = 1; i < len; i += 2) {
    int src_idx = static_cast<int>(mid_pt) + i / 2;
    int prev = NormalizeIndex(i - 1, len);
    int next = NormalizeIndex(i + 1, len);
    dst[i] = src[src_idx] + (dst[prev] + dst[next]) / 2;
  }
}

void ForwardWavelet2D(const int16_t *src, size_t src_rowbytes,
                      int16_t *dst, size_t dst_rowbytes, size_t dim) {
  // Allocate a bit of scratch memory
  std::vector<int16_t> scratch(dim * dim);
  const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src);
  uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst);

  // Go through and do all the rows
  for (size_t row = 0; row < dim; ++row) {
    const int16_t *img = reinterpret_cast<const int16_t *>(src_bytes + row*src_rowbytes);
    ForwardWavelet1D(img, scratch.data() + row*dim, dim);
  }

  // Do all the columns...
  Transpose(scratch.data(), dim, sizeof(scratch[0]) * dim);
  for (size_t col = 0; col < dim; ++col) {
    int16_t *dst_img = reinterpret_cast<int16_t *>(dst_bytes + col*dst_rowbytes);
    ForwardWavelet1D(scratch.data() + col * dim, dst_img, dim);
  }

  Transpose(dst, dim, dst_rowbytes);
}

extern void InverseWavelet2D(const int16_t *src, size_t src_rowbytes,
                             int16_t *dst, size_t dst_rowbytes, size_t dim) {
  // Allocate a bit of scratch memory
  std::vector<int16_t> scratch(dim * dim);
  const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src);
  uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst);

  // Copy src into scratch
  for (size_t row = 0; row < dim; ++row) {
    const int16_t *img = reinterpret_cast<const int16_t *>(src_bytes + row * src_rowbytes);
    memcpy(scratch.data() + row*dim, img, sizeof(scratch[0]) * dim);
  }

  // Do all the columns, store into dst
  Transpose(scratch.data(), dim, sizeof(scratch[0]) * dim);

  for (size_t col = 0; col < dim; ++col) {
    int16_t *dst_img = reinterpret_cast<int16_t *>(dst_bytes + col * dst_rowbytes);
    InverseWavelet1D(scratch.data() + col * dim, dst_img, dim);
  }

  Transpose(dst, dim, dst_rowbytes);

  // Copy dst back into scratch
  for (size_t row = 0; row < dim; ++row) {
    const int16_t *img = reinterpret_cast<const int16_t *>(dst_bytes + row * dst_rowbytes);
    memcpy(scratch.data() + row*dim, img, sizeof(scratch[0]) * dim);
  }

  // Do all the rows, store into dst
  for (size_t col = 0; col < dim; ++col) {
    int16_t *dst_img = reinterpret_cast<int16_t *>(dst_bytes + col * dst_rowbytes);
    InverseWavelet1D(scratch.data() + col * dim, dst_img, dim);
  }
}

}