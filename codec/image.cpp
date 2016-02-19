#include "image.h"

namespace GenTC {

uint64_t ReadValue(const std::vector<uint8_t> &img_data, size_t *bit_offset, size_t prec) {
  size_t next_bit_offset = *bit_offset + prec;

  uint64_t ch_val = 0;

  size_t bits_to_byte = 8 - (*bit_offset % 8);
  size_t bits_left = std::min(bits_to_byte, prec);
  if (8 != bits_left) {
    size_t val = img_data[*bit_offset / 8] & ((1 << bits_to_byte) - 1);
    if (bits_left != bits_to_byte) {
      val >>= bits_to_byte - bits_left;
    }
    ch_val |= val;
    *bit_offset += bits_left;
  }

  assert(*bit_offset == next_bit_offset || (*bit_offset % 8) == 0);

  size_t next_byte_offset = next_bit_offset / 8;
  while ((*bit_offset / 8) < next_byte_offset) {
    ch_val <<= 8;
    ch_val |= img_data[*bit_offset / 8];
    *bit_offset += 8;
  }

  bits_left = next_bit_offset - *bit_offset;
  assert(bits_left < 8);
  if (bits_left > 0) {
    size_t shift = 8 - bits_left;
    ch_val <<= bits_left;
    ch_val |= img_data[*bit_offset / 8] >> shift;
  }

  assert(*bit_offset + bits_left == next_bit_offset);
  *bit_offset = next_bit_offset;
  return ch_val;
}

}  // namespace GenTC
