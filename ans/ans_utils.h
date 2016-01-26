#ifndef __ANS_UTILS_H__
#define __ANS_UTILS_H__

#include <cstdint>
#include <vector>
#include <numeric>

////////////////////////////////////////////////////////////////////////////////
//
// Utilities
//

// Yes, I know there are faster implementations of this...
static int IntLog2(uint32_t x) {
  if (x == 0) {
    return 0;
  }

  int l = 0;
  while (x >>= 1) {
    l++;
  }
  return l;
}

// Yes, I know there's probably a faster way to do this, too...
static uint32_t BitReverse(uint32_t x, uint32_t num_bits) {
  uint32_t result = 0;
  for (uint32_t i = 0; i < num_bits; ++i) {
    result <<= 1;
    result |= (x & 1);
    x >>= 1;
  }
  return result;
}

static std::vector<uint32_t> CumulativeSum(const std::vector<uint32_t> &x) {
  std::vector<uint32_t> t(x.size(), 0);
  std::partial_sum(x.begin(), x.end() - 1, t.begin() + 1);
  return std::move(t);
}

#endif  // __ANS_UTILS_H__