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
// static uint32_t BitReverse(uint32_t x, uint32_t num_bits) {
//   uint32_t result = 0;
//   for (uint32_t i = 0; i < num_bits; ++i) {
//     result <<= 1;
//     result |= (x & 1);
//     x >>= 1;
//   }
//   return result;
// }

static std::vector<uint32_t> CumulativeSum(const std::vector<uint32_t> &x) {
  std::vector<uint32_t> t(x.size(), 0);
  std::partial_sum(x.begin(), x.end() - 1, t.begin() + 1);
  return std::move(t);
}

static bool OptionsValid(const ans::Options &opts) {
  bool ok = true;

  // We have to have at least some frequencies
  ok = ok && opts.Fs.size() > 1;

  // Make sure we can represent all states
  ok = ok && (static_cast<uint64_t>(opts.b) *
              static_cast<uint64_t>(opts.k) *
              static_cast<uint64_t>(opts.M)) < (1ULL << 32);

  // We have to spit out at least bits somehow.
  ok = ok && opts.b > 0;
  ok = ok && (opts.b & (opts.b - 1)) == 0;

  // This is our state resolution factor, it has to be
  // at least one, otherwise our states are zero.
  ok = ok && opts.k > 0;

  ok = ok && opts.M > 0;

  return ok;
}

// Returns false if unfixable...
static bool FixInvalidOptions(ans::Options *opts) {
  if (OptionsValid(*opts))
    return true;

  // If our probability denominator is invalid, set it
  // to the proper denominator...
  if (0 == opts->M) {
    opts->M = std::accumulate(opts->Fs.begin(), opts->Fs.end(), 0);
  }

  if (0 == opts->k) {
    opts->k = 1;
  }

  if (0 == opts->b) {
    opts->b = 2;
  }

  return OptionsValid(*opts);
}


#endif  // __ANS_UTILS_H__
