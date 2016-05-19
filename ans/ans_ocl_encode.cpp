#include "ans.h"
#include "histogram.h"

namespace ans {
namespace ocl {
  std::vector<uint32_t> NormalizeFrequencies(const std::vector<uint32_t> &F) {
    return std::move(ans::GenerateHistogram(F, kANSTableSize));
  }

  ans::Options GetOpenCLOptions(const std::vector<uint32_t> &F) {
    Options opts;
    opts.b = 1 << 16;
    opts.k = 1 << 4;
    opts.M = kANSTableSize;
    opts.Fs = F;
    opts.type = eType_rANS;
    return opts;
  }
}  // namespace ocl
}  // namespace ans
