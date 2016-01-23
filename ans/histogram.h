#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <cstdint>
#include <limits>
#include <vector>

namespace ans {

  // Generates a frequency table based on `counts` and the
  // shared denominator 'M'. Symbols are assumed to be 
  // [0, counts.size()).
  std::vector<uint32_t> GenerateHistogram(const std::vector<uint32_t> &counts,
                                          const int M);

  template<typename T>
  std::vector<uint32_t> CountSymbols(const std::vector<T> &symbols) {
    uint32_t minval = static_cast<uint32_t>(std::numeric_limits<T>::min());
    uint32_t maxval = static_cast<uint32_t>(std::numeric_limits<T>::max());
    uint32_t range = maxval - minval + 1;

    std::vector<uint32_t> counts(range, 0);
    for (auto s : symbols) {
      counts[s - minval]++;
    }

    return std::move(counts);
  }
} // namespace ans

#endif // __HISTOGRAM_H__
