#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <cstdint>
#include <limits>
#include <vector>

namespace ans {

  // Generates a frequency table based on `counts` and the
  // shared denominator 'M'. Symbols are assumed to be 
  // [0, counts.size()).
  std::vector<int> GenerateHistogram(const std::vector<int> &counts,
                                     const int M);

  template<typename T>
  std::vector<int> CountSymbols(const std::vector<T> &symbols) {
    uint32_t minval = static_cast<uint32_t>(std::numeric_limits<T>::min());
    uint32_t maxval = static_cast<uint32_t>(std::numeric_limits<T>::max());
    uint32_t range = maxval - minval + 1;

    std::vector<int> counts(range, 0);
    for (auto s : symbols) {
      counts[s - minval]++;
    }

    return std::move(counts);
  }
} // namespace ans

#endif // __HISTOGRAM_H__
