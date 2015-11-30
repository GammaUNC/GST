#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <cstdint>
#include <vector>

namespace ans {

  // Generates a frequency table based on `counts` and the
  // shared denominator 'M'. Symbols are assumed to be 
  // [0, counts.size()).
  std::vector<int> GenerateHistogram(const std::vector<int> &counts,
                                     const int M);
} // namespace ans

#endif // __HISTOGRAM_H__
