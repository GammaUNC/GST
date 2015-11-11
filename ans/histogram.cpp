#include "histogram.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

namespace ans {
  struct Symbol {
    int symbol;
    double rank;
    bool operator<(const ans::Symbol &rhs) const {
      return rank < rhs.rank;
    }
    bool operator>(const ans::Symbol &rhs) const {
      return rank > rhs.rank;
    }
  };

#ifdef WIN32
  double log2(double n) {
    return log(n) / log(2.0);
  }
#endif

  static double GetFreqChange(int count, int new_count, int correction_sign) {
    return log2(static_cast<double>(new_count) /
                static_cast<double>(new_count + correction_sign)) *
      static_cast<double>(count);
  }

  // This normalization technique is taken from the discussion presented
  // on Charles Bloom's blog:
  // http://cbloomrants.blogspot.com/2014/02/02-11-14-understanding-ans-10.html
  void GenerateHistogram(std::vector<int> *histogram,
                         const std::vector<int> &counts,
                         const int M) {
    if (!histogram) {
      assert(!"NULL passed for output!");
      return;
    }

    if (M <= 0) {
      assert(!"Improper target sum for frequencies!");
      return;
    }

    histogram->clear();
    histogram->reserve(counts.size());

    int sum = std::accumulate(counts.begin(), counts.end(), 0, std::plus<int>());
    for (size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] == 0) {
        continue;
      }

      double from_scaled = static_cast<float>(counts[i] * M) / static_cast<float>(sum);      
      int down = static_cast<int>(from_scaled);

      histogram->push_back(std::max(1, (from_scaled * from_scaled <= down * (down + 1))? down : down + 1));
    }

    int correction = M - std::accumulate(histogram->begin(), histogram->end(), 0, std::plus<int>());
    if (correction == 0) {
      // No work to do, averaging was exact.
      return;
    }

    if (correction == M) {
      assert(!"No symbols have any frequency!");
      return;
    }

    std::vector<Symbol> symbols;
    symbols.reserve(counts.size());

    int correction_sign = (correction >> 31) | static_cast<int>(correction > 0);
    for (size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] == 0) {
        continue;
      }

      assert(histogram->at(i) > 0);
      if (histogram->at(i) > 1 || correction > 0) {
        Symbol s;
        s.symbol = static_cast<int>(i);
        s.rank = GetFreqChange(counts[i], histogram->at(i), correction_sign);
        symbols.push_back(s);
      }
    }

    std::make_heap(symbols.begin(), symbols.end(), std::greater<Symbol>());

    while (correction != 0) {
      assert(!symbols.empty());
      std::pop_heap(symbols.begin(), symbols.end(), std::greater<Symbol>());

      Symbol s = symbols.back();
      symbols.pop_back();

      int i = s.symbol;
      assert(counts[i] != 0);

      histogram->at(i) += correction_sign;
      correction -= correction_sign;
      assert(histogram->at(i) != 0);

      if (histogram->at(i) > 1 || correction_sign == 1) {
        Symbol s;
        s.symbol = i;
        s.rank = GetFreqChange(counts[i], histogram->at(i), correction_sign);
        symbols.push_back(s);

        std::push_heap(symbols.begin(), symbols.end(), std::greater<Symbol>());
      }
    }

    assert(std::accumulate(histogram->begin(), histogram->end(), 0, std::plus<int>()) == M);
  }

} // namespace ans
