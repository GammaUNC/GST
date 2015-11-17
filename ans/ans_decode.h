#ifndef __ANS_DECODE_H__
#define __ANS_DECODE_H__

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include "bits.h"

namespace ans {

  // rANS encode.
  template<uint32_t b, uint32_t k>
  class Decoder {
  public:
    // The constructor initializes M to be the sum of all Fs. It uses k
    // to determine the proper normalization interval for encoding. It uses
    // b to know how many bytes/bits etc to emit at a time.
    Decoder(uint32_t state, const std::vector<uint32_t> &Fs)
      : _F(Fs)
      , _B([&Fs] {
          std::vector<uint32_t> t(Fs.size(), 0);
          std::partial_sum(Fs.begin(), Fs.end() - 1, t.begin() + 1);
          return t;
        }())
      , _M(_B.back() + _F.back())
      , _L(k * _M)
      , _log_b([] {
          uint32_t bb = b;
          if (bb == 0) { return 0; }

          int l = 0;
          while(bb >>= 1) { l++; }
          return l;
        }())
      , _state(state)
    {
      static_assert((b & (b - 1)) == 0, "rANS encoder may only emit powers-of-two for renormalization!");
      static_assert((k & (k - 1)) == 0, "rANS encoder must have power-of-two multiple of precision!");
      assert(_L < (1ULL << 32));
      assert((b * _L) < (1ULL << 32));
    }

    size_t Decode(BitReader &r) {
      assert(_L <= _state && _state < (b * _L));

      // Decode
      int symbol = FindSymbol(_state % _M);
      _state = (_state / _M) * _F[symbol] - _B[symbol] + (_state % _M);

      // Renormalize
      while (_state < _L) {
        int new_bits = r.ReadBits(_log_b);
        assert(new_bits < (1 << _log_b));
        _state <<= _log_b;
        _state |= new_bits;
      }

      return symbol;
    }

    uint32_t GetState() const { return _state; }

  private:
    const std::vector<uint32_t> _F;
    const std::vector<uint32_t> _B;

    const uint32_t _M;
    const uint32_t _L;
    const uint32_t _log_b;

    uint32_t _state;

    size_t FindSymbol(uint32_t x) {
      size_t low = 0;
      size_t high = _B.size();

      // !SPEED! it might be better to unroll this table for small M...
      // Search for symbol in Bs...
      for (;;) {
        size_t midpoint = (high + low) >> 1;
        assert(midpoint >= 0);
        assert(midpoint < _F.size());

        if (_B[midpoint] <= x) {
          if (midpoint >= (_B.size() - 1) || _B[midpoint + 1] > x) {
            return midpoint;
          }

          low = midpoint;
        } else {
          if (midpoint == 0 || _B[midpoint - 1] <= x) {
            midpoint = midpoint - 1;
            return midpoint;
          }

          high = midpoint;
        }
      }
    }
  };

}  // namespace ans

#endif  // __ANS_ENCODE_H__
