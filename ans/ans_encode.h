#ifndef __ANS_ENCODE_H__
#define __ANS_ENCODE_H__

namespace ans {

  // rANS codec.
  template<uint32_t b, uint32_t k>
  class ANSEncoder {
    // The constructor initializes M to be the sum of all Fs. It uses k
    // to determine the proper normalization interval for encoding. It uses
    // b to know how many bytes/bits etc to emit at a time.
    ANSEncoder(const std::vector<uint32_t> &Fs)
      : _F(Fs)
      , _B([&Fs] {
          std::vector<uint32_t> t(Fs.size(), 0);
          std::partial_sum(Fs.begin(), Fs.end() - 1, t.begin() + 1);
          return t;
        })
      , _M(_B.last() + _F.last())
      , _L(k * _M)
    {
      static_assert((b & (b - 1)) == 0, "rANS encoder may only emit powers-of-two for renormalization!");
      assert(_L < (1ULL << 32));
      assert((b * _L) < (1ULL << 32));
    }

    void Encode(size_t symbol) {
      assert(_L <= _state && _state < (b * _L));
      assert(symbol < _F.size());

      // Renormalize
      uint32_t upper_bound = b * k * _F[symbol];
      while (_state >= upper_bound) {
        _state /= b;
      }

      // Encode
      _state = ((_state / _F[symbol]) * _M) + _B[symbol] + (_state % _F[symbol]);
    }

    uint32_t GetState() const { return _state; }

  private:
    uint32_t _state;
    const uint32_t _L;
    const uint32_t _M;

    const std::vector<uint32_t> _F;
    const std::vector<uint32_t> _B;
  };

}  // namespace ans

#endif  // __ANS_ENCODE_H__
