#include <numeric>

#include "ans.h"

namespace ans {

// rANS decode.
class rANS_Decoder : public Decoder {
public:
  // The constructor initializes M to be the sum of all Fs. It uses k
  // to determine the proper normalization interval for encoding. It uses
  // b to know how many bytes/bits etc to emit at a time.
  rANS_Decoder(uint32_t state, const std::vector<uint32_t> &Fs, uint32_t b, uint32_t k)
    : _F(Fs)
    , _B([&Fs] {
      std::vector<uint32_t> t(Fs.size(), 0);
      std::partial_sum(Fs.begin(), Fs.end() - 1, t.begin() + 1);
      return t;
    }())
    , _M(_B.back() + _F.back())
    , _k(k)
    , _b(b)
    , _log_b([b] {
      uint32_t bb = b;
      if (bb == 0) { return 0; }

      int l = 0;
      while (bb >>= 1) { l++; }
      return l;
    }())
    , _state(state)
  {
    assert((b & (_b - 1)) == 0 || "rANS encoder may only emit powers-of-two for renormalization!");
    assert((k & (_k - 1)) == 0 || "rANS encoder must have power-of-two multiple of precision!");
    assert(_k * _M < (1ULL << 32));
    assert((b * _k * _M) < (1ULL << 32));
  }

  virtual uint32_t Decode(BitReader *r) override {
    assert(_k * _M <= _state && _state < (_b * _k * _M));

    // Decode
    uint32_t symbol = FindSymbol(_state % _M);
    _state = (_state / _M) * _F[symbol] - _B[symbol] + (_state % _M);

    // Renormalize
    while (_state < _k * _M) {
      int new_bits = r->ReadBits(_log_b);
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
  const uint32_t _k;
  const uint32_t _b;
  const uint32_t _log_b;

  uint32_t _state;

  uint32_t FindSymbol(uint32_t x) {
    uint32_t low = 0;
    uint32_t high = static_cast<uint32_t>(_B.size());

    // !SPEED! it might be better to unroll this table for small M...
    // Search for symbol in Bs...
    for (;;) {
      uint32_t midpoint = (high + low) >> 1;
      assert(midpoint >= 0);
      assert(midpoint < _F.size());

      if (_B[midpoint] <= x) {
        if (midpoint >= (_B.size() - 1) || _B[midpoint + 1] > x) {
          return midpoint;
        }

        low = midpoint;
      }
      else {
        if (midpoint == 0 || _B[midpoint - 1] <= x) {
          midpoint = midpoint - 1;
          return midpoint;
        }

        high = midpoint;
      }
    }
  }
};

std::unique_ptr<Decoder> Decoder::Create(uint32_t state, const std::vector<uint32_t> &Fs, const Options &opts) {
  std::unique_ptr<Decoder> dec;
  switch (opts.type) {
  case eType_rANS:
    dec.reset(new rANS_Decoder(state, Fs, opts.b, opts.k));
    break;
  }

  return std::move(dec);
}

}  // namespace ans