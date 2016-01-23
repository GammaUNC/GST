#include <vector>
#include <numeric>

#include "bits.h"
#include "ans.h"

namespace ans {

class rANS_Encoder : public Encoder {
public:
  // The constructor initializes M to be the sum of all Fs. It uses k
  // to determine the proper normalization interval for encoding. It uses
  // b to know how many bytes/bits etc to emit at a time.
  rANS_Encoder(const std::vector<uint32_t> &Fs, uint32_t b, uint32_t k)
    : Encoder()
    , _F(Fs)
    , _B([&Fs] {
      std::vector<uint32_t> t(Fs.size(), 0);
      std::partial_sum(Fs.begin(), Fs.end() - 1, t.begin() + 1);
      return t;
    }())
    , _M(_B.back() + _F.back())
    , _b(b)
    , _k(k)
    , _log_b([b] {
      uint32_t bb = b;
      if (bb == 0) {
        return 0;
      }

      int l = 0;
      while (bb >>= 1) {
        l++;
      }
      return l;
    }())
    , _state(_k * _M)
  {
    assert((b & (b - 1)) == 0 || "rANS encoder may only emit powers-of-two for renormalization!");
    assert((k & (k - 1)) == 0 || "rANS encoder must have power-of-two multiple of precision!");
    assert(_k * _M < (1ULL << 32));
    assert((b * _k * _M) < (1ULL << 32));
  }

  virtual void Encode(uint32_t symbol, BitWriter *w) override {
    assert(_k * _M <= _state && _state < _b * _k * _M);
    assert(symbol < _F.size());

    // Renormalize
    uint32_t upper_bound = _b * _k * _F[symbol];
    while (_state >= upper_bound) {
      w->WriteBits(_state & (_b - 1), _log_b);
      _state /= _b;
    }

    // Encode
    _state = ((_state / _F[symbol]) * _M) + _B[symbol] + (_state % _F[symbol]);
  }

  virtual uint32_t GetState() const override { return _state; }

private:
  const std::vector<uint32_t> _F;
  const std::vector<uint32_t> _B;

  const uint32_t _M;
  const uint32_t _b;
  const uint32_t _k;
  const int _log_b;

  uint32_t _state;
};

std::unique_ptr<Encoder> Encoder::Create(const std::vector<uint32_t> &Fs, const Options &opts) {
  std::unique_ptr<Encoder> enc;
  switch (opts.type) {
    case eType_rANS:
      enc.reset(new rANS_Encoder(Fs, opts.b, opts.k));
      break;
  }

  return std::move(enc);
}

}  // namespace ans