#include <algorithm>
#include <random>
#include <numeric>

#include "ans.h"
#include "ans_utils.h"

namespace ans {

// rANS decode.
class rANS_Decoder : public Decoder {
public:
  // The constructor initializes M to be the sum of all Fs. It uses k
  // to determine the proper normalization interval for encoding. It uses
  // b to know how many bytes/bits etc to emit at a time.
  rANS_Decoder(uint32_t state, const std::vector<uint32_t> &Fs, uint32_t b, uint32_t k)
    : _F(Fs)
    , _B(CumulativeSum(Fs))
    , _M(_B.back() + _F.back())
    , _k(k)
    , _b(b)
    , _log_b(IntLog2(b))
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

////////////////////////////////////////////////////////////////////////////////
//
// tANS
//

class tANS_Decoder : public Decoder {
public:
  // The constructor initializes M to be the sum of all Fs. It uses k
  // to determine the proper normalization interval for encoding. It uses
  // b to know how many bytes/bits etc to emit at a time.
  tANS_Decoder(uint32_t state, const std::vector<uint32_t> &Fs, uint32_t b, uint32_t k);

  virtual uint32_t Decode(BitReader *r) override;
  virtual uint32_t GetState() const override { return _state; }

private:
  const std::vector<uint32_t> _F;
  const std::vector<uint32_t> _B;

  const uint32_t _M;
  const uint32_t _b;
  const uint32_t _k;
  const int _log_b;

  // See tANS encoder for more details.
  //
  // If our encoding step is thus:
  // x' = (x / Fs) * M + _enc_table[Bs + (x % Fs)];
  //
  // Then our decoding step is:
  // s = _dec_table[x' % M]
  // offset = _offset_table[x' % M];
  // x = Fs * (x / M) + offset
  const std::vector<uint32_t> _dec_table;
  const std::vector<uint32_t> _offset_table;

  uint32_t _state;
};

static std::vector<uint32_t> BuildDecTable(const std::vector<uint32_t> &Fs, const uint32_t M) {
  // Collect symbols...
  std::vector<uint32_t> dec_table = std::vector<uint32_t>(M);

  // Let's just start with bit reversal...
  for (uint32_t i = 0, sidx = 0; i < Fs.size(); ++i) {
    for (uint32_t j = 0; j < Fs[i]; ++j, ++sidx) {
      // dec_table[BitReverse(sidx, IntLog2(M))] = i;
      dec_table[sidx] = i;
    }
  }

  unsigned seed = 0;
  std::shuffle(dec_table.begin(), dec_table.end(), std::default_random_engine(seed));

  return std::move(dec_table);
}

static std::vector<uint32_t> BuildOffsetTable(const std::vector<uint32_t> &Fs,
                                              const std::vector<uint32_t> &symbols, const uint32_t M) {
  std::vector<uint32_t> offset_table;
  offset_table.reserve(M);

  std::vector<uint32_t> idx(Fs.size(), 0);
  for (auto symbol : symbols) {
    offset_table.push_back(idx[symbol]++);
  }

  return std::move(offset_table);
}

tANS_Decoder::tANS_Decoder(uint32_t state, const std::vector<uint32_t> &Fs, uint32_t b, uint32_t k)
  : _F(Fs)
  , _B(CumulativeSum(Fs))
  , _M(_B.back() + _F.back())
  , _b(b)
  , _k(k)
  , _log_b(IntLog2(b))
  , _dec_table(std::move(BuildDecTable(Fs, _M)))
  , _offset_table(std::move(BuildOffsetTable(Fs, _dec_table, _M)))
  , _state(state)
{
  assert((b & (_b - 1)) == 0 || "rANS encoder may only emit powers-of-two for renormalization!");
  assert((k & (_k - 1)) == 0 || "rANS encoder must have power-of-two multiple of precision!");
  assert(_k * _M < (1ULL << 32));
  assert((b * _k * _M) < (1ULL << 32));
}

uint32_t tANS_Decoder::Decode(BitReader *r) {
  assert(_k * _M <= _state && _state < (_b * _k * _M));

  // Decode
  uint32_t symbol = _dec_table[_state % _M];
  _state = (_state / _M) * _F[symbol] + _offset_table[_state % _M];

  // Renormalize
  while (_state < _k * _M) {
    int new_bits = r->ReadBits(_log_b);
    assert(new_bits < (1 << _log_b));
    _state <<= _log_b;
    _state |= new_bits;
  }

  return symbol;
}

////////////////////////////////////////////////////////////////////////////////
//
// ANS Factory
//

std::unique_ptr<Decoder> Decoder::Create(uint32_t state, const std::vector<uint32_t> &Fs, const Options &opts) {
  std::unique_ptr<Decoder> dec;
  switch (opts.type) {
  case eType_rANS:
    dec.reset(new rANS_Decoder(state, Fs, opts.b, opts.k));
    break;
  case eType_tANS:
    dec.reset(new tANS_Decoder(state, Fs, opts.b, opts.k));
    break;

  default:
    assert(!"Unknown type!");
    break;
  }

  return std::move(dec);
}

}  // namespace ans
