#ifndef __ANS_ENCODE_H__
#define __ANS_ENCODE_H__

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include "bits.h"

namespace ans {

  enum EType {
    eType_rANS,
    eType_tANS,

    kNumTypes
  };

  // Various resolution knobs for the encoder:
  // b  - the renormalization factor. 2 means bits, 256 means bytes
  // k  - the state resolution factor. This is just a tunable variable
  //      keeping it at "one" quite fine>
  // M  - the common denominator for symbol probabilities. The larger
  //      this is, the more accurate we can predict symbols.
  // Fs - The frequency of each symbol. This implies that the probability
  //      of encountering each symbol i is Fs[i] / Sum(Fs).
  //
  // Limitations:
  //   - Since our encoders and decoders use 32-bit state, the value of
  //   b*k*M must be less than (1 << 32).
  //
  // Optimizations:
  //   - If we have that b >= L = k*M, then our normalization loop
  //     only needs to execute once.
  struct Options {
    EType type = eType_rANS;
    uint32_t b = 0;
    uint32_t k = 0;
    uint32_t M = 0;
    std::vector<uint32_t> Fs;
  };

  class Encoder {
   public:
     virtual ~Encoder() { }
     virtual void Encode(uint32_t symbol, BitWriter *w) = 0;
     virtual uint32_t GetState() const = 0;
     static std::unique_ptr<Encoder> Create(const Options &opts);
  protected:
    Encoder() { }
  };

  class Decoder {
   public:
     virtual ~Decoder() { }
     virtual uint32_t Decode(BitReader *r) = 0;
     virtual uint32_t GetState() const = 0;
     static std::unique_ptr<Decoder> Create(uint32_t state, const Options &opts);
  protected:
    Decoder() { }
  };

  std::vector<uint8_t> EncodeInterleaved(const std::vector<uint8_t> &symbols,
                                         const Options &opts, size_t num_streams);

  std::vector<uint8_t> DecodeInterleaved(const std::vector<uint8_t> &data,
                                         size_t num_symbols,
                                         const Options &opts, size_t num_streams);
}  // namespace ans

#endif  // __ANS_ENCODE_H__
