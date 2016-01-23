#ifndef __ANS_ENCODE_H__
#define __ANS_ENCODE_H__

#include <cstdint>
#include <memory>
#include <vector>
#include "bits.h"

namespace ans {

  enum EType {
    eType_rANS,
    eType_tANS,

    kNumTypes
  };

  struct Options {
    uint32_t b;
    uint32_t k;
    EType type;
  };

  class Encoder {
   public:
     virtual ~Encoder() { }
     virtual void Encode(uint32_t symbol, BitWriter *w) = 0;
     virtual uint32_t GetState() const = 0;
     static std::unique_ptr<Encoder> Create(
       const std::vector<uint32_t> &Fs, const Options &opts);
  protected:
    Encoder() { }
  };

  class Decoder {
   public:
     virtual ~Decoder() { }
     virtual uint32_t Decode(BitReader *r) = 0;
     virtual uint32_t GetState() const = 0;
     static std::unique_ptr<Decoder> Create(
       uint32_t state, const std::vector<uint32_t> &Fs, const Options &opts);
  protected:
    Decoder() { }
  };
}  // namespace ans

#endif  // __ANS_ENCODE_H__
