#ifndef __TCAR_ENTROPY_H__
#define __TCAR_ENTROPY_H__

#include "pipeline.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace GenTC {

// Rearrange the stream of values such that the current stream
// is treated as a matrix with "row_length" number of columns.
// The values are rearranged such that blocks with 'block_length'
// number of columns are linearized in order and placed on the stream
template<typename T>
class RearrangeStream : public PipelineUnit<std::vector<T>, std::vector<T> > {
 public:
  typedef PipelineUnit<std::vector<T>, std::vector<T> > Base;
  static std::unique_ptr<Base> New(size_t row_length, size_t block_length) {
    return std::unique_ptr<Base>(new RearrangeStream<T>(row_length, block_length));
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    // Make sure we have some data
    assert(in->size() > 0);

    // Make sure our data is a multiple of the row length
    assert((in->size() % _row_length) == 0);

    // Make sure the number of rows is a multiple of the block length
    assert(((in->size() / _row_length) % _block_length) == 0);

    std::vector<T> *result = new std::vector<T>;
    result->reserve(in->size());

    size_t j = 0;
    while (j < in->size()) {
      for (size_t i = 0; i < _row_length; i += _block_length) {
        for (size_t y = 0; y < _block_length; ++y) {
          for (size_t x = 0; x < _block_length; ++x) {
            size_t idx = j + y * _row_length + i + x;
            result->push_back(in->at(idx));
          }
        }
      }
      j += _row_length * _block_length;
    }

    return std::move(std::unique_ptr<std::vector<T> >(result));
  }

 private:
  size_t _row_length;
  size_t _block_length;

  RearrangeStream<T>(size_t row_length, size_t block_length)
    : _row_length(row_length)
    , _block_length(block_length)
  {
    assert((_row_length % _block_length) == 0);
  }
};

template<typename From, typename To>
class ReducePrecision : public PipelineUnit<std::vector<From>, std::vector<To> > {
  static_assert(std::is_integral<To>::value, "Only operates on integral values");
  static_assert(std::is_integral<From>::value, "Only operates on integral values");
public:
  typedef PipelineUnit<std::vector<From>, std::vector<To> > Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ReducePrecision<From, To>);
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    std::vector<To> *result = new std::vector<To>;
    result->reserve(in->size());
    for (size_t i = 0; i < in->size(); ++i) {
      assert(in->at(i) < (1ULL << (sizeof(To) * 8)));
      result->push_back(static_cast<To>(in->at(i)));
    }
    return std::move(std::unique_ptr<std::vector<To> >(result));
  }
};

class ShortEncoder {
 public:
  typedef PipelineUnit<std::vector<int16_t>, std::vector<uint8_t> > EncodeUnit;
  typedef PipelineUnit<std::vector<uint8_t>, std::vector<int16_t> > DecodeUnit;

  static std::unique_ptr<EncodeUnit> Encoder(size_t symbols_per_thread) {
    return std::unique_ptr<EncodeUnit>(new Encode(symbols_per_thread));
  }

  static std::unique_ptr<DecodeUnit> Decoder(size_t symbols_per_thread) {
    return std::unique_ptr<DecodeUnit>(new Decode(symbols_per_thread));
  }

 private:
  class Encode : public EncodeUnit {
   public:
    Encode(size_t spt): EncodeUnit(), _symbols_per_thread(spt) { }
    EncodeUnit::ReturnType Run(const EncodeUnit::ArgType &in) const override;

   private:
    const size_t _symbols_per_thread;
  };

  class Decode : public DecodeUnit {
   public:
    Decode(size_t spt): DecodeUnit(), _symbols_per_thread(spt) { }
    DecodeUnit::ReturnType Run(const DecodeUnit::ArgType &in) const override;

   private:
    const size_t _symbols_per_thread;
  };
};

class ByteEncoder {
 public:
  typedef PipelineUnit<std::vector<uint8_t>, std::vector<uint8_t> > Base;

  static std::unique_ptr<Base> Encoder(size_t symbols_per_thread) {
    return std::unique_ptr<Base>(new EncodeBytes(symbols_per_thread));
  }

  static std::unique_ptr<Base> Decoder(size_t symbols_per_thread) {
    return std::unique_ptr<Base>(new DecodeBytes(symbols_per_thread));
  }

 private:
  class EncodeBytes : public Base {
   public:
    EncodeBytes(size_t spt) :Base(), _symbols_per_thread(spt) { }
    Base::ReturnType Run(const Base::ArgType &in) const override;
    
   private:
    const size_t _symbols_per_thread;
  };

  class DecodeBytes : public Base {
   public:
    DecodeBytes(size_t spt) :Base(), _symbols_per_thread(spt) { }
    Base::ReturnType Run(const Base::ArgType &in) const override;

   private:
    const size_t _symbols_per_thread;
  };
};

}  // namespace GenTC

#endif  // __TCAR_ENTROPY_H__
