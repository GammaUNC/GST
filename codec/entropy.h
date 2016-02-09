#ifndef __TCAR_ENTROPY_H__
#define __TCAR_ENTROPY_H__

#include "pipeline.h"

#include <cassert>
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
  static std::unique_ptr<Base> New(int row_length, size_t block_length) {
    return std::unique_ptr<Base>(new RearrangeStream<T>(row_length, block_length));
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    assert((in->size() % _img_length) == 0);
    assert(((in->size() / _img_length) % _block_length) == 0);

    std::vector<T> *result = new std::vector<T>;
    result->reserve(in->size());

    size_t j = 0;
    while (j < result->size()) {
      for (size_t i = 0; i < _img_length; i += _block_length) {
        for (size_t y = 0; y < _block_length; ++y) {
          for (size_t x = 0; x < _block_length; ++x) {
            size_t idx = (j + y) * _img_length + i + x;
            result->push_back(in->at(idx));
          }
        }
      }
      j += _img_length * _block_length;
    }

    return std::move(std::unique_ptr<std::vector<T> >(result));
  }

 private:
  size_t _img_length;
  size_t _block_length;

  RearrangeStream<T>(size_t row_length, size_t block_length)
    : _img_length(row_length)
    , _block_length(block_length)
  {
    assert((_img_length % block_length) == 0);
  }
};

}  // namespace GenTC

#endif  // __TCAR_ENTROPY_H__
