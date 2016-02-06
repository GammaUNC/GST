#ifndef __CODEC_IMAGE_UTILS_H__
#define __CODEC_IMAGE_UTILS_H__

#include "image.h"
#include "pipeline.h"

#include <array>

namespace GenTC {

template <unsigned NumChannels, typename Prec>
class ImageSplit
  : public PipelineUnit<Image<NumChannels, Prec>,
                        std::array<Image<1>, NumChannels> > {
 public:
  typedef PipelineUnit<Image<NumChannels, Prec>,
                       std::array<Image<1>, NumChannels> > Base;
  typename Base::ReturnType
  Run(const std::unique_ptr<Image<NumChannels, Prec> > &in) const override {
    typename Base::ReturnValueType result;

    for (uint32_t i = 0; i < NumChannels; ++i) {
      result[i] = Image<1>(in->Width(), in->Height());
    }

    for (size_t j = 0; j < in->Height(); ++j) {
      for (size_t i = 0; i < in->Width(); ++i) {
        std::array<uint32_t, NumChannels> pixel = std::move(in->GetAt(i, j));
        for (size_t ch = 0; ch < NumChannels; ++ch) {
          result[ch].SetAt(i, j, {{ pixel[ch] }});
        }
      }
    }

    return typename Base::ReturnType(new typename Base::ReturnValueType(std::move(result)));
  };
};

}  // namespace GenTC

#endif  //  __CODEC_IMAGE_UTILS_H__
