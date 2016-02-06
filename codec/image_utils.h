#ifndef __CODEC_IMAGE_UTILS_H__
#define __CODEC_IMAGE_UTILS_H__

#include "image.h"
#include "pipeline.h"

#include <array>

namespace GenTC {

template <unsigned NumChannels, typename Prec>
std::unique_ptr<std::array<Image<1>, NumChannels> >
SplitImage(const Image<NumChannels, Prec> *img) {
  typedef std::array<Image<1>, NumChannels> ReturnValueType;
  typedef std::unique_ptr<ReturnValueType> ReturnType;

  ReturnValueType result;

  for (uint32_t i = 0; i < NumChannels; ++i) {
    result[i] = Image<1>(img->Width(), img->Height());
  }

  for (size_t j = 0; j < img->Height(); ++j) {
    for (size_t i = 0; i < img->Width(); ++i) {
      auto pixel = std::move(img->GetAt(i, j));
      for (size_t ch = 0; ch < NumChannels; ++ch) {
        result[ch].SetAt(i, j, {{ pixel[ch] }});
      }
    }
  }

  return ReturnType(new ReturnValueType(std::move(result)));  
}

template <unsigned NumChannels, typename Prec>
class ImageSplit
  : public PipelineUnit<Image<NumChannels, Prec>,
                        std::array<Image<1>, NumChannels> > {
 public:
  typedef PipelineUnit<Image<NumChannels, Prec>,
                       std::array<Image<1>, NumChannels> > Base;
  typename Base::ReturnType
  Run(const std::unique_ptr<Image<NumChannels, Prec> > &in) const override {
    return std::move(SplitImage(in.get()));
  }
};

template <unsigned NumChannels, typename Prec>
class PackedImageSplit
  : public PipelineUnit<PackedImage<NumChannels, Prec>,
                        std::array<Image<1>, NumChannels> > {
 public:
  typedef PipelineUnit<PackedImage<NumChannels, Prec>,
                       std::array<Image<1>, NumChannels> > Base;
  typename Base::ReturnType
  Run(const std::unique_ptr<PackedImage<NumChannels, Prec> > &in) const override {
    return std::move(SplitImage(in.get()));
  }
};

typedef PackedImageSplit<3, RGB> RGBSplitter;
typedef PackedImageSplit<4, RGBA> RGBASplitter;
typedef PackedImageSplit<3, RGB565> RGB565Splitter;

}  // namespace GenTC

#endif  //  __CODEC_IMAGE_UTILS_H__
