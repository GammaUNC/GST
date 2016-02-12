#ifndef __TCAR_IMAGE_UTILS_H__
#define __TCAR_IMAGE_UTILS_H__

#include "image.h"
#include "pipeline.h"

#include <array>

namespace GenTC {

template<unsigned NumChannels, typename Prec>
class ToUnpackedImage
  : public PipelineUnit < PackedImage<NumChannels, Prec>, Image<NumChannels, Prec> > {
public:
  typedef PipelineUnit < PackedImage<NumChannels, Prec>, Image<NumChannels, Prec> > Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ToUnpackedImage<NumChannels, Prec>);
  }

  typename Base::ReturnType
  Run(const typename Base::ArgType &in) const override {
    return std::move(typename Base::ReturnType(new Image<NumChannels, Prec>(*(in.get()))));
  }
};

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
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ImageSplit<NumChannels, Prec>());
  }

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

  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new PackedImageSplit<NumChannels, Prec>());
  }

  typename Base::ReturnType
  Run(const std::unique_ptr<PackedImage<NumChannels, Prec> > &in) const override {
    return std::move(SplitImage(in.get()));
  }
};

typedef PackedImageSplit<3, RGB> RGBSplitter;
typedef PackedImageSplit<4, RGBA> RGBASplitter;
typedef PackedImageSplit<3, RGB565> RGB565Splitter;

typedef ImageSplit<3, RGB> YCrCbSplitter;

}  // namespace GenTC

#endif  //  __TCAR_IMAGE_UTILS_H__
