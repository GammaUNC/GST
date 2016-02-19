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

template <unsigned NumChannels, typename T, typename Prec>
std::unique_ptr<std::array<Image<1, T, Alpha>, NumChannels> >
SplitImage(const Image<NumChannels, T, Prec> *img) {
  typedef std::array<Image<1, T, Alpha>, NumChannels> ReturnValueType;
  typedef std::unique_ptr<ReturnValueType> ReturnType;

#ifndef NDEBUG
  for (size_t i = 0; i < NumChannels; ++i) {
    assert(img->Precision(i) <= 8);
  }
#endif

  ReturnValueType result;

  for (uint32_t i = 0; i < NumChannels; ++i) {
    result[i] = Image<1, T, Alpha>(img->Width(), img->Height());
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

template <unsigned NumChannels, typename T, typename Prec>
class ImageSplit
  : public PipelineUnit<Image<NumChannels, T, Prec>,
                        std::array<Image<1, T, Alpha>, NumChannels> > {
 public:
  typedef PipelineUnit<Image<NumChannels, T, Prec>,
                       std::array<Image<1, T, Alpha>, NumChannels> > Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ImageSplit<NumChannels, T, Prec>());
  }

  typename Base::ReturnType
  Run(const std::unique_ptr<Image<NumChannels, T, Prec> > &in) const override {
    return std::move(SplitImage(in.get()));
  }
};

template <unsigned NumChannels, typename T, typename Prec>
class PackedImageSplit
  : public PipelineUnit<PackedImage<NumChannels, T, Prec>,
                        std::array<Image<1, T, Alpha>, NumChannels> > {
 public:
  typedef PipelineUnit<PackedImage<NumChannels, T, Prec>,
                       std::array<Image<1, T, Alpha>, NumChannels> > Base;

  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new PackedImageSplit<NumChannels, T, Prec>());
  }

  typename Base::ReturnType
  Run(const std::unique_ptr<PackedImage<NumChannels, T, Prec> > &in) const override {
    return std::move(SplitImage(in.get()));
  }
};

typedef PackedImageSplit<3, uint8_t, RGB> RGBSplitter;
typedef PackedImageSplit<4, uint8_t, RGBA> RGBASplitter;
typedef PackedImageSplit<3, uint8_t, RGB565> RGB565Splitter;

typedef ImageSplit<3, uint8_t, RGB> YCrCbSplitter;

template<typename T, typename Prec>
class Linearize : public PipelineUnit<Image<1, T, Prec>, std::vector<T> > {
 public:
  typedef PipelineUnit<Image<1, T, Prec>, std::vector<T> > Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new Linearize<T, Prec>); }
  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    assert(in->Width() > 0);
    assert(in->Height() > 0);

    std::vector<T> *result = new std::vector<T>;
    result->reserve(in->Width() * in->Height() * Image<1, T, Prec>::kNumChannels);

    for (size_t j = 0; j < in->Height(); ++j) {
      for (size_t i = 0; i < in->Width(); ++i) {
        auto pixel = in->GetAt(i, j);
        for (auto ch : pixel) {
          result->push_back(ch);
        }
      }
    }

    // This only works on single channel images, so the following
    // should hold true..
    assert(result->size() == in->Height() * in->Width());

    return std::move(std::unique_ptr<std::vector<T> >(result));
  }
};

class DropAlpha : public PipelineUnit<RGBAImage, RGBImage> {
 public:
  typedef PipelineUnit<RGBAImage, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new DropAlpha); }
  Base::ReturnType Run(const Base::ArgType &in) const override;
};

extern void WriteAlphaImage(const std::string &fname, size_t w, size_t h,
                            std::vector<uint8_t> &&pixels);

template<typename T, typename Prec>
class WriteGrayscale : public Sink<Image<1, T, Prec> > {
  static_assert(std::is_integral<T>::value, "Only operates on integral values");
 public:
  typedef Image<1, T, Prec> InputImage;
  typedef Sink<InputImage> SinkBase;
  static std::unique_ptr<typename SinkBase::Base> New(const char *fn) {
    return std::unique_ptr<typename SinkBase::Base>(new WriteGrayscale<T, Prec>(fn));
  }

  virtual void Finish(const std::unique_ptr<Image<1, T, Prec> > &in) const override {
    std::vector<uint8_t> pixels(in->GetPixels().size(), 0);

    assert(std::numeric_limits<T>::max() > 0);
    uint64_t range = static_cast<uint64_t>(std::numeric_limits<T>::max());

    assert((0 - std::numeric_limits<T>::min()) >= 0);
    uint64_t offset = static_cast<uint64_t>(0 - std::numeric_limits<T>::min());

    if (std::is_signed<T>::value) {
      range += offset;
    }

    auto dst = pixels.begin();
    for (const auto &pixel : in->GetPixels()) {
      T p = pixel[0];
      double pd = static_cast<double>(p) + static_cast<double>(offset);
      pd /= static_cast<double>(range);
      pd = (pd * 255.0) + 0.5;

      *dst = static_cast<uint8_t>(pd);
      dst++;
    }

    WriteAlphaImage(_filename, in->Width(), in->Height(), std::move(pixels));
  }

 private:
  WriteGrayscale(const char *filename) : Sink<InputImage>(), _filename(filename) { }
  std::string _filename;
};

template<typename T, typename Prec>
class InspectGrayscale : public PipelineUnit<Image<1, T, Prec>, Image<1, T, Prec> > {
  static_assert(std::is_integral<T>::value, "Only operates on integral values");
 public:
  typedef Image<1, T, Prec> ImageTy;
  typedef PipelineUnit<ImageTy, ImageTy> Base;
  static std::unique_ptr<Base> New(const char *fn) {
    return std::unique_ptr<Base>(new InspectGrayscale<T, Prec>(fn));
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    auto writer = Pipeline<ImageTy, int>::Create(WriteGrayscale<T, Prec>::New(_fname));
    writer->Run(in);
    // !FIXME! perhaps we should use a std::move here...
    return std::move(std::unique_ptr<ImageTy>(new ImageTy(*(in.get()))));
  }

 private:
  InspectGrayscale<T, Prec>(const char *filename) : Base(), _fname(filename) { }
  const char *const _fname;
};

}  // namespace GenTC

#endif  //  __TCAR_IMAGE_UTILS_H__
