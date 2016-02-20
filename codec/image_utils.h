#ifndef __TCAR_IMAGE_UTILS_H__
#define __TCAR_IMAGE_UTILS_H__

#include "image.h"
#include "pipeline.h"

#include <array>

namespace GenTC {

template <typename T>
class ImageSplit { };

template <typename T1, typename T2, typename T3>
class ImageSplit<std::tuple<T1, T2, T3> >
  : public PipelineUnit<Image<std::tuple<T1, T2, T3> >,
                        std::array<Image<Alpha>, 3> > {
  typedef std::tuple<T1, T2, T3> PixelTy;
  static_assert(PixelTraits::NumChannels<PixelTy>::value == 3,
                "Pixel3 has three channels!");
  static const size_t kNumChannels = PixelTraits::NumChannels<PixelTy>::value;
 public:
  typedef PipelineUnit<Image<PixelTy>, std::array<Image<Alpha>, kNumChannels> > Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ImageSplit<PixelTy>());
  }

  typename Base::ReturnType Run(const std::unique_ptr<Image<PixelTy> > &in) const override {
    typedef std::array<Image<Alpha>, kNumChannels> ReturnValueType;
    typedef std::unique_ptr<ReturnValueType> ReturnType;
    ReturnValueType *result = new ReturnValueType;

    for (size_t i = 0; i < kNumChannels; ++i) {
      (*result)[i] = Image<Alpha>(in->Width(), in->Height());
    }

    for (size_t j = 0; j < in->Height(); ++j) {
      for (size_t i = 0; i < in->Width(); ++i) {
        PixelTy pixel = in->GetAt(i, j);

        assert(PixelTraits::Max<T1>::value < 256);
        assert(PixelTraits::Min<T1>::value >= 0);
        assert(PixelTraits::Max<T2>::value < 256);
        assert(PixelTraits::Min<T2>::value >= 0);
        assert(PixelTraits::Max<T3>::value < 256);
        assert(PixelTraits::Min<T3>::value >= 0);

        (*result)[0].SetAt(i, j, static_cast<Alpha>(std::get<0>(pixel)));
        (*result)[1].SetAt(i, j, static_cast<Alpha>(std::get<1>(pixel)));
        (*result)[2].SetAt(i, j, static_cast<Alpha>(std::get<2>(pixel)));
      }
    }

    return std::move(ReturnType(result));
  }
};

template <typename T1, typename T2, typename T3, typename T4>
class ImageSplit<std::tuple<T1, T2, T3, T4> >
  : public PipelineUnit<Image<std::tuple<T1, T2, T3, T4> >,
                        std::array<Image<Alpha>, 4> > {
  typedef std::tuple<T1, T2, T3, T4> PixelTy;
  static_assert(PixelTraits::NumChannels<PixelTy>::value == 4,
                "Pixel4 has four channels!");
  static const size_t kNumChannels = PixelTraits::NumChannels<PixelTy>::value;
 public:
  typedef PipelineUnit<Image<PixelTy>, std::array<Image<Alpha>, kNumChannels> > Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ImageSplit<PixelTy>());
  }

  typename Base::ReturnType Run(const std::unique_ptr<Image<PixelTy> > &in) const override {
    typedef std::array<Image<Alpha>, kNumChannels> ReturnValueType;
    typedef std::unique_ptr<ReturnValueType> ReturnType;
    ReturnValueType *result = new ReturnValueType;

    for (size_t i = 0; i < kNumChannels; ++i) {
      (*result)[i] = Image<Alpha>(in->Width(), in->Height());
    }

    for (size_t j = 0; j < in->Height(); ++j) {
      for (size_t i = 0; i < in->Width(); ++i) {
        PixelTy pixel = in->GetAt(i, j);

        assert(PixelTraits::Max<T1>::value < 256);
        assert(PixelTraits::Min<T1>::value >= 0);
        assert(PixelTraits::Max<T2>::value < 256);
        assert(PixelTraits::Min<T2>::value >= 0);
        assert(PixelTraits::Max<T3>::value < 256);
        assert(PixelTraits::Min<T3>::value >= 0);
        assert(PixelTraits::Max<T4>::value < 256);
        assert(PixelTraits::Min<T4>::value >= 0);

        (*result)[0].SetAt(i, j, static_cast<Alpha>(std::get<0>(pixel)));
        (*result)[1].SetAt(i, j, static_cast<Alpha>(std::get<1>(pixel)));
        (*result)[2].SetAt(i, j, static_cast<Alpha>(std::get<2>(pixel)));
        (*result)[3].SetAt(i, j, static_cast<Alpha>(std::get<3>(pixel)));
      }
    }

    return std::move(ReturnType(result));
  }
};

typedef ImageSplit<RGB> RGBSplitter;
typedef ImageSplit<RGBA> RGBASplitter;
typedef ImageSplit<RGB> YCrCbSplitter;

template<typename T>
class Linearize : public PipelineUnit<Image<T>, std::vector<T> > {
 public:
  typedef PipelineUnit<Image<T>, std::vector<T> > Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new Linearize<T>); }
  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    assert(in->Width() > 0);
    assert(in->Height() > 0);

    std::vector<T> *result = new std::vector<T>(in->GetPixels());
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

template<typename T>
class WriteGrayscale : public Sink<Image<T> > {
  static_assert(PixelTraits::NumChannels<T>::value == 1,
                "Only single channel images can be output as grayscale");
 public:
  typedef Image<T> InputImage;
  typedef Sink<InputImage> SinkBase;
  static std::unique_ptr<typename SinkBase::Base> New(const char *fn) {
    return std::unique_ptr<typename SinkBase::Base>(new WriteGrayscale<T>(fn));
  }

  virtual void Finish(const std::unique_ptr<Image<T> > &in) const override {
    std::vector<uint8_t> pixels(in->GetPixels().size(), 0);

    T max = PixelTraits::Max<T>::value;
    assert(PixelTraits::Max<T>::value > 0);
    uint64_t range = static_cast<uint64_t>(max);

    T min = PixelTraits::Min<T>::value;
    assert((0 - min) >= 0);
    uint64_t offset = static_cast<uint64_t>(0 - min);

    if (PixelTraits::IsSigned<T>::value) {
      range += offset;
    }

    auto dst = pixels.begin();
    for (const auto &pixel : in->GetPixels()) {
      double pd = static_cast<double>(pixel) + static_cast<double>(offset);
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

template<typename T>
class InspectGrayscale : public PipelineUnit<Image<T>, Image<T> > {
 public:
  typedef Image<T> ImageTy;
  typedef PipelineUnit<ImageTy, ImageTy> Base;
  static std::unique_ptr<Base> New(const char *fn) {
    return std::unique_ptr<Base>(new InspectGrayscale<T>(fn));
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    auto writer = Pipeline<ImageTy, int>::Create(WriteGrayscale<T>::New(_fname));
    writer->Run(in);
    // !FIXME! perhaps we should use a std::move here...
    return std::move(std::unique_ptr<ImageTy>(new ImageTy(*(in.get()))));
  }

 private:
  InspectGrayscale<T>(const char *filename) : Base(), _fname(filename) { }
  const char *const _fname;
};

}  // namespace GenTC

#endif  //  __TCAR_IMAGE_UTILS_H__
