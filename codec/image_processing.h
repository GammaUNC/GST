#ifndef __TCAR_IMAGE_PROCESSING_H__
#define __TCAR_IMAGE_PROCESSING_H__

#include "pipeline.h"
#include "image.h"

#include <array>

namespace GenTC {

class RGBtoYCrCb: public PipelineUnit<RGBImage, YCbCrImage> {
 public:
  typedef PipelineUnit<RGBImage, YCbCrImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new RGBtoYCrCb); }
  Base::ReturnType Run(const Base::ArgType &) const override;
};

class YCrCbtoRGB: public PipelineUnit<YCbCrImage, RGBImage> {
 public:
  typedef PipelineUnit<YCbCrImage, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new YCrCbtoRGB); }
  Base::ReturnType Run(const Base::ArgType &) const override;
};

class Expand565
: public PipelineUnit<RGB565Image, RGBImage> {
 public:
  typedef PipelineUnit<RGB565Image, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new Expand565); }
  std::unique_ptr<RGBImage> Run(const std::unique_ptr<RGB565Image> &) const override;
};

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

template<typename T, typename Prec>
class Quantize8x8
  : public PipelineUnit < Image<1, T, Prec>, Image<1, T, Prec> > {
 public:
   typedef Image<1, T, Prec> ImageType;
   typedef PipelineUnit<ImageType, ImageType> Base;

   static std::unique_ptr<Base> QuantizeJPEGLuma() {
     return std::unique_ptr<Base>(new Quantizer(eQuantizeType_JPEGLuma));
   }

   static std::unique_ptr<Base> QuantizeJPEGChroma() {
     return std::unique_ptr<Base>(new Quantizer(eQuantizeType_JPEGChroma));
   }

   static std::unique_ptr<Base> DequantizeJPEGLuma() {
     return std::unique_ptr<Base>(new Dequantizer(eQuantizeType_JPEGLuma));
   }

   static std::unique_ptr<Base> DequantizeJPEGChroma() {
     return std::unique_ptr<Base>(new Dequantizer(eQuantizeType_JPEGChroma));
   }

 protected:
   enum QuantizeType {
     eQuantizeType_JPEGLuma,
     eQuantizeType_JPEGChroma
   };

   class Quantizer : public Quantize8x8 < T, Prec > {
   public:
     Quantizer(QuantizeType ty) : Quantize8x8<T, Prec>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) const override {
       ImageType *result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (size_t j = 0; j < in->Height(); j += 8) {
         for (size_t i = 0; i < in->Width(); i += 8) {
           for (size_t y = 0; y < 8; ++y) {
             for (size_t x = 0; x < 8; ++x) {
               auto pixel = in->GetAt(i + x, j + y);
               pixel[0] /= static_cast<T>(_coeffs[y * 8 + x]);
               result->SetAt(i + x, j + y, std::move(pixel));
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   class Dequantizer : public Quantize8x8 < T, Prec > {
   public:
     Dequantizer(QuantizeType ty) : Quantize8x8<T, Prec>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) const override {
       ImageType *result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (size_t j = 0; j < in->Height(); j += 8) {
         for (size_t i = 0; i < in->Width(); i += 8) {
           for (size_t y = 0; y < 8; ++y) {
             for (size_t x = 0; x < 8; ++x) {
               auto pixel = in->GetAt(i + x, j + y);
               pixel[0] *= static_cast<T>(_coeffs[y * 8 + x]);
               result->SetAt(i + x, j + y, std::move(pixel));
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   Quantize8x8<T, Prec>(QuantizeType ty)
     : Base()
     , _coeffs(ty == eQuantizeType_JPEGLuma ?
               std::move(std::array<uint32_t, 64>
                  {{ 16, 11, 10, 16, 24, 40, 51, 61,
                     12, 12, 14, 19, 26, 58, 60, 55,
                     14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62,
                     18, 22, 37, 56, 68, 109, 103, 77,
                     24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101,
                     72, 92, 95, 98, 112, 100, 103, 99 }}) :
               std::move(std::array<uint32_t, 64> {
                   { 17, 18, 24, 47, 99, 99, 99, 99,
                     18, 21, 26, 66, 99, 99, 99, 99,
                     24, 26, 56, 99, 99, 99, 99, 99,
                     47, 66, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99 }})
   ) { }

   const std::array<uint32_t, 64> _coeffs;
};

class DropAlpha : public PipelineUnit<RGBAImage, RGBImage> {
 public:
  typedef PipelineUnit<RGBAImage, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new DropAlpha); }
  Base::ReturnType Run(const Base::ArgType &in) const override {
    std::vector<uint8_t> img_data;
    img_data.reserve(in->Width() * in->Height() * 3);

    for (size_t j = 0; j < in->Height(); ++j) {
      for (size_t i = 0; i < in->Width(); ++i) {
        auto pixel = in->GetAt(i, j);
        for (size_t ch = 0; ch < 3; ++ch) {
          assert(pixel[ch] < 256);
          img_data.push_back(static_cast<uint8_t>(pixel[ch]));
        }
      }
    }

    RGBImage *img = new RGBImage(in->Width(), in->Height(), std::move(img_data));
    return std::move(std::unique_ptr<RGBImage>(img));
  }
};

}  // namespace GenTC

#endif  // __TCAR_IMAGE_PROCESSING_H__
