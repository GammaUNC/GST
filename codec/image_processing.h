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

template<typename Prec>
class Quantize8x8
  : public PipelineUnit < Image<1, Prec>, Image<1, Prec> > {
 public:
   typedef Image<1, Prec> ImageType;
   typedef PipelineUnit<ImageType, ImageType> Base;

   static std::unique_ptr<typename Base> QuantizeJPEGLuma() {
     return std::unique_ptr<typename Base>(new Quantizer<Prec>(eQuantizeType_JPEGLuma));
   }

   static std::unique_ptr<typename Base> QuantizeJPEGChroma() {
     return std::unique_ptr<typename Base>(new Quantizer<Prec>(eQuantizeType_JPEGChroma));
   }

   static std::unique_ptr<typename Base> DequantizeJPEGLuma() {
     return std::unique_ptr<typename Base>(new Dequantizer<Prec>(eQuantizeType_JPEGLuma));
   }

   static std::unique_ptr<typename Base> DequantizeJPEGChroma() {
     return std::unique_ptr<typename Base>(new Dequantizer<Prec>(eQuantizeType_JPEGChroma));
   }

 protected:
   enum QuantizeType {
     eQuantizeType_JPEGLuma,
     eQuantizeType_JPEGChroma
   };

   class Quantizer : public Quantize8x8 < Prec > {
   public:
     Quantizer(QuantizeType ty) : Quantize8x8<Prec>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) override {
       ImageType result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (int j = 0; j < in->Height(); j += 8) {
         for (int i = 0; i < in->Width(); i += 8) {
           for (int y = 0; y < 8; ++y) {
             for (int x = 0; x < 8; ++x) {
               auto pixel = in->GetAt(i + x, j + y);
               pixel[0] /= _coeffs[y * 8 + x];
               result->SetAt(i + x, j + y, pixel);
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   class Dequantizer : public Quantize8x8 < Prec > {
   public:
     Dequantizer(QuantizeType ty) : Quantize8x8<Prec>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) override {
       ImageType result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (int j = 0; j < in->Height(); j += 8) {
         for (int i = 0; i < in->Width(); i += 8) {
           for (int y = 0; y < 8; ++y) {
             for (int x = 0; x < 8; ++x) {
               auto pixel = in->GetAt(i + x, j + y);
               pixel[0] /= _coeffs[y * 8 + x];
               result->SetAt(i + x, j + y, pixel);
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   Quantize8x8<Prec>(QuantizeType ty)
     : Base()
     : _coeffs(ty == eQuantizeType_JPEGLuma ?
   { 16, 11, 10, 16, 24, 40, 51, 61,
     12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77,
     24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101,
     72, 92, 95, 98, 112, 100, 103, 99 } :
   { 17, 18, 24, 47, 99, 99, 99, 99,
     18, 21, 26, 66, 99, 99, 99, 99,
     24, 26, 56, 99, 99, 99, 99, 99,
     47, 66, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99 };
   ) { }

   const std::array<uint32_t, 64> _coeffs;
};

}  // namespace GenTC

#endif  // __TCAR_IMAGE_PROCESSING_H__
