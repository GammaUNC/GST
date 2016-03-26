#ifndef __TCAR_IMAGE_PROCESSING_H__
#define __TCAR_IMAGE_PROCESSING_H__

#include "fast_dct.h"
#include "wavelet.h"

#include "pipeline.h"
#include "image.h"

#include <array>

#include <iostream>

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

class RGB565toYCoCg667
: public PipelineUnit<RGB565Image, YCoCg667Image> {
 public:
  typedef PipelineUnit<RGB565Image, YCoCg667Image> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new RGB565toYCoCg667); }
  std::unique_ptr<YCoCg667Image> Run(const std::unique_ptr<RGB565Image> &) const override;
};

class YCoCg667toRGB565
: public PipelineUnit<YCoCg667Image, RGB565Image> {
 public:
  typedef PipelineUnit<YCoCg667Image, RGB565Image> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new YCoCg667toRGB565); }
  std::unique_ptr<RGB565Image> Run(const std::unique_ptr<YCoCg667Image> &) const override;
};

template<typename T>
class Quantize8x8
  : public PipelineUnit < Image<T>, Image<T> > {
  static_assert(PixelTraits::NumChannels<T>::value == 1, "Can only quantize single channel images!");
 public:
   typedef Image<T> ImageType;
   typedef PipelineUnit<Image<T>, Image<T>> Base;

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

   class Quantizer : public Quantize8x8<T> {
   public:
     Quantizer(QuantizeType ty) : Quantize8x8<T>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) const override {
       ImageType *result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (size_t j = 0; j < in->Height(); j += 8) {
         for (size_t i = 0; i < in->Width(); i += 8) {
           for (size_t y = 0; y < 8; ++y) {
             for (size_t x = 0; x < 8; ++x) {
               T pixel = in->GetAt(i + x, j + y);
               pixel /= static_cast<T>(_coeffs[y * 8 + x]);
               result->SetAt(i + x, j + y, pixel);
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   class Dequantizer : public Quantize8x8<T> {
   public:
     Dequantizer(QuantizeType ty) : Quantize8x8<T>(ty) { }
     std::unique_ptr<ImageType> Run(const std::unique_ptr<ImageType> &in) const override {
       ImageType *result = new ImageType(in->Width(), in->Height());

       assert((in->Width() % 8) == 0);
       assert((in->Height() % 8) == 0);

       for (size_t j = 0; j < in->Height(); j += 8) {
         for (size_t i = 0; i < in->Width(); i += 8) {
           for (size_t y = 0; y < 8; ++y) {
             for (size_t x = 0; x < 8; ++x) {
               T pixel = in->GetAt(i + x, j + y);
               pixel *= static_cast<T>(_coeffs[y * 8 + x]);
               result->SetAt(i + x, j + y, pixel);
             }
           }
         }
       }

       return std::move(std::unique_ptr<ImageType>(result));
     }
   };

   Quantize8x8<T>(QuantizeType ty)
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

template<typename T>
static void Transpose8x8(T *block) {
  // Transpose block
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      if (x < y) {
        uint32_t idx1 = y * 8 + x;
        uint32_t idx2 = x * 8 + y;
        std::swap(block[idx1], block[idx2]);
      }
    }
  }
}

template<typename T>
class ForwardDCT : PipelineUnit<Image<T>, SixteenBitImage > {
  static_assert(PixelTraits::NumChannels<T>::value == 1,
                "DCT is a single-channel operation!");
public:
  typedef PipelineUnit<Image<T>, SixteenBitImage> Base;

  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new ForwardDCT<T>);
  }

  typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    assert(in->Width() % 8 == 0);
    assert(in->Height() % 8 == 0);

    // Sixteen bit images are packed...
    std::vector<uint8_t> result(in->Width() * in->Height() * 2);

    for (uint32_t j = 0; j < in->Height(); j += 8) {
      for (uint32_t i = 0; i < in->Width(); i += 8) {
        float block[64];

        // Read block
        for (uint32_t y = 0; y < 8; ++y) {
          for (uint32_t x = 0; x < 8; ++x) {
            uint32_t idx = y * 8 + x;
            block[idx] = static_cast<float>(in->GetAt(i + x, j + y));
          }
        }

        // Run forward DCT...
        for (int r = 0; r < 8; ++r) {
          float *row = block + r * 8;
          fdct(row, row);
        }

        // Transpose
        Transpose8x8(block);

        // Run forward DCT...
        for (int r = 0; r < 8; ++r) {
          float *row = block + r * 8;
          fdct(row, row);
        }

        // Transpose
        Transpose8x8(block);

        // Write block
        for (size_t y = 0; y < 8; ++y) {
          for (size_t x = 0; x < 8; ++x) {
            size_t idx = y * 8 + x;

            size_t dst_idx = ((j + y) * in->Width() + i + x) * 2;
            uint8_t *ptr = result.data();

            int16_t v = static_cast<int16_t>(block[idx]);
            ptr[dst_idx + 0] = (v >> 8) & 0xFF;
            ptr[dst_idx + 1] = v & 0xFF;
          }
        }
      }
    }

    SixteenBitImage *ret_img =
      new SixteenBitImage(in->Width(), in->Height(), result);
    return std::move(typename Base::ReturnType(ret_img));
  }
};

class InverseDCT : PipelineUnit<SixteenBitImage, AlphaImage> {
public:
  typedef PipelineUnit<SixteenBitImage, AlphaImage> Base;
  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new InverseDCT);
  }

  virtual Base::ReturnType Run(const Base::ArgType &in) const override;
};

// !HACK! We might just need to switch do different wavelets
template <typename T, bool IsSigned>
struct WaveletResultTy {
  typedef typename PixelTraits::SignedTypeForBits<PixelTraits::BitsUsed<T>::value + 2>::Ty DstTy;
  static const size_t kNumDstBits = PixelTraits::BitsUsed<T>::value + 2;
};

template <typename T>
struct WaveletResultTy<T, true> {
  typedef typename PixelTraits::SignedTypeForBits<PixelTraits::BitsUsed<T>::value + 1>::Ty DstTy;
  static const size_t kNumDstBits = PixelTraits::BitsUsed<T>::value + 1;
};

template <typename T, size_t BlockSize>
class FWavelet2D : public PipelineUnit<Image<T>,
  Image< typename WaveletResultTy<T, PixelTraits::IsSigned<T>::value >::DstTy > > {
public:
  typedef WaveletResultTy<T, PixelTraits::IsSigned<T>::value> ResultTy;
  static const size_t kNumSrcBits = PixelTraits::BitsUsed<T>::value;
  static const size_t kNumDstBits = ResultTy::kNumDstBits;

  typedef typename PixelTraits::SignedTypeForBits<kNumDstBits>::Ty DstTy;
  typedef Image<T> InputImage;
  typedef Image<DstTy> OutputImage;
  typedef PipelineUnit<InputImage, OutputImage> Base;

  static_assert(PixelTraits::NumChannels<T>::value == 1,
    "Wavelet transform only operates on single channel images");
  static_assert(PixelTraits::BitsUsed<typename OutputImage::PixelType>::value <= 16,
    "Wavelet coefficients end up in 16 bit signed integers!");
  static_assert((BlockSize & (BlockSize - 1)) == 0,
    "Block size must be a power of two!");

  static std::unique_ptr<Base> New() {
    return std::unique_ptr<Base>(new FWavelet2D<T, BlockSize>);
  }

  virtual typename Base::ReturnType Run(const typename Base::ArgType &in) const override {
    assert((in->Width() % BlockSize) == 0);
    assert((in->Height() % BlockSize) == 0);
    OutputImage *result = new OutputImage(in->Width(), in->Height());

    for (size_t j = 0; j < in->Height(); j += BlockSize) {
      for (size_t i = 0; i < in->Width(); i += BlockSize) {
        std::vector<int16_t> block(BlockSize * BlockSize);

        // Populate block
        for (size_t y = 0; y < BlockSize; ++y) {
          for (size_t x = 0; x < BlockSize; ++x) {
            size_t local_idx = y * BlockSize + x;
            T pixel = in->GetAt(i + x, j + y);
            assert(static_cast<int64_t>(pixel) <= PixelTraits::Max<int16_t>::value);
            assert(static_cast<int64_t>(pixel) >= PixelTraits::Min<int16_t>::value);
            block[local_idx] = static_cast<int16_t>(pixel);
          }
        }

        // Do transform
        static const size_t kRowBytes = sizeof(int16_t) * BlockSize;
        size_t dim = BlockSize;
        while (dim > 1) {
          ForwardWavelet2D(block.data(), kRowBytes, block.data(), kRowBytes, dim);
          dim /= 2;
        }

        // Output to image...
        for (size_t y = 0; y < BlockSize; ++y) {
          for (size_t x = 0; x < BlockSize; ++x) {
            size_t local_idx = y * BlockSize + x;
            assert(static_cast<DstTy>(block[local_idx]) <= PixelTraits::Max<DstTy>::value);
            assert(static_cast<DstTy>(block[local_idx]) >= PixelTraits::Min<DstTy>::value);
            result->SetAt(i + x, j + y, static_cast<DstTy>(block[local_idx]));
          }
        }
      }
    }

    return std::move(typename Base::ReturnType(result));
  }
};

}  // namespace GenTC

#endif  // __TCAR_IMAGE_PROCESSING_H__
