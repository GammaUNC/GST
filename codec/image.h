#ifndef __TCAR_IMAGE_H__
#define __TCAR_IMAGE_H__

#include <iostream>

#include <array>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "pixel_traits.h"

namespace GenTC {

extern uint64_t ReadValue(const std::vector<uint8_t> &img_data, size_t *bit_offset, size_t prec);

template<typename T>
struct ImageUnpacker {
  static std::vector<T> go( const std::vector<uint8_t> &img_data, size_t width, size_t height) {
    std::vector<T> result;
    result.reserve(width * height);

    static const int kNumChannels = PixelTraits::NumChannels<T>::value;
    static_assert(kNumChannels == 1, "This better be true!");

    size_t bit_offset = 0;

    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const size_t prec = PixelTraits::BitsUsed<T>::value;
        T v = PixelTraits::ConvertUnsigned<T>::cvt(ReadValue(img_data, &bit_offset, prec));
        result.push_back(v);
      }
    }
 
    return std::move(result);
  }
};

template <typename T1, typename T2, typename T3>
struct ImageUnpacker<std::tuple<T1, T2, T3> > {
  typedef std::tuple<T1, T2, T3> PixelTy;
  static std::vector<PixelTy> go( const std::vector<uint8_t> &img_data, size_t width, size_t height) {
    std::vector<PixelTy> result;
    result.reserve(width * height);

    static const int kNumChannels = PixelTraits::NumChannels<PixelTy>::value;
    static_assert(kNumChannels == 3, "Pixel3's should have 3 channels!");

    size_t bit_offset = 0;
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {

        static const size_t prec1 = PixelTraits::BitsUsed<T1>::value;
        static const size_t prec2 = PixelTraits::BitsUsed<T2>::value;
        static const size_t prec3 = PixelTraits::BitsUsed<T3>::value;

        T1 r = PixelTraits::ConvertUnsigned<T1>::cvt(ReadValue(img_data, &bit_offset, prec1));
        T2 g = PixelTraits::ConvertUnsigned<T2>::cvt(ReadValue(img_data, &bit_offset, prec2));
        T3 b = PixelTraits::ConvertUnsigned<T3>::cvt(ReadValue(img_data, &bit_offset, prec3));
        PixelTy v = std::make_tuple(r, g, b);

        result.push_back(v);
      }
    }

    return std::move(result);
  }
};

template <typename T1, typename T2, typename T3, typename T4>
struct ImageUnpacker<std::tuple<T1, T2, T3, T4> > {
  typedef std::tuple<T1, T2, T3, T4> PixelTy;
  static std::vector<PixelTy> go( const std::vector<uint8_t> &img_data, size_t width, size_t height) {
    std::vector<PixelTy> result;
    result.reserve(width * height);

    static const int kNumChannels = PixelTraits::NumChannels<PixelTy>::value;
    static_assert(kNumChannels == 4, "Pixel4's should have 4 channels!");

    size_t bit_offset = 0;
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {

        static const size_t prec1 = PixelTraits::BitsUsed<T1>::value;
        static const size_t prec2 = PixelTraits::BitsUsed<T2>::value;
        static const size_t prec3 = PixelTraits::BitsUsed<T3>::value;
        static const size_t prec4 = PixelTraits::BitsUsed<T4>::value;

        T1 r = PixelTraits::ConvertUnsigned<T1>::cvt(ReadValue(img_data, &bit_offset, prec1));
        T2 g = PixelTraits::ConvertUnsigned<T2>::cvt(ReadValue(img_data, &bit_offset, prec2));
        T3 b = PixelTraits::ConvertUnsigned<T3>::cvt(ReadValue(img_data, &bit_offset, prec3));
        T4 a = PixelTraits::ConvertUnsigned<T4>::cvt(ReadValue(img_data, &bit_offset, prec4));
        PixelTy v = std::make_tuple(r, g, b, a);

        result.push_back(v);
      }
    }

    return std::move(result);
  }
};

template <typename T>
class Image {
 public:
  static const size_t kNumChannels = PixelTraits::NumChannels<T>::value;

  Image<T>()
    : _width(0)
    , _height(0)
    , _pixels() { }

  Image<T>(size_t w, size_t h)
    : _width(w)
    , _height(h)
    , _pixels(w * h) { }

  Image<T>(size_t w, size_t h, const std::vector<uint8_t> &img_data)
    : _width(w)
    , _height(h)
    , _pixels(ImageUnpacker<T>::go(img_data, _width, _height)) { }

  size_t Width() const { return _width; }
  size_t Height() const { return _height; }

  size_t BitDepth() const { return PixelTraits::BitsUsed<T>::value; }

  const std::vector<T> &GetPixels() const { return _pixels; }

  Image<T> &operator=(const Image<T> &other) {
    _width = other._width;
    _height = other._width;
    _pixels = other._pixels;
    return *this;
  }

  T GetAt(size_t x, size_t y) const {
    assert(x < Width());
    assert(y < Height());
    return _pixels[y * Width() + x];
  }

  void SetAt(size_t x, size_t y, T pixel) {
    assert(x < Width());
    assert(y < Height());
    _pixels[y * Width() + x] = pixel;
  }

  std::vector<uint8_t> Pack() const {
    size_t bit_offset = 0;
    const size_t kPixelSize = ((PixelTraits::BitsUsed<T>::value + 7) / 8);
    std::vector<uint8_t> result(Width() * Height() * kPixelSize, 0);
    for (const auto &p : _pixels) {
      assert(((bit_offset + 7) / 8) < result.size());
      PixelTraits::BitPacker<T>::pack(p, result.data(), &bit_offset);
    }
    assert(((bit_offset + 7) / 8) <= result.size());
    result.resize((bit_offset + 7) / 8);
    return std::move(result);
  }

 private:
  size_t _width;
  size_t _height;
  std::vector<T> _pixels;
};

typedef std::tuple<uint8_t, uint8_t, uint8_t> RGB;
typedef std::tuple<UnsignedBits<5>, UnsignedBits<6>, UnsignedBits<5> > RGB565;
typedef std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> RGBA;
typedef uint8_t Alpha;

typedef Image<RGB> RGBImage;
typedef Image<RGB565> RGB565Image;
typedef Image<RGBA> RGBAImage;
typedef Image<Alpha> AlphaImage;
typedef Image<UnsignedBits<1> > BinaryImage;
typedef Image<UnsignedBits<2> > TwoBitImage;
typedef Image<UnsignedBits<3> > ThreeBitImage;
typedef Image<UnsignedBits<4> > FourBitImage;
typedef Image<int16_t> SixteenBitImage;

// YCbCrImages aren't packed since we only really get them
// from RGB images...
typedef Image<RGB> YCbCrImage;

typedef std::tuple<UnsignedBits<6>, SignedBits<6>, SignedBits<7> > YCoCg667;
typedef Image<YCoCg667> YCoCg667Image;

}  // namespace GenTC

#endif  // __TCAR_IMAGE_H__
