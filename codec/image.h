#ifndef __TCAR_IMAGE_H__
#define __TCAR_IMAGE_H__

#include <array>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace GenTC {

template <unsigned NumChannels>
class Precision {
 public:
  Precision() {
    // Default to thirty-two bits per channel...
    _precision.fill(32);
  }

  uint32_t PixelSize() const { return PixelSizeInBits() / 8; }
  uint32_t PixelSizeInBits() const {
    return std::accumulate(_precision.begin(), _precision.end(), 0);
  }

  const uint32_t operator[](const uint32_t idx) const {
    return _precision[idx];
  }

  const uint32_t operator[](const size_t idx) const {
    return _precision[idx];
  }

 protected:
  std::array<uint8_t, NumChannels> _precision;
};

struct RGB : public Precision<3> {
  RGB() : Precision<3>() {
    _precision[0] = 8;
    _precision[1] = 8;
    _precision[2] = 8;
  }
};

struct RGBA : public Precision<4> {
  RGBA() : Precision<4>() {
    _precision[0] = 8;
    _precision[1] = 8;
    _precision[2] = 8;
    _precision[3] = 8;
  }
};

template<unsigned SingleChannelBits>
struct SingleChannel : public Precision < 1 > {
  SingleChannel<SingleChannelBits>() : Precision<1>() {
    _precision[0] = SingleChannelBits;
  }
};

typedef SingleChannel<8> Alpha;

struct RGB565 : public Precision<3> {
  RGB565() : Precision<3>() {
    _precision[0] = 5;
    _precision[1] = 6;
    _precision[2] = 5;
  }
};

template<unsigned NumChannels, typename Prec = Precision<NumChannels> >
std::vector<std::array<uint32_t, NumChannels> >
UnpackImage( const std::vector<uint8_t> &img_data,
             const Prec &precision, size_t width, size_t height) {
  std::vector<std::array<uint32_t, NumChannels> > result;
  result.reserve(width * height);

  size_t bit_offset = 0;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {

      std::array<uint32_t, NumChannels> pixel;
      for (size_t ch = 0; ch < NumChannels; ++ch) {
        assert(precision[ch] < 32);
        size_t prec = static_cast<size_t>(precision[ch]);
        size_t next_bit_offset = bit_offset + prec;

        uint32_t ch_val = 0;

        size_t bits_to_byte = 8 - (bit_offset % 8);
        size_t bits_left = std::min(bits_to_byte, prec);
        if (8 != bits_left) {
          size_t val = img_data[bit_offset / 8] & ((1 << bits_to_byte) - 1);
          if (bits_left != bits_to_byte) {
            val >>= bits_to_byte - bits_left;
          }
          ch_val |= val;
          bit_offset += bits_left;
        }

        assert(bit_offset == next_bit_offset || (bit_offset % 8) == 0);

        size_t next_byte_offset = next_bit_offset / 8;
        while ((bit_offset / 8) < next_byte_offset) {
          ch_val <<= 8;
          ch_val |= img_data[bit_offset / 8];
          bit_offset += 8;
        }

        bits_left = next_bit_offset - bit_offset;
        assert(bits_left < 8);
        if (bits_left > 0) {
          size_t shift = 8 - bits_left;
          ch_val <<= bits_left;
          ch_val |= img_data[bit_offset / 8] >> shift;
        }

        assert(bit_offset + bits_left == next_bit_offset);
        bit_offset = next_bit_offset;
        pixel[ch] = ch_val;
      }

      result.push_back(std::move(pixel));
    }
  }

  return std::move(result);
}

template <unsigned NumChannels, typename Prec = Precision<NumChannels> >
class Image {
 public:
  static const int kNumChannels = NumChannels;

  Image<NumChannels, Prec>()
    : _width(0)
    , _height(0)
    , _precision(Prec())
    , _pixels() { }

  Image<NumChannels, Prec>(size_t w, size_t h)
    : _width(w)
    , _height(h)
    , _precision(Prec())
    , _pixels(w * h) { }

  Image<NumChannels, Prec>(size_t w, size_t h, const std::vector<uint8_t> &img_data)
    : _width(w)
    , _height(h)
    , _precision(Prec())
    , _pixels(UnpackImage<NumChannels, Prec>(img_data, _precision, _width, _height)) { }

  virtual ~Image<NumChannels, Prec>() { }

  size_t Width() const { return _width; }
  size_t Height() const { return _height; }

  size_t BitDepth() const { return _precision.PixelSizeInBits(); }
  size_t Precision(size_t channel) const { return _precision[channel]; }

  const std::vector<uint8_t> &GetPixels() const { return _pixels; }

  Image<NumChannels, Prec> &operator=(const Image<NumChannels, Prec> &other) {
    _width = other._width;
    _height = other._width;
    _pixels = other._pixels;
    return *this;
  }

  virtual std::array<uint32_t, NumChannels> GetAt(int x, int y) const {
    assert(0 <= x && static_cast<size_t>(x) < Width());
    assert(0 <= y && static_cast<size_t>(y) < Height());
    return _pixels[y * Width() + x];
  }

  virtual void SetAt(int x, int y, std::array<uint32_t, NumChannels> &&pixel) {
    assert(0 <= x && static_cast<size_t>(x) < Width());
    assert(0 <= y && static_cast<size_t>(y) < Height());
    _pixels[y * Width() + x] = pixel;
  }

 private:
  size_t _width;
  size_t _height;
  const Prec _precision;
  std::vector<std::array<uint32_t, NumChannels> > _pixels;
};

template <unsigned NumChannels, typename Prec = Precision<NumChannels> >
class PackedImage : public Image<NumChannels, Prec> {
 public:
   static const int kNumChannels = NumChannels;

   PackedImage<NumChannels, Prec>(size_t w, size_t h, std::vector<uint8_t> &&img_data)
     : Image<NumChannels, Prec>(w, h, std::vector<uint8_t>(img_data.begin(), img_data.end()))
     , _data(img_data) {
     assert(_data.size() == (this->BitDepth() * this->Width() * this->Height()) / 8);
   }

   const std::vector<uint8_t> &GetData() const { return _data; }

   void SetAt(int x, int y, std::array<uint32_t, NumChannels> &&pixel) override {
     assert(!"Unimplemented -- unpack, set, and repack image!");
   }

 private:
   std::vector<uint8_t> _data;
};

typedef PackedImage<3, RGB> RGBImage;
typedef PackedImage<3, RGB565> RGB565Image;
typedef PackedImage<4, RGBA> RGBAImage;
typedef PackedImage<1, Alpha> AlphaImage;
typedef PackedImage<1, SingleChannel<1> > BinaryImage;
typedef PackedImage<1, SingleChannel<2> > TwoBitImage;
typedef PackedImage<1, SingleChannel<3> > ThreeBitImage;
typedef PackedImage<1, SingleChannel<4> > FourBitImage;
typedef PackedImage<1, SingleChannel<16> > SixteenBitImage;

// YCbCrImages aren't packed since we only really get them
// from RGB images...
typedef Image<3, RGB> YCbCrImage;

}  // namespace GenTC

#endif  // __TCAR_IMAGE_H__
