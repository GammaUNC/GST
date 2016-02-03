#ifndef __CODEC_IMAGE_H__
#define __CODEC_IMAGE_H__

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

template <unsigned NumChannels, typename Prec = Precision<NumChannels> >
class Image {
 public:
   Image<NumChannels, Prec>(size_t w, size_t h, const std::vector<uint8_t> &img)
     : _width(w), _height(h), _precision(Prec()), _data(img) {
     assert(img.size() == (_precision.PixelSizeInBits() * _width * _height) / 8);
   }

   const std::vector<uint8_t> &GetData() const { return _data; }

   std::array<uint32_t, NumChannels> At(int x, int y) const {
     size_t bit_offset = (y * _width + x) * _precision.PixelSizeInBits();

     std::array<uint32_t, NumChannels> result;
     for (size_t ch = 0; ch < NumChannels; ++ch) {
       assert(_precision[ch] < 32);
       size_t prec = static_cast<size_t>(_precision[ch]);
       size_t next_bit_offset = bit_offset + prec;

       uint32_t pixel = 0;

       size_t bits_to_byte = 8 - (bit_offset % 8);
       size_t bits_left = std::min(bits_to_byte, prec);
       if (8 != bits_left) {
         size_t val = _data[bit_offset / 8] & ((1 << bits_to_byte) - 1);
         if (bits_left != bits_to_byte) {
           val >>= bits_to_byte - bits_left;
         }
         pixel |= val;
         bit_offset += bits_left;
       }

       assert(bit_offset == next_bit_offset || (bit_offset % 8) == 0);

       size_t next_byte_offset = next_bit_offset / 8;
       while ((bit_offset / 8) < next_byte_offset) {
         pixel <<= 8;
         pixel |= _data[bit_offset / 8];
         bit_offset += 8;
       }

       bits_left = next_bit_offset - bit_offset;
       assert(bits_left < 8);
       if (bits_left > 0) {
         size_t shift = 8 - bits_left;
         pixel <<= bits_left;
         pixel |= _data[bit_offset / 8] >> shift;
       }

       assert(bit_offset + bits_left == next_bit_offset);
       bit_offset = next_bit_offset;
       result[ch] = pixel;
     }

     return std::move(result);
   }
   
 private:
   const Prec _precision;
   std::vector<uint8_t> _data;
   size_t _width;
   size_t _height;
};

}  // namespace GenTC

#endif  // __CODEC_IMAGE_H__