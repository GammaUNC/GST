#ifndef __ANS_BITS_H__
#define __ANS_BITS_H__

#include <cassert>
#include <cstring>

namespace ans {

  class BitWriter {
   public:
    BitWriter(unsigned char* out)
      : _out(out), _bytes_written(0), _bits_left(8) { }

    int BytesWritten() const { return _bytes_written; }

    void WriteBit(int bit) {
      if (8 == _bits_left) {
        _bytes_written++;
      }

      const int shift = 8 - _bits_left;
      *(_out) &= ~(1 << shift);
      *(_out) |= (!!bit) << shift;
      if (0 == --_bits_left) {
        _bits_left = 8;
        _out += 1;
      }
    }

    void WriteBits(int val, int num_bits) {
      // If we're outputting bytes, then we can just write to memory
      if (8 == _bits_left && (num_bits & 7) == 0) {
        assert(reinterpret_cast<unsigned char*>(&num_bits)[0] == num_bits ||
               !"Must be little endian!");
        const int num_bytes = num_bits / 8;
        memcpy(_out, reinterpret_cast<const void*>(&val), num_bytes);
        _bytes_written += num_bytes;
        _out += num_bytes;
      } else {
        for (int i = 0; i < num_bits; ++i) {
          WriteBit(val & 1);
          val >>= 1;
        }
      }
    }

  private:
    unsigned char* _out;
    int _bytes_written;
    int _bits_left;
  };

  class BitReader {
   public:
    BitReader(const unsigned char* in)
      : _in(in), _bytes_read(0), _bits_left(8) { }

    int BytesRead() const { return _bytes_read; }

    int ReadBit() {
      if (8 == _bits_left) {
        _bytes_read++;
      }

      const int shift = 8 - _bits_left;
      int result = !!((*_in) & (1 << shift));
      if (0 == --_bits_left) {
        _bits_left = 8;
        _in += 1;
      }
      return result;
    }

    int ReadBits(int num_bits) {
      // If we're outputting bytes, then we can just write to memory
      int result = 0;
      if (8 == _bits_left && (num_bits & 7) == 0) {
        assert(reinterpret_cast<unsigned char*>(&num_bits)[0] == num_bits ||
               !"Must be little endian!");
        const int num_bytes = num_bits / 8;
        memcpy(reinterpret_cast<void*>(&result), _in, num_bytes);
        _bytes_read += num_bytes;
        _in += num_bytes;
      } else {
        for (int i = 0; i < num_bits; ++i) {
          result |= (ReadBit() << i);
        }
      }
      return result;
    }

  private:
    const unsigned char* _in;
    int _bytes_read;
    int _bits_left;
  };

}  // namespace ans

#endif  // __ANS_BITS_H__
