#ifndef __ANS_BITS_H__
#define __ANS_BITS_H__

#include <cassert>
#include <cstring>
#include <vector>

namespace ans {

  class BitWriter {
   public:
    BitWriter(unsigned char* out)
      : _out(out), _bits_written(0), _bytes_written(0), _bits_left(8) { }

    BitWriter(unsigned char* out, int bit_offset)
      : _out(out), _bits_written(0), _bytes_written(0), _bits_left(8 - bit_offset) {
      assert(bit_offset < 8);
      assert(bit_offset >= 0);
    }

    virtual ~BitWriter() { }

    virtual int BytesWritten() const { return _bytes_written; }
    virtual int BitsWritten() const { return _bits_written; }

    virtual void WriteBit(int bit) {
      _bits_written++;
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

    virtual void WriteBits(int val, int num_bits) {
      // If we're outputting bytes, then we can just write to memory
      if (8 == _bits_left && (num_bits & 7) == 0) {
        assert(reinterpret_cast<unsigned char*>(&num_bits)[0] == num_bits ||
               !"Must be little endian!");
        const int num_bytes = num_bits / 8;
        memcpy(_out, reinterpret_cast<const void*>(&val), num_bytes);
        _bytes_written += num_bytes;
        _bits_written += num_bytes * 8;
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
    int _bits_written;
    int _bytes_written;
    int _bits_left;
  };

  class ContainedBitWriter : public BitWriter {
   public:
    ContainedBitWriter() : BitWriter(NULL), _bytes_written(0), _bits_written(0) { }

    virtual void WriteBit(int bit) override {
      if ((_bits_written % 8) == 0) {
        _out.resize(_bytes_written + 1);
        BitWriter w(_out.data() + _bytes_written);
        w.WriteBit(bit);
      } else {
        BitWriter w(_out.data() + _bytes_written - 1, _bits_written % 8);
        w.WriteBit(bit);
      }

      if ((_bits_written % 8) == 0) {
        ++_bytes_written;
      }
      ++_bits_written;
    }

    virtual void WriteBits(int val, int num_bits) override {
      size_t target_bytes = (_bits_written + num_bits + 7) / 8;
      _out.resize(target_bytes);

      if ((_bits_written % 8) == 0) {
        BitWriter w(_out.data() + _bytes_written);
        w.WriteBits(val, num_bits);
      } else {
        BitWriter w(_out.data() + _bytes_written - 1, _bits_written % 8);
        w.WriteBits(val, num_bits);
      }

      _bits_written += num_bits;
      _bytes_written = target_bytes;
    }

    virtual int BytesWritten() const override { return _bytes_written; }
    virtual int BitsWritten() const override { return _bits_written; }

    std::vector<uint8_t> GetData() const { return _out; }

   private:
    std::vector<uint8_t> _out;
    int _bytes_written;
    int _bits_written;
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
