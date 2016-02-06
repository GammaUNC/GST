#ifndef __TCAR_DATA_STREAM_H__
#define __TCAR_DATA_STREAM_H__

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace GenTC {

class DataStream {
 public:
   DataStream() : _read_idx(0) { }
   explicit DataStream(const std::vector<uint8_t> &d) : _read_idx(0), _data(d) { }
   const std::vector<uint8_t> &GetData() const { return _data; }

   void WriteByte(uint8_t x);
   void WriteShort(uint16_t x);
   void WriteInt(uint32_t x);
   void WriteLong(uint64_t x);

   uint8_t ReadByte();
   uint16_t ReadShort();
   uint32_t ReadInt();
   uint64_t ReadLong();

 private:
   size_t _read_idx;
   std::vector<uint8_t> _data;
};

}  // namespace codec

#endif  // __TCAR_DATA_STREAM_H__
