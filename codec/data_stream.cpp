#include "data_stream.h"

namespace GenTC {

static void WriteBytes(std::vector<uint8_t> *buf, const uint8_t *bytes, size_t num_bytes) {
  size_t sz = buf->size();
  buf->resize(sz + num_bytes);
  for (size_t i = 0; i < num_bytes; ++i) {
    buf->at(sz + i) = bytes[i];
  }
}

#ifdef DATA_STREAM_WRITE_TYPE
#error "zzz"
#endif
#define DATA_STREAM_WRITE_TYPE(ty) \
  do { \
    WriteBytes(&_data, reinterpret_cast<const uint8_t *>(&x), sizeof(ty)); \
  } while(0)

void DataStream::WriteByte(uint8_t x) {
  DATA_STREAM_WRITE_TYPE(uint8_t);
}

void DataStream::WriteShort(uint16_t x) {
  DATA_STREAM_WRITE_TYPE(uint16_t);
}

void DataStream::WriteInt(uint32_t x) {
  DATA_STREAM_WRITE_TYPE(uint32_t);
}

void DataStream::WriteLong(uint64_t x) {
  DATA_STREAM_WRITE_TYPE(uint64_t);
}

#ifdef DATA_STREAM_READ_TYPE
#error "zzz"
#endif
#define DATA_STREAM_READ_TYPE(ty) \
  do { \
    ty x; \
    memcpy(&x, _data.data() + _read_idx, sizeof(ty)); \
    _read_idx += sizeof(ty); \
    return x; \
  } while(0)

uint8_t DataStream::ReadByte() {
  DATA_STREAM_READ_TYPE(uint8_t);
}

uint16_t DataStream::ReadShort() {
  DATA_STREAM_READ_TYPE(uint16_t);
}

uint32_t DataStream::ReadInt() {
  DATA_STREAM_READ_TYPE(uint32_t);
}

uint64_t DataStream::ReadLong() {
  DATA_STREAM_READ_TYPE(uint64_t);
}

}  // namespace GenTC