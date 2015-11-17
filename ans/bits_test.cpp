#include <cstdint>

#include "bits.h"
#include "gtest/gtest.h"

TEST(Bits, CanWriteBytes) {
  uint32_t x;
  ans::BitWriter w(reinterpret_cast<unsigned char*>(&x));
  w.WriteBits(1, 8);
  w.WriteBits(0, 8);
  w.WriteBits(0, 8);
  w.WriteBits(0, 8);
  EXPECT_EQ(x, 1);
  EXPECT_EQ(w.BytesWritten(), 4);

  w = ans::BitWriter(reinterpret_cast<unsigned char*>(&x));
  w.WriteBits(0xbeef, 16);
  w.WriteBits(0xdead, 16);
  EXPECT_EQ(x, 0xdeadbeef);
  EXPECT_EQ(w.BytesWritten(), 4);
}

TEST(Bits, CanWriteBits) {
  int32_t x;
  ans::BitWriter w(reinterpret_cast<unsigned char*>(&x));
  for (int i = 0; i < 32; ++i) {
    w.WriteBit(1);
  }

  EXPECT_EQ(x, -1);
  EXPECT_EQ(w.BytesWritten(), 4);

  w = ans::BitWriter(reinterpret_cast<unsigned char*>(&x));
  for (int i = 0; i < 32; ++i) {
    if (i % 2 == 0) {
      w.WriteBit(0);
    } else {
      w.WriteBit(1);
    }
  }

  EXPECT_EQ(0xAAAAAAAA, x);
  EXPECT_EQ(w.BytesWritten(), 4);
}

TEST(Bits, CanWriteBytesAndBits) {
  int32_t x = 0;
  ans::BitWriter w(reinterpret_cast<unsigned char*>(&x));
  w.WriteBit(1);
  EXPECT_EQ(w.BytesWritten(), 1);
  w.WriteBits(3, 2);
  EXPECT_EQ(w.BytesWritten(), 1);
  w.WriteBits(7, 3);
  EXPECT_EQ(w.BytesWritten(), 1);
  w.WriteBits(15, 4);
  EXPECT_EQ(w.BytesWritten(), 2);
  w.WriteBits(31, 5);
  EXPECT_EQ(w.BytesWritten(), 2);
  w.WriteBit(1);
  EXPECT_EQ(w.BytesWritten(), 2);
  w.WriteBits(255, 8);
  EXPECT_EQ(w.BytesWritten(), 3);
  w.WriteBits(63, 6);

  EXPECT_EQ(0x3FFFFFFF, x);
  EXPECT_EQ(w.BytesWritten(), 4);
}

TEST(Bits, CanReadBytes) {
  uint32_t x = 1;
  ans::BitReader r(reinterpret_cast<unsigned char*>(&x));
  EXPECT_EQ(r.ReadBits(8), 1);
  EXPECT_EQ(r.ReadBits(8), 0);
  EXPECT_EQ(r.ReadBits(8), 0);
  EXPECT_EQ(r.ReadBits(8), 0);
  EXPECT_EQ(r.BytesRead(), 4);

  x = 0xdeadbeef;
  r = ans::BitReader(reinterpret_cast<unsigned char*>(&x));
  EXPECT_EQ(r.ReadBits(16), 0xbeef);
  EXPECT_EQ(r.ReadBits(16), 0xdead);
  EXPECT_EQ(r.BytesRead(), 4);
}

TEST(Bits, CanReadBits) {
  int32_t x = -1;
  ans::BitReader r(reinterpret_cast<unsigned char*>(&x));
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(r.ReadBit(), 1);
  }
  EXPECT_EQ(r.BytesRead(), 4);

  x = 0xAAAAAAAA;
  r = ans::BitReader(reinterpret_cast<unsigned char*>(&x));
  for (int i = 0; i < 32; ++i) {
    if (i % 2 == 0) {
      EXPECT_EQ(r.ReadBit(), 0);
    } else {
      EXPECT_EQ(r.ReadBit(), 1);
    }
  }
  EXPECT_EQ(r.BytesRead(), 4);
}

TEST(Bits, CanReadBytesAndBits) {
  int32_t x = 0x3FFFFFFF;
  ans::BitReader r(reinterpret_cast<unsigned char*>(&x));
  EXPECT_EQ(1, r.ReadBit());
  EXPECT_EQ(1, r.BytesRead());
  EXPECT_EQ(3, r.ReadBits(2));
  EXPECT_EQ(1, r.BytesRead());
  EXPECT_EQ(7, r.ReadBits(3));
  EXPECT_EQ(1, r.BytesRead());
  EXPECT_EQ(15, r.ReadBits(4));
  EXPECT_EQ(2, r.BytesRead());
  EXPECT_EQ(31, r.ReadBits(5));
  EXPECT_EQ(2, r.BytesRead());
  EXPECT_EQ(1, r.ReadBit());
  EXPECT_EQ(2, r.BytesRead());
  EXPECT_EQ(255, r.ReadBits(8));
  EXPECT_EQ(3, r.BytesRead());
  EXPECT_EQ(63, r.ReadBits(6));
  EXPECT_EQ(4, r.BytesRead());
}

TEST(Bits, CanWriteThenReadSameValues) {
  uint8_t stream[8];
  ans::BitWriter w(stream);
  for (int i = 1; i < 11; ++i) {
    w.WriteBits(i - 1, i);
  }

  EXPECT_EQ(stream[0], 0xD2);
  EXPECT_EQ(stream[1], 0x90);
  EXPECT_EQ(stream[2], 0xC2);
  EXPECT_EQ(7, w.BytesWritten());

  ans::BitReader r(stream);
  for (int i = 1; i < 11; ++i) {
    EXPECT_EQ(i - 1, r.ReadBits(i));
  }
}
