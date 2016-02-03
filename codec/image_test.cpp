#include <cstdint>

#include "image.h"
#include "gtest/gtest.h"

TEST(Image, CanReadBytes) {
  GenTC::Image<1, GenTC::Alpha> img = GenTC::Image<1, GenTC::Alpha>(4, 4,
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint32_t, 1> pixel = img.At(i, j);
      EXPECT_EQ(pixel[0], j * 4 + i);
    }
  }
}

TEST(Image, CanReadRGBPixels) {
  GenTC::Image<3, GenTC::RGB> img = GenTC::Image<3, GenTC::RGB>(4, 4,
  { 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00 });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint32_t, 3> pixel = img.At(i, j);
      EXPECT_EQ(pixel[0], 0xFF);
      if (i == 1 && j == 2) {
        EXPECT_EQ(pixel[1], 0xC0);
      } else {
        EXPECT_EQ(pixel[1], 0x00);
      }
      EXPECT_EQ(pixel[2], 0x00);
    }
  }
}

TEST(Image, CanReadRGB565) {
  GenTC::Image<3, GenTC::RGB565> img = GenTC::Image<3, GenTC::RGB565>(4, 4,
  { 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x3F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint32_t, 3> pixel = img.At(i, j);
      EXPECT_EQ(pixel[0], 0x1F);
      if (i == 1 && j == 2) {
        EXPECT_EQ(pixel[1], 0x01);
      } else {
        EXPECT_EQ(pixel[1], 0x00);
      }
      EXPECT_EQ(pixel[2], 0x1F);
    }
  }
}

TEST(Image, CanReadBinaryImage) {
  GenTC::Image<1, GenTC::SingleChannel<1> > img =
    GenTC::Image<1, GenTC::SingleChannel<1> >(4, 4, { 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint32_t, 1> pixel = img.At(i, j);
      if ((i + j) & 1) {
        EXPECT_EQ(pixel[0], 0x01);
      } else {
        EXPECT_EQ(pixel[0], 0x00);
      }
    }
  }
}

TEST(Image, CanReadTwoBitImage) {
  GenTC::Image<1, GenTC::SingleChannel<2> > img =
    GenTC::Image<1, GenTC::SingleChannel<2> >(4, 4, { 0x5A, 0x5A, 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint32_t, 1> pixel = img.At(i, j);
      if (i < 2) {
        EXPECT_EQ(pixel[0], 0x01);
      } else {
        EXPECT_EQ(pixel[0], 0x02);
      }
    }
  }
}