#include <cstdint>

#include "image.h"
#include "image_utils.h"
#include "pipeline.h"
#include "gtest/gtest.h"

TEST(Image, CanReadPackedBytes) {
  GenTC::AlphaImage img = GenTC::AlphaImage(4, 4,
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 1> pixel = img.GetAt(i, j);
      EXPECT_EQ(pixel[0], j * 4 + i);
    }
  }
}

TEST(Image, CanReadPackedRGBPixels) {
  GenTC::RGBImage img = GenTC::RGBImage(4, 4,
  { 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00 });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 3> pixel = img.GetAt(i, j);
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

TEST(Image, CanReadPackedRGB565) {
  GenTC::RGB565Image img = GenTC::RGB565Image(4, 4,
  { 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x3F, 0xF8, 0x1F, 0xF8, 0x1F,
    0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 3> pixel = img.GetAt(i, j);
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

TEST(Image, CanReadPackedBinaryImage) {
  typedef GenTC::PackedImage<1, GenTC::SingleChannel<1> > ImageType;
  GenTC::BinaryImage img = GenTC::BinaryImage(4, 4, { 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 1> pixel = img.GetAt(i, j);
      if ((i + j) & 1) {
        EXPECT_EQ(pixel[0], 0x01);
      } else {
        EXPECT_EQ(pixel[0], 0x00);
      }
    }
  }
}

TEST(Image, CanReadPackedTwoBitImage) {
  GenTC::TwoBitImage img = GenTC::TwoBitImage(4, 4, { 0x5A, 0x5A, 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 1> pixel = img.GetAt(i, j);
      if (i < 2) {
        EXPECT_EQ(pixel[0], 0x01);
      } else {
        EXPECT_EQ(pixel[0], 0x02);
      }
    }
  }
}

TEST(Image, CanSplitImage) {
  std::unique_ptr<GenTC::RGBImage> img(new GenTC::RGBImage(4, 4,
  { 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00 }));

  typedef std::array<GenTC::Image<1, uint8_t>, 3> SplitResultType;
  std::unique_ptr<GenTC::RGBSplitter> splitter(new GenTC::RGBSplitter);
  auto p = GenTC::Pipeline<GenTC::RGBImage, SplitResultType>
    ::Create(GenTC::RGBSplitter::New());
  auto result = p->Run(img);

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      std::array<uint8_t, 1> pixel = result->at(0).GetAt(i, j);
      EXPECT_EQ(pixel[0], 0xFF);

      pixel = result->at(1).GetAt(i, j);
      if (i == 1 && j == 2) {
        EXPECT_EQ(pixel[0], 0xC0);
      } else {
        EXPECT_EQ(pixel[0], 0x00);
      }

      pixel = result->at(2).GetAt(i, j);
      EXPECT_EQ(pixel[0], 0x00);      
    }
  }
}
