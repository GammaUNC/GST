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
      uint8_t pixel = img.GetAt(i, j);
      EXPECT_EQ(pixel, j * 4 + i);
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
      GenTC::RGB pixel = img.GetAt(i, j);
      EXPECT_EQ(std::get<0>(pixel), 0xFF);
      if (i == 1 && j == 2) {
        EXPECT_EQ(std::get<1>(pixel), 0xC0);
      } else {
        EXPECT_EQ(std::get<1>(pixel), 0x00);
      }
      EXPECT_EQ(std::get<2>(pixel), 0x00);
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
      GenTC::RGB565 pixel = img.GetAt(i, j);
      EXPECT_EQ(std::get<0>(pixel), 0x1F);
      if (i == 1 && j == 2) {
        EXPECT_EQ(std::get<1>(pixel), 0x01);
      } else {
        EXPECT_EQ(std::get<1>(pixel), 0x00);
      }
      EXPECT_EQ(std::get<2>(pixel), 0x1F);
    }
  }
}

TEST(Image, CanReadPackedBinaryImage) {
  GenTC::BinaryImage img = GenTC::BinaryImage(4, 4, { 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      auto pixel = img.GetAt(i, j);
      if ((i + j) & 1) {
        EXPECT_EQ(pixel, 0x01);
      } else {
        EXPECT_EQ(pixel, 0x00);
      }
    }
  }
}

TEST(Image, CanReadPackedTwoBitImage) {
  GenTC::TwoBitImage img = GenTC::TwoBitImage(4, 4, { 0x5A, 0x5A, 0x5A, 0x5A });
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      auto pixel = img.GetAt(i, j);
      if (i < 2) {
        EXPECT_EQ(pixel, 0x01);
      } else {
        EXPECT_EQ(pixel, 0x02);
      }
    }
  }
}

TEST(Image, CanPackBinaryImage) {
  GenTC::BinaryImage img = GenTC::BinaryImage(4, 2);
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 4; ++i) {
      img.SetAt(i, j, static_cast<unsigned>(((i * j) % 2) == 0));
    }
  }

  std::vector<uint8_t> packed = img.Pack();
  EXPECT_EQ(packed.size(), 1);
  EXPECT_EQ(packed[0], 0xFA);
}

TEST(Image, CanRepackRGB565) {
  std::vector<uint8_t> buf_565 = { 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
                                   0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F,
                                   0xF8, 0x1F, 0xF8, 0x3F, 0xF8, 0x1F, 0xF8, 0x1F,
                                   0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F, 0xF8, 0x1F };
  GenTC::RGB565Image img = GenTC::RGB565Image(4, 4, std::vector<uint8_t>(buf_565));

  std::vector<uint8_t> packed_565 = img.Pack();
  ASSERT_EQ(packed_565.size(), buf_565.size());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(packed_565[i], buf_565[i]);
  }
}

TEST(Image, CanSplitImage) {
  std::unique_ptr<GenTC::RGBImage> img(new GenTC::RGBImage(4, 4,
  { 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00 }));

  typedef std::array<GenTC::Image<GenTC::Alpha>, 3> SplitResultType;
  std::unique_ptr<GenTC::RGBSplitter> splitter(new GenTC::RGBSplitter);
  auto p = GenTC::Pipeline<GenTC::RGBImage, SplitResultType>
    ::Create(GenTC::RGBSplitter::New());
  auto result = p->Run(img);

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      uint8_t pixel = result->at(0).GetAt(i, j);
      EXPECT_EQ(pixel, 0xFF);

      pixel = result->at(1).GetAt(i, j);
      if (i == 1 && j == 2) {
        EXPECT_EQ(pixel, 0xC0);
      } else {
        EXPECT_EQ(pixel, 0x00);
      }

      pixel = result->at(2).GetAt(i, j);
      EXPECT_EQ(pixel, 0x00);      
    }
  }
}
