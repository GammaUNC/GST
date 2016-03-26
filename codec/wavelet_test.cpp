#include "wavelet.h"

#include <cstdint>
#include <cmath>

#include "gtest/gtest.h"

TEST(Wavelet, ForwardTransform) {
  int16_t xs[10] = { 10, 11, 12, 10, 12, 11, 10, 11, 11, 12 };
  int16_t out[10];
  size_t mid = GenTC::ForwardWavelet1D(xs, out, 10);

  EXPECT_EQ(mid, 5);
  EXPECT_EQ(out[0], 10);
  EXPECT_EQ(out[1], 12);
  EXPECT_EQ(out[2], 12);
  EXPECT_EQ(out[3], 10);
  EXPECT_EQ(out[4], 12);
  EXPECT_EQ(out[5], 0);
  EXPECT_EQ(out[6], -2);
  EXPECT_EQ(out[7], 0);
  EXPECT_EQ(out[8], 1);
  EXPECT_EQ(out[9], 1);
}

TEST(Wavelet, ForwardAndBackwardTransform) {
  int16_t xs[10] = { 10, 11, 12, 10, 12, 11, 10, 11, 11, 12 };
  int16_t tmp[10];
  int16_t out[10];
  memset(tmp, 0, sizeof(tmp));
  memset(out, 0, sizeof(out));

  size_t mid = GenTC::ForwardWavelet1D(xs, tmp, 10);
  GenTC::InverseWavelet1D(tmp, out, 10);

  EXPECT_EQ(mid, 5);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}

TEST(Wavelet, ReversibleWithOddNumberCoefficients) {
  int16_t xs[9] = { 10, 11, 12, 10, 12, 11, 10, 11, 11 };
  int16_t tmp[9];
  int16_t out[9];
  memset(tmp, 0, sizeof(tmp));
  memset(out, 0, sizeof(out));

  size_t mid = GenTC::ForwardWavelet1D(xs, tmp, 9);
  GenTC::InverseWavelet1D(tmp, out, 9);

  EXPECT_EQ(mid, 5);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}

TEST(Wavelet, ReversibleWithTwoCoefficients) {
  int16_t xs[] = { 10, 100 };
  static const size_t kNumCoeffs = sizeof(xs) / sizeof(xs[0]);
  int16_t tmp[kNumCoeffs];
  int16_t out[kNumCoeffs];
  memset(tmp, 0, sizeof(tmp));
  memset(out, 0, sizeof(out));

  size_t mid = GenTC::ForwardWavelet1D(xs, tmp, kNumCoeffs);
  GenTC::InverseWavelet1D(tmp, out, kNumCoeffs);

  EXPECT_EQ(mid, 1);
  for (size_t i = 0; i < kNumCoeffs; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}

TEST(Wavelet, IdentityOnSingleCoeff) {
  for (int16_t x = 0; x < 256; ++x) {
    int16_t xs[] = { x };
    static const int kNumCoeffs = sizeof(xs) / sizeof(xs[0]);
    int16_t tmp[kNumCoeffs];
    int16_t out[kNumCoeffs];
    memset(tmp, 0, sizeof(tmp));
    memset(out, 0, sizeof(out));

    size_t mid = GenTC::ForwardWavelet1D(xs, tmp, kNumCoeffs);
    GenTC::InverseWavelet1D(tmp, out, kNumCoeffs);

    EXPECT_EQ(mid, 0);
    EXPECT_EQ(out[0], x);
  }
}

TEST(Wavelet, ExtremeFrequency) {
  int16_t xs[] = { 255, 0, 254, 1, 253, 2, 252, 3, 251, 4, 250 };
  static const size_t kNumCoeffs = sizeof(xs) / sizeof(xs[0]);
  int16_t tmp[kNumCoeffs];
  int16_t out[kNumCoeffs];
  memset(tmp, 0, sizeof(tmp));
  memset(out, 0, sizeof(out));

  size_t mid = GenTC::ForwardWavelet1D(xs, tmp, kNumCoeffs);
  GenTC::InverseWavelet1D(tmp, out, kNumCoeffs);

  EXPECT_EQ(mid, kNumCoeffs/2 + 1);
  for (size_t i = 0; i < kNumCoeffs; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}

TEST(Wavelet, Small2DWavelet) {
  int16_t xs[] = { 234, 215, 223, 211 };
  static const size_t kNumCoeffs = sizeof(xs) / sizeof(xs[0]);
  static const size_t kDim = static_cast<size_t>(std::sqrt(kNumCoeffs));
  ASSERT_EQ(kDim * kDim, kNumCoeffs);

  int16_t tmp[kNumCoeffs];
  int16_t out[kNumCoeffs];
  GenTC::ForwardWavelet2D(xs, sizeof(xs[0]) * kDim, tmp, sizeof(tmp[0]) * kDim, kDim);
  for (size_t i = 0; i < kNumCoeffs; ++i) {
    EXPECT_NE(tmp[i], xs[i]) << "At index: " << i;
  }

  GenTC::InverseWavelet2D(tmp, sizeof(tmp[0]) * kDim, out, sizeof(out[0]) * kDim, kDim);

  for (size_t i = 0; i < kNumCoeffs; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}

TEST(Wavelet, Inverse2DWavelet) {
  int16_t xs[] = {
    63,  64,  0, -1,
    66,  60,  6,  9,
     0,   2, -2, -2,
     7, -18, 16, 36
  };

  int16_t tmp[16];
  GenTC::InverseWavelet2D(xs, 8, tmp, 8, 4);

  const int16_t expected[] = {
    63, 63, 63, 63,
    63, 63, 64, 63,
    63, 65, 62, 64,
    62, 65, 31, 69
  };

  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(tmp[i], expected[i]) << "At index: " << i;
  }
  
  int16_t out[16];
  GenTC::ForwardWavelet2D(tmp, 8, out, 8, 4);

  // Forward wavelet should recreate it..
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }  
}

TEST(Wavelet, Recursive2DWavelet) {
  int16_t xs[] = {
    234, 215, 223, 211,
    205, 21, 34, 245,
    101, 234, 110, 159,
    201, 198, 112, 174
  };

  static const size_t kNumCoeffs = sizeof(xs) / sizeof(xs[0]);
  static const size_t kDim = static_cast<size_t>(std::sqrt(kNumCoeffs));
  ASSERT_EQ(kDim * kDim, kNumCoeffs);

  static const size_t kRowBytes = sizeof(xs[0]) * kDim;

  int16_t tmp[kNumCoeffs];
  int16_t out[kNumCoeffs];

  GenTC::ForwardWavelet2D(xs, kRowBytes, tmp, kRowBytes, kDim);
  GenTC::ForwardWavelet2D(tmp, kRowBytes, tmp, kRowBytes, kDim / 2);
  GenTC::InverseWavelet2D(tmp, kRowBytes, tmp, kRowBytes, kDim / 2);
  GenTC::InverseWavelet2D(tmp, kRowBytes, out, kRowBytes, kDim);

  for (size_t i = 0; i < kNumCoeffs; ++i) {
    EXPECT_EQ(out[i], xs[i]) << "At index: " << i;
  }
}
