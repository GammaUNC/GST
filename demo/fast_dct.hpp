#ifndef __FAST__DCT_H__
#define __FAST__DCT_H__

#include <opencv2/opencv.hpp>
#include <cmath>

namespace dct {
  // Constants:
  static const float s1 = sin(1. * M_PI / 16.f);
  static const float c1 = cos(1. * M_PI / 16.f);
  static const float s3 = sin(3. * M_PI / 16.f);
  static const float c3 = cos(3. * M_PI / 16.f);
  static const float s6 = sin(3. * M_PI / 8.f);
  static const float c6 = cos(3. * M_PI / 8.f);
  static const float r2s6 = sqrt(2.f) * sin(3.f * M_PI / 8.f);
  static const float r2c6 = sqrt(2.f) * cos(3.f * M_PI / 8.f);

  static void fdct(float in[8], float out[8]) {
    // After stage 1:
    const float s1_0 = in[0] + in[7];
    const float s1_1 = in[1] + in[6];
    const float s1_2 = in[2] + in[5];
    const float s1_3 = in[3] + in[4];
    const float s1_4 = in[3] - in[4];
    const float s1_5 = in[2] - in[5];
    const float s1_6 = in[1] - in[6];
    const float s1_7 = in[0] - in[7];

    // After stage 2:
    const float s2_0 = s1_0 + s1_3;
    const float s2_1 = s1_1 + s1_2;
    const float s2_2 = s1_1 - s1_2;
    const float s2_3 = s1_0 - s1_3;

    const float z1 = c3 * (s1_7 + s1_4);
    const float s2_4 = ( s3-c3) * s1_7 + z1;
    const float s2_7 = (-s3-c3) * s1_4 + z1;

    const float z2 = c1 * (s1_6 + s1_5);
    const float s2_5 = ( s1-c1) * s1_6 + z2;
    const float s2_6 = (-s1-c1) * s1_5 + z2;

    // After stage 3:
    const float s3_0 =  s2_0 + s2_1;
    const float s3_1 = -s2_1 + s2_0;

    const float z3 = r2c6 * (s2_3 + s2_2);
    const float s3_2 = ( r2s6-r2c6) * s2_3 + z3;
    const float s3_3 = (-r2s6-r2c6) * s2_2 + z3;

    const float s3_4 =  s2_4 + s2_6;
    const float s3_5 = -s2_5 + s2_7;
    const float s3_6 = -s2_6 + s2_4;
    const float s3_7 =  s2_7 + s2_5;

    // After stage 4:
    const float s4_4 = -s3_4 + s3_7;
    const float s4_5 =  s3_5 * sqrt(2.f);
    const float s4_6 =  s3_6 * sqrt(2.f);
    const float s4_7 =  s3_7 + s3_4;

    // Shuffle and scaling:
    out[0] = s3_0 / sqrt(8.f);
    out[4] = s3_1 / sqrt(8.f);
    out[2] = s3_2 / sqrt(8.f);
    out[6] = s3_3 / sqrt(8.f);
    out[7] = s4_4 / sqrt(8.f);
    out[3] = s4_5 / sqrt(8.f);  // Alternative: s3_5 / 2
    out[5] = s4_6 / sqrt(8.f);
    out[1] = s4_7 / sqrt(8.f);
  }

  static void idct(float in[8], float out[8]) {
    const float s3_0 = out[0] * sqrt(8.f);
    const float s3_1 = out[4] * sqrt(8.f);
    const float s3_2 = out[2] * sqrt(8.f);
    const float s3_3 = out[6] * sqrt(8.f);
    const float s4_4 = out[7] * sqrt(8.f);
    const float s4_5 = out[3] * sqrt(8.f);
    const float s4_6 = out[5] * sqrt(8.f);
    const float s4_7 = out[1] * sqrt(8.f);

    const float s3_4 = 0.5f * (s4_7 - s4_4);
    const float s3_5 = s4_5 / sqrt(2.f);
    const float s3_6 = s4_6 / sqrt(2.f);
    const float s3_7 = 0.5f * (s4_7 + s4_4);

    const float s2_0 = 0.5f * (s3_0 + s3_1);
    const float s2_1 = 0.5f * (s3_0 - s3_1);
    const float s2_2 = (c6 * s3_2 - s6 * s3_3) / sqrt(2.f);
    const float s2_3 = (s6 * s3_2 + c6 * s3_3) / sqrt(2.f);
    const float s2_4 = 0.5f * (s3_4 + s3_6);
    const float s2_5 = 0.5f * (s3_7 - s3_5);
    const float s2_6 = 0.5f * (s3_4 - s3_6);
    const float s2_7 = 0.5f * (s3_7 + s3_5);

    const float s1_0 = 0.5f * (s2_0 + s2_3);
    const float s1_1 = 0.5f * (s2_1 + s2_2);
    const float s1_2 = 0.5f * (s2_1 - s2_2);
    const float s1_3 = 0.5f * (s2_0 - s2_3);
    const float s1_4 = c3 * s2_4 - s3 * s2_7;
    const float s1_5 = c1 * s2_5 - s1 * s2_6;
    const float s1_6 = s1 * s2_5 + c1 * s2_6;
    const float s1_7 = s3 * s2_4 + c3 * s2_7;

    in[0] = 0.5f * (s1_0 + s1_7);
    in[1] = 0.5f * (s1_1 + s1_6);
    in[2] = 0.5f * (s1_2 + s1_5);
    in[3] = 0.5f * (s1_3 + s1_4);
    in[4] = 0.5f * (s1_3 - s1_4);
    in[5] = 0.5f * (s1_2 - s1_5);
    in[6] = 0.5f * (s1_1 - s1_6);
    in[7] = 0.5f * (s1_0 - s1_7);
  }

  static void RunDCT(cv::Mat *m) {
    cv::Mat flt_m;
    m->convertTo(flt_m, CV_32FC1);

    for (int j = 0; j < flt_m.rows / 8; ++j) {
      for (int i = 0; i < flt_m.cols / 8; ++i) {
        cv::Mat block = flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)).clone();

        for (int r = 0; r < 8; ++r) {
          float *row = reinterpret_cast<float *>(block.ptr(r));
          fdct(row, row);
        }
        cv::transpose(block, block);

        for (int r = 0; r < 8; ++r) {
          float *row = reinterpret_cast<float *>(block.ptr(r));
          fdct(row, row);
        }
        cv::transpose(block, block);

        block.copyTo(flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)));
      }
    }

    *m = flt_m;
  }

  static void RunIDCT(cv::Mat *m) {
    cv::Mat flt_m;
    m->convertTo(flt_m, CV_32FC1);

    for (int j = 0; j < flt_m.rows / 8; ++j) {
      for (int i = 0; i < flt_m.cols / 8; ++i) {
        cv::Mat block = flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)).clone();

        for (int r = 0; r < 8; ++r) {
          float *row = reinterpret_cast<float *>(block.ptr(r));
          idct(row, row);
        }
        cv::transpose(block, block);

        for (int r = 0; r < 8; ++r) {
          float *row = reinterpret_cast<float *>(block.ptr(r));
          idct(row, row);
        }
        cv::transpose(block, block);

        block.copyTo(flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)));
      }
    }

    *m = flt_m;
  }

}  // namespace dct

#endif  // #define __FAST__DCT_H__
