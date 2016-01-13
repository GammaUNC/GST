#ifndef __OPENCV__DCT_H__
#define __OPENCV__DCT_H__

#include <opencv2/opencv.hpp>

namespace dct {

  static void RunDCT(cv::Mat *m) {
    cv::Mat flt_m;
    m->convertTo(flt_m, CV_32FC1);

    for (int j = 0; j < flt_m.rows / 8; ++j) {
      for (int i = 0; i < flt_m.cols / 8; ++i) {
        cv::Mat block = flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)).clone();
        cv::dct(block, block);
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
        cv::idct(block, block);
        block.copyTo(flt_m(cv::Rect_<int>(i*8, j*8, 8, 8)));
      }
    }

    *m = flt_m;
  }

}

#endif  // #define __OPENCV__DCT_H__
