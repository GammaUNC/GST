#include "fast_dct.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// #define VERBOSE
#include "codec.h"
#include "gpu.h"
#include "kernel_cache.h"

#include <opencv2/opencv.hpp>

cv::Mat dft_opencv(const cv::Mat I) {
  using namespace cv; // !YUCK!
  
  Mat padded_flts;
  I.convertTo(padded_flts, CV_32FC1);

  Mat padded;                            //expand input image to optimal size
  int m = getOptimalDFTSize( I.rows );
  int n = getOptimalDFTSize( I.cols ); // on the border add zero values
  copyMakeBorder(I, padded_flts, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

  Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
  Mat complexI;
  merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  dft(complexI, complexI);            // this way the result may fit in the source matrix

  // compute the magnitude and switch to logarithmic scale
  // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  Mat magI = planes[0];

  magI += Scalar::all(1);                    // switch to logarithmic scale
  log(magI, magI);
  
  // crop the spectrum, if it has an odd number of rows or columns
  magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI.cols/2;
  int cy = magI.rows/2;

  Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a

  Mat result;
  magI.convertTo(result, CV_8UC1, 255.0, 0.5);
  
  return result;
}

int main(int argc, char **argv) {
  // Make sure that we have the proper number of arguments...
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << "<original> [compressed]" << std::endl;
    return 1;
  }

  std::unique_ptr<gpu::GPUContext> ctx = gpu::GPUContext::InitializeOpenCL(false);

  const char *orig_fn = argv[1];
  const char *cmp_fn = (argc == 2) ? NULL : argv[2];

#if 0
  GenTC::DXTImage dxt_img = GenTC::DXTImage(orig_fn, cmp_fn);
#else
  std::vector<uint8_t> cmp_img = std::move(GenTC::CompressDXT(orig_fn, cmp_fn));
  GenTC::DXTImage dxt_img = GenTC::DecompressDXT(ctx, cmp_img);
#endif

  // Decompress into image...
  std::vector<uint8_t> decomp_rgba = std::move(dxt_img.DecompressedImage()->Pack());

  cv::Mat decomp_img(dxt_img.Height(), dxt_img.Width(), CV_8UC4, decomp_rgba.data());
  cv::Mat decomp_output;
  cv::cvtColor(decomp_img, decomp_output, cv::COLOR_BGRA2RGBA);
  cv::imwrite("img_dxt.png", decomp_output);

  // Visualize interpolation data...
  std::vector<uint8_t> interp_img_data = std::move(dxt_img.InterpolationImage());
  cv::Mat interp_img = cv::Mat(dxt_img.Height(), dxt_img.Width(), CV_8UC1, interp_img_data.data());
  cv::imwrite("img_dxt_interp.png", interp_img);

  // cv::imwrite("img_dxt_interp_dft.png", dft_opencv(interp_img));

  clFlush(ctx->GetDefaultCommandQueue());
  clFinish(ctx->GetDefaultCommandQueue());
  gpu::GPUKernelCache::Clear();

  return 0;
}
