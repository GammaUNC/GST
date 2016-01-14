#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

#define USE_FAST_DCT

#ifdef USE_FAST_DCT
#include "fast_dct.hpp"
#else
#include "opencv_dct.hpp"
#endif

#include <opencv2/opencv.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#define STB_DXT_IMPLEMENTATION
#include "stb_dxt.h"
#pragma GCC diagnostic pop

static uint64_t CompressRGB(const uint8_t *img, int width) {
  unsigned char block[64];
  memset(block, 0, sizeof(block));

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      int src_idx = (j * width + i) * 3;
      int dst_idx = (j * 4 + i) * 4;

      unsigned char *block_pixel = block + dst_idx;
      const unsigned char *img_pixel = img + src_idx;

      block_pixel[0] = img_pixel[0];
      block_pixel[1] = img_pixel[1];
      block_pixel[2] = img_pixel[2];
      block_pixel[3] = 0xFF;
    }
  }

  uint64_t result;
  stb_compress_dxt_block(reinterpret_cast<unsigned char *>(&result),
                         block, 0, STB_DXT_HIGHQUAL);
  return result;
}

static uint64_t CompressRGBA(const uint8_t *img, int width) {
  unsigned char block[64];
  memset(block, 0, sizeof(block));

  for (int j = 0; j < 4; ++j) {
    memcpy(block + j*16, img + width*4*j, 16);
  }

  uint64_t result;
  stb_compress_dxt_block(reinterpret_cast<unsigned char *>(&result),
                         block, 0, STB_DXT_HIGHQUAL);
  return result;
}

union DXTBlock {
  struct {
    uint16_t ep1;
    uint16_t ep2;
    uint32_t interpolation;
  };
  uint64_t dxt_block;
};

uint32_t Cvt565(uint16_t x) {
  uint32_t r = (x >> 11);
  r = (r << 3) | (r >> 2);

  uint32_t g = (x >> 5) & 0x3F;
  g = (g << 2) | (g >> 4);

  uint32_t b = x & 0x1F;
  b = (b << 3) | (b >> 2);

  return 0xFF000000 | (b << 16) | (g << 8) | r;
}

void encode(const cv::Mat &img, std::vector<uint8_t> *result) {
  // Collect the average of all the DC coefficients...
  uint32_t avg_dc = 0;
  uint32_t num_dcs = 0;
  for (int j = 0; j < img.rows; j += 8) {
    for (int i = 0; i < img.cols; i += 8) {
      avg_dc += img.at<int16_t>(j, i);
      num_dcs++;
    }
  }

  avg_dc /= num_dcs;

  // Collect stats for frequency analysis
  int16_t min_coeff = std::numeric_limits<int16_t>::max();
  int16_t max_coeff = std::numeric_limits<int16_t>::min();
  int32_t num_outliers = 0;
  int32_t num_zeros = 0;
  for (int j = 0; j < img.rows; j++) {
    for (int i = 0; i < img.cols; i++) {
      int16_t coeff = img.at<int16_t>(j, i);
      if (i % 8 == 0 && j % 8 == 0) {
        coeff = avg_dc - coeff;
      }

      if (coeff < -128 || coeff > 127) {
        num_outliers++;
        continue;
      }

      if (coeff == 0) {
        num_zeros++;
        continue;
      }

      min_coeff = std::min(min_coeff, coeff);
      max_coeff = std::max(max_coeff, coeff);
    }
  }

  std::cout << "Total symbols: " << img.cols * img.rows << std::endl;
  std::cout << "Num outliers: " << num_outliers << std::endl;
  std::cout << "Num zeros: " << num_zeros << std::endl;
  std::cout << "Min coefficient: " << min_coeff << std::endl;
  std::cout << "Max coefficient: " << max_coeff << std::endl << std::endl;
}

void quantize(cv::Mat *dct, bool is_chroma) {
  static cv::Mat quant_table_luma = (cv::Mat_<int16_t>(8, 8) <<
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99);

  static cv::Mat quant_table_chroma = (cv::Mat_<int16_t>(8, 8) <<
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99);

  for (int j = 0; j < dct->rows / 8; ++j) {
    for (int i = 0; i < dct->cols / 8; ++i) {
      cv::Rect_<int> window(i * 8, j * 8, 8, 8);
      cv::Mat block = (*dct)(window).clone();

      cv::divide(block, is_chroma ? quant_table_chroma : quant_table_luma, block);

      block.copyTo((*dct)(window));
    }
  }

  static int quantized_copy = 0;
  char fname[256];
  sprintf(fname, "quantized_%d.png", quantized_copy);
  cv::imwrite(fname, *dct);
  quantized_copy++;
}

void compressChannel(const cv::Mat &img, std::vector<uint8_t> *result, bool is_chroma) {
  assert(img.channels() == 1);

  // DCT
  cv::Mat dct_img = img.clone();
  dct::RunDCT(&dct_img);

  assert(img.type() == CV_16SC1);

  // Quantize
  quantize(&dct_img, is_chroma);

  // Encode
  encode(dct_img, result);
}

std::vector<uint8_t> compress(const cv::Mat &img) {
  cv::Mat img_YCrCb;
  cv::cvtColor(img, img_YCrCb, CV_RGB2YCrCb);

  cv::Mat channels[3];
  cv::split(img_YCrCb, channels);

  // Subsample chroma...
  cv::resize(channels[1], channels[1], cv::Size((img.cols + 1) / 2, (img.rows + 1) / 2));
  cv::resize(channels[2], channels[2], cv::Size((img.cols + 1) / 2, (img.rows + 1) / 2));

  std::vector<uint8_t> result;
  result.resize(8);

  uint32_t *resultBuf = reinterpret_cast<uint32_t *>(result.data());
  resultBuf[0] = img.cols;
  resultBuf[1] = img.rows;

  compressChannel(channels[0], &result, false);
  compressChannel(channels[1], &result, true);
  compressChannel(channels[2], &result, true);

  return std::move(result);
}

cv::Mat decompress(const std::vector<uint8_t> &stream) {
  assert(!"unimplemented");
  return cv::Mat();
}

cv::Mat codec(const cv::Mat &img) {
  return decompress(compress(img));
}

int main(int argc, char **argv) {
  // Make sure that we have the proper number of arguments...
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  // Otherwise, load the file
  const cv::Mat img = cv::imread(argv[1], -1);
  if (!img.data) {
    std::cerr << "Error loading image: " << argv[1] << std::endl;
    return 1;
  }

  assert((img.cols & 0x3) == 0);
  assert((img.rows & 0x3) == 0);

  const int num_blocks_x = (img.cols + 3) / 4;
  const int num_blocks_y = (img.rows + 3) / 4;
  const int num_blocks = num_blocks_x * num_blocks_y;

  // Now do the dxt compression...
  std::vector<DXTBlock> dxt_blocks(num_blocks);

  for (int j = 0; j < img.rows; j += 4) {
    for (int i = 0; i < img.cols; i += 4) {
      int block_idx = (j / 4) * num_blocks_x + (i / 4);
      const unsigned char *offset_data = img.ptr(j) + i * img.channels();
      if (3 == img.channels()) {
        dxt_blocks[block_idx].dxt_block = CompressRGB(offset_data, img.cols);
      } else if (4 == img.channels()) {
        dxt_blocks[block_idx].dxt_block = CompressRGBA(offset_data, img.cols);
      } else {
        std::cerr << "Error! Only accepts RGB or RGBA images!" << std::endl;
      }
    }
  }

  // !FIXME! Find structure in interpolation values
  // Aidos this is where you work your magic.

  // Collect stats...
  size_t num_zero = 0;
  size_t num_one = 0;
  size_t num_two = 0;
  size_t num_three = 0;
  for (int i = 0; i < num_blocks; ++i) {
    uint32_t interp = dxt_blocks[i].interpolation;
    for (int j = 0; j < 16; ++j) {
      switch (interp & 0x3) {
        case 0: num_zero++; break;
        case 1: num_one++; break;  
        case 2: num_two++; break;
        case 3: num_three++; break;  
      }
      interp >>= 2;
    }
  }

  std::cout << "Number of 0 interpolation values: " << num_zero << std::endl;
  std::cout << "Number of 1 interpolation values: " << num_one << std::endl;
  std::cout << "Number of 2 interpolation values: " << num_two << std::endl;
  std::cout << "Number of 3 interpolation values: " << num_three << std::endl;

  // Visualize data...
  uint8_t interp_map[4] = { 0, 85, 170, 255 };
  cv::Mat interp_vis(img.rows, img.cols, CV_8UC1);
  for (int j = 0; j < img.rows; j+=4) {
    for (int i = 0; i < img.cols; i+=4) {
      int block_idx = (j / 4) * num_blocks_x + (i / 4);
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          int local_pixel_idx = y * 4 + x;
          int interp = (dxt_blocks[block_idx].interpolation >> local_pixel_idx) & 0x3;
          interp_vis.at<uint8_t>(j + y, i + x) = interp_map[interp];
        }
      }
    }
  }

  cv::imwrite("dxt_interp.png", interp_vis);

  cv::Mat img_A(num_blocks_y, num_blocks_x, CV_8UC4);
  cv::Mat img_B(num_blocks_y, num_blocks_x, CV_8UC4);
  for (int j = 0; j < num_blocks_y; j ++) {
    for (int i = 0; i < num_blocks_x; i ++) {
      int block_idx = j * num_blocks_x + i;
      img_A.at<uint32_t>(j, i) = Cvt565(dxt_blocks[block_idx].ep1);
      img_B.at<uint32_t>(j, i) = Cvt565(dxt_blocks[block_idx].ep2);
    }
  }

  cv::imwrite("dxt_img_A.png", img_A);
  cv::imwrite("dxt_img_B.png", img_B);

  cv::Mat decomp_A = codec(img_A);
  cv::Mat decomp_B = codec(img_B);

  cv::Mat img_A_YCrCb, img_B_YCrCb;
  cv::cvtColor(img_A, img_A_YCrCb, CV_RGB2YCrCb);
  cv::cvtColor(img_B, img_B_YCrCb, CV_RGB2YCrCb);

  cv::Mat channels[2][3];
  cv::split(img_A_YCrCb, channels[0]);
  cv::split(img_B_YCrCb, channels[1]);

#ifdef USE_FAST_DCT
  bool is_fast = true;
#else
  bool is_fast = false;
#endif

  cv::imwrite("dxt_img_A_Y.png", channels[0][0]);

  dct::RunDCT(channels[0]);
  cv::imwrite(is_fast ? "dxt_img_A_Y_fast_dct.png" : "dxt_img_A_Y_dct.png", channels[0][0]);
  dct::RunIDCT(channels[0]);
  cv::imwrite(is_fast ? "dxt_img_A_Y_fast_idct.png" : "dxt_img_A_Y_idct.png", channels[0][0]);

  cv::imwrite("dxt_img_A_Cr.png", channels[0][1]);
  cv::imwrite("dxt_img_A_Cb.png", channels[0][2]);

  dct::RunDCT(channels[1]);
  cv::imwrite(is_fast ? "dxt_img_B_Y_fast_dct.png" : "dxt_img_B_Y_dct.png", channels[1][0]);
  dct::RunIDCT(channels[1]);
  cv::imwrite(is_fast ? "dxt_img_B_Y_fast_idct.png" : "dxt_img_B_Y_idct.png", channels[1][0]);

  cv::imwrite("dxt_img_B_Cr.png", channels[1][1]);
  cv::imwrite("dxt_img_B_Cb.png", channels[1][2]);

  return 0;
}
