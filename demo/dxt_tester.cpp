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

  cv::Mat img_A_YCrCb, img_B_YCrCb;
  cv::cvtColor(img_A, img_A_YCrCb, CV_RGB2YCrCb);
  cv::cvtColor(img_B, img_B_YCrCb, CV_RGB2YCrCb);

  cv::Mat channels[3];
  cv::split(img_A_YCrCb, channels);

  cv::imwrite("dxt_img_A_Y.png", channels[0]);
  dct::RunDCT(channels);
#ifdef USE_FAST_DCT
  cv::imwrite("dxt_img_A_Y_fast_dct.png", channels[0]);
#endif
  dct::RunIDCT(channels);
#ifdef USE_FAST_DCT
  cv::imwrite("dxt_img_A_Y_fast_idct.png", channels[0]);
#endif

  cv::imwrite("dxt_img_A_Cr.png", channels[1]);
  cv::imwrite("dxt_img_A_Cb.png", channels[2]);

  cv::split(img_B_YCrCb, channels);

  cv::imwrite("dxt_img_B_Y.png", channels[0]);
  cv::imwrite("dxt_img_B_Cr.png", channels[1]);
  cv::imwrite("dxt_img_B_Cb.png", channels[2]);

  return 0;
}
