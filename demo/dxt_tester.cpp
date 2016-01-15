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

#include "histogram.h"
#include "ans_ocl.h"

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

static const uint32_t kNumStreams = 16;
void encode(const cv::Mat &img, std::vector<uint8_t> *result) {
  // Collect stats for frequency analysis
  std::vector<int16_t> coeffs(img.rows * img.cols, 0);
  assert(coeffs.size() % (256 * 16) == 0);

  int16_t min_coeff = std::numeric_limits<int16_t>::max();
  int16_t max_coeff = std::numeric_limits<int16_t>::min();
  int32_t num_outliers = 0;
  int32_t num_zeros = 0;
  for (int j = 0; j < img.rows; j++) {
    for (int i = 0; i < img.cols; i++) {
      int16_t coeff = img.at<int16_t>(j, i);
      if (coeff < -127 || coeff > 127) {
        num_outliers++;
      }

      if (coeff == 0) {
        num_zeros++;
      }

      min_coeff = std::min(min_coeff, coeff);
      max_coeff = std::max(max_coeff, coeff);

      coeffs[j*img.cols + i] = coeff;
    }
  }

  std::vector<uint8_t> symbols;
  symbols.reserve(coeffs.size());

  for (const auto coeff : coeffs) {
    if (coeff < -127 || coeff > 127) {
      symbols.push_back(0);
    } else {
      symbols.push_back(static_cast<uint8_t>(coeff + 128));
    }
  }

  const std::vector<int> counts = std::move(ans::CountSymbols(symbols));
  assert(counts.size() == 256);

  std::vector<uint8_t> encoded_symbols(256, 0);
  std::vector<int> encoded_counts;
  encoded_counts.reserve(256);

  uint32_t sym_idx = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    if (counts[i] == 0) {
      continue;
    }

    encoded_symbols[i] = sym_idx++;
    encoded_counts.push_back(counts[i]);
  }

  size_t bytes_written = result->size();
  result->resize(bytes_written + 2);
  *reinterpret_cast<uint16_t *>(result->data() + bytes_written) =
    static_cast<uint16_t>(encoded_counts.size());
  bytes_written += 2;

  for (uint32_t i = 0; i < 256; ++i) {
    if (counts[i] == 0) {
      continue;
    }

    result->resize(bytes_written + 3);

    uint8_t *symbol_ptr = result->data() + bytes_written;
    uint16_t *count_ptr = reinterpret_cast<uint16_t *>(symbol_ptr + 1);

    *symbol_ptr = static_cast<uint8_t>(i);
    *count_ptr = static_cast<uint16_t>(counts[i]);
    bytes_written += 3;
  }

  std::cout << std::endl << "Total symbols: " << img.cols * img.rows << std::endl;
  std::cout << "Num outliers: " << num_outliers << std::endl;
  std::cout << "Num zeros: " << num_zeros << std::endl;
  std::cout << "Num unique symbols: " << encoded_counts.size() << std::endl;
  std::cout << "Min coefficient: " << min_coeff << std::endl;
  std::cout << "Max coefficient: " << max_coeff << std::endl;

  for (uint32_t i = 0; i < counts.size(); ++i) {
    std::cout << counts[i] << ", ";
    if ((i + 1) % 16 == 0) {
      std::cout << std::endl;
    }
  }

  // Write counts to output
  std::vector<ans::OpenCLEncoder> encoders;
  encoders.reserve(kNumStreams);
  for (uint32_t i = 0; i < kNumStreams; ++i) {
    encoders.push_back(std::move(ans::OpenCLEncoder(encoded_counts)));
  }

  std::vector<uint8_t> encoded(10, 0);
  uint32_t encoded_bytes_written = 0;
  uint32_t last_encoded_bytes_written = 0;

  uint32_t symbol_offset = 0;
  while(symbol_offset < symbols.size()) {
    for (uint32_t sym_idx = 0; sym_idx < ans::kNumEncodedSymbols; ++sym_idx) {
      // Make sure that we have at least 4*kNumStreams bytes available
      encoded.resize(encoded_bytes_written + (4*kNumStreams));

      int16_t *output = reinterpret_cast<int16_t *>(encoded.data() + encoded_bytes_written);
      for (uint32_t strm_idx = 0; strm_idx < kNumStreams; strm_idx++) {
        uint32_t sidx = symbol_offset + strm_idx * ans::kNumEncodedSymbols + sym_idx;
        if (symbols[sidx] == 0) {
          *output = coeffs[sym_idx];
          output++;
          encoded_bytes_written += 2;
        }
      }

      for (uint32_t strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        ans::BitWriter w(encoded.data() + encoded_bytes_written);
        uint32_t sidx = symbol_offset + strm_idx * ans::kNumEncodedSymbols + sym_idx;
        uint8_t symbol = encoded_symbols[symbols[sidx]];

        assert(symbol < encoded_counts.size());
        assert(counts[symbols[sidx]] > 0);

        encoders[strm_idx].Encode(symbol, w);
        encoded_bytes_written += w.BytesWritten();
      }
    }

    // Dump all of the encoder states
    encoded.resize(encoded_bytes_written + 4*kNumStreams);

    uint32_t *encoder_state = reinterpret_cast<uint32_t *>(encoded.data() + encoded_bytes_written);
    for (uint32_t i = 0; i < kNumStreams; ++i) {
      encoder_state[i] = encoders[i].GetState();
    }
    encoded_bytes_written += 4 * kNumStreams;

    // Add the offset to the stream...
    uint32_t offset = encoded_bytes_written - last_encoded_bytes_written;
    result->resize(bytes_written + 2);
    *reinterpret_cast<uint16_t *>(result->data() + bytes_written) = static_cast<uint16_t>(offset);
    bytes_written += 2;
    assert(offset <= ((1 << 16) - 1));
    last_encoded_bytes_written = encoded_bytes_written;

    // Advance the symbol offset...
    symbol_offset += kNumStreams * ans::kNumEncodedSymbols;
  }

  // Add the encoded bytes
  result->resize(bytes_written + encoded_bytes_written);
  memcpy(result->data() + bytes_written, encoded.data(), encoded_bytes_written);
}

static const cv::Mat quant_table_luma = (cv::Mat_<int16_t>(8, 8) <<
  16, 11, 10, 16, 24, 40, 51, 61,
  12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103, 99);

static const cv::Mat quant_table_chroma = (cv::Mat_<int16_t>(8, 8) <<
  17, 18, 24, 47, 99, 99, 99, 99,
  18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99,
  47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99);

void quantize(cv::Mat *dct, bool is_chroma) {
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

void dequantize(cv::Mat *dct, bool is_chroma) {
  for (int j = 0; j < dct->rows / 8; ++j) {
    for (int i = 0; i < dct->cols / 8; ++i) {
      cv::Rect_<int> window(i * 8, j * 8, 8, 8);
      cv::Mat block = (*dct)(window).clone();

      cv::multiply(block, is_chroma ? quant_table_chroma : quant_table_luma, block);

      block.copyTo((*dct)(window));
    }
  }

  static int quantized_copy = 0;
  char fname[256];
  sprintf(fname, "dequantized_%d.png", quantized_copy);
  cv::imwrite(fname, *dct);
  quantized_copy++;
}

void compressChannel(const cv::Mat &img, std::vector<uint8_t> *result, bool is_chroma) {
  assert(img.channels() == 1);

  // DCT
  cv::Mat dct_img = img.clone();
  dct::RunDCT(&dct_img);

  assert(dct_img.type() == CV_16SC1);

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

  std::cout << "Uncompressed size: " << img.cols * img.rows * 2 << std::endl;
  std::cout << "Compressed size: " << result.size() << std::endl;

  return std::move(result);
}

int decode(cv::Mat *result, const uint8_t *buf) {
  int offset = 0;

  uint16_t num_symbols = *reinterpret_cast<const uint16_t *>(buf + offset);
  offset += 2;

  std::vector<uint8_t> symbols(num_symbols, 0);
  std::vector<int> counts(num_symbols, 0);

  for (uint32_t i = 0; i < num_symbols; ++i) {
    symbols[i] = *reinterpret_cast<const uint8_t *>(buf + offset);
    counts[i] = static_cast<int>(*reinterpret_cast<const uint16_t *>(buf + offset + 1));
    offset += 3;
  }

  int num_macroblocks = (result->cols * result->rows) / (kNumStreams * ans::kNumEncodedSymbols);
  assert(num_macroblocks * kNumStreams * ans::kNumEncodedSymbols == result->cols * result->rows);
  std::vector<uint16_t> macroblock_offsets(num_macroblocks);

  for (int i = 0; i < num_macroblocks; ++i) {
    macroblock_offsets[i] = *reinterpret_cast<const uint16_t *>(buf + offset);
    offset += 2;
  }

  std::vector<int16_t> coeffs(result->cols * result->rows);

  int symbol_offset = 0;
  int last_offset = 0;
  for (uint16_t mb_off : macroblock_offsets) {
    int mb_size = mb_off - last_offset;

    const uint32_t *states = reinterpret_cast<const uint32_t *>(buf + offset + mb_off) - kNumStreams;
    std::vector<ans::OpenCLCPUDecoder> decoders;
    decoders.reserve(kNumStreams);
    for (uint32_t i = 0; i < kNumStreams; ++i) {
      decoders.push_back(ans::OpenCLCPUDecoder(states[kNumStreams - i - 1], counts));
    }

    int data_sz_bytes = mb_size - 4 * kNumStreams;
    assert(data_sz_bytes % 2 == 0);

    std::vector<uint16_t> mb_data(data_sz_bytes / 2);
    memcpy(mb_data.data(), buf + offset + last_offset, data_sz_bytes);
    std::reverse(mb_data.begin(), mb_data.end());

    ans::BitReader r(reinterpret_cast<const uint8_t *>(mb_data.data()));
    for (int sym_idx = 0; sym_idx < ans::kNumEncodedSymbols; sym_idx++) {
      for (int strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        uint32_t sidx = symbol_offset + (strm_idx + 1) * ans::kNumEncodedSymbols - sym_idx - 1;
        coeffs[sidx] = static_cast<int16_t>(decoders[strm_idx].Decode(r)) - 128;
      }

      size_t coeffs_sz = coeffs.size();
      for (int strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        uint32_t idx = symbol_offset + (strm_idx + 1) * ans::kNumEncodedSymbols - sym_idx - 1;
        if (coeffs[idx] == -128) {
          coeffs[idx] = static_cast<int16_t>(r.ReadBits(16));
        }
      }
    }

    last_offset = mb_off;
    symbol_offset += kNumStreams * ans::kNumEncodedSymbols;
  }

  offset += macroblock_offsets[num_macroblocks - 1];

  // Populate the image properly
  assert(result->type() == CV_16SC1);
  uint32_t coeff_idx = 0;
  for (int j = 0; j < result->rows; ++j) {
    for (int i = 0; i < result->cols; ++i) {
      result->at<int16_t>(j, i) = coeffs[coeff_idx++];
    }
  }

  return offset;
}

int decompressChannel(cv::Mat *result, const uint8_t *buf, int width, int height, bool is_chroma) {
  int num_symbols = width * height;

  *result = cv::Mat(height, width, CV_16SC1);
  int offset = decode(result, buf);

  dequantize(result, is_chroma);
  dct::RunIDCT(result);

  return offset;
}

cv::Mat decompress(const std::vector<uint8_t> &stream) {
  const uint8_t *stream_buf = stream.data();

  const uint32_t *size_buf = reinterpret_cast<const uint32_t *>(stream_buf);
  int width = size_buf[0];
  int height = size_buf[1];

  int num_symbols_luma = width * height;
  int num_symbols_chroma = ((width + 1) / 2) * ((height + 1) / 2);

  int offset = 8;

  cv::Mat channels[3];
  offset += decompressChannel(channels + 0, stream_buf + offset, width, height, false);
  offset += decompressChannel(channels + 1, stream_buf + offset, (width + 1) / 2, (height + 1) / 2, true);
  offset += decompressChannel(channels + 2, stream_buf + offset, (width + 1) / 2, (height + 1) / 2, true);

  // Resize chroma to full...
  cv::resize(channels[1], channels[1], cv::Size(width, height));
  cv::resize(channels[2], channels[2], cv::Size(width, height));

  cv::Mat img_YCrCb;
  cv::merge(channels, 3, img_YCrCb);

  cv::Mat result;
  cv::cvtColor(img_YCrCb, result, CV_YCrCb2RGB);
  return result;
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

  cv::imwrite("dxt_img_codec_A.png", decomp_A);
  cv::imwrite("dxt_img_codec_B.png", decomp_B);

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
