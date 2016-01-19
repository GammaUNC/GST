#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

// #define VERBOSE
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

uint32_t From565(uint16_t x) {
  uint32_t r = (x >> 11);
  r = (r << 3) | (r >> 2);

  uint32_t g = (x >> 5) & 0x3F;
  g = (g << 2) | (g >> 4);

  uint32_t b = x & 0x1F;
  b = (b << 3) | (b >> 2);

  return 0xFF000000 | (b << 16) | (g << 8) | r;
}

uint16_t Into565(uint8_t r, uint8_t g, uint8_t b) {
  uint16_t rr = (r >> 3) & 0x1F;
  uint16_t gg = (g >> 2) & 0x3F;
  uint16_t bb = (b >> 3) & 0x1F;

  return (rr << 11) | (gg << 5) | bb;
}

union DXTBlock {
  struct {
    uint16_t ep1;
    uint16_t ep2;
    uint32_t interpolation;
  };
  uint64_t dxt_block;
};

uint32_t LerpChannels(uint32_t a, uint32_t b, int num, int div) {
  uint8_t *a_ptr = reinterpret_cast<uint8_t *>(&a);
  uint8_t *b_ptr = reinterpret_cast<uint8_t *>(&b);

  uint32_t result;
  uint8_t *result_ptr = reinterpret_cast<uint8_t *>(&result);
  for (int i = 0; i < 4; ++i) {
    result_ptr[i] = (static_cast<int>(a_ptr[i]) * (div - num) + static_cast<int>(b_ptr[i]) * num) / div;
  }
  return result;
}

void get_dxt_palette(const DXTBlock &block, uint8_t out[12]) {
  // unpack the endpoints
  uint32_t palette[4];
  palette[0] = From565(block.ep1);
  palette[1] = From565(block.ep2);

  if (block.ep1 <= block.ep2) {
    palette[2] = LerpChannels(palette[0], palette[1], 1, 2);
    palette[3] = 0;
  }
  else {
    palette[2] = LerpChannels(palette[0], palette[1], 1, 3);
    palette[3] = LerpChannels(palette[0], palette[1], 2, 3);
  }

  for (int i = 0; i < 4; ++i) {
    out[3 * i + 0] = palette[i] & 0xFF;
    out[3 * i + 1] = (palette[i] >> 8) & 0xFF;
    out[3 * i + 2] = (palette[i] >> 16) & 0xFF;
  }
}

cv::Mat DecompressDXTBlock(const DXTBlock &block) {
  cv::Mat result(4, 4, CV_8UC4);

  uint8_t palette[12];
  get_dxt_palette(block, palette);

  // unpack the indices
  uint8_t const* bytes = reinterpret_cast<const uint8_t *>(&block);
  uint8_t indices[16];
  for (int k = 0; k < 4; ++k) {
    uint8_t packed = bytes[4 + k];

    indices[0 + 4 * k] = packed & 0x3;
    indices[1 + 4 * k] = (packed >> 2) & 0x3;
    indices[2 + 4 * k] = (packed >> 4) & 0x3;
    indices[3 + 4 * k] = (packed >> 6) & 0x3;
  }

  // store out the colours
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      uint8_t offset = indices[y * 4 + x];
      uint32_t pixel = 0xFF000000;
      pixel |= palette[3 * offset + 0];
      pixel |= (static_cast<uint32_t>(palette[3 * offset + 1]) << 8);
      pixel |= (static_cast<uint32_t>(palette[3 * offset + 2]) << 16);
      result.at<uint32_t>(y, x) = pixel;
    }
  }

  return result;
}

void get_dxt_color_at(const std::vector<DXTBlock> &blocks, int x, int y, int width, uint8_t out[3]) {
  int block_idx = (y / 4) * (width / 4) + (x / 4);
  cv::Mat block = DecompressDXTBlock(blocks[block_idx]);

  int i = x % 4;
  int j = y % 4;

  uint32_t c = block.at<uint32_t>(j, i);
  out[0] = c & 0xFF;
  out[1] = (c >> 8) & 0xFF;
  out[2] = (c >> 16) & 0xFF;
}

cv::Mat DecompressDXT(const std::vector<DXTBlock> &blocks, int width, int height) {
  cv::Mat result(height, width, CV_8UC4);

  int block_idx = 0;
  for (int j = 0; j < height; j += 4) {
    for (int i = 0; i < width; i += 4) {
      cv::Mat block = DecompressDXTBlock(blocks[block_idx]);
      block.copyTo(result(cv::Rect_<int>(i, j, 4, 4)));
      block_idx++;
    }
  }

  return result;
}

static const int coeff_offset = 512;
static const uint32_t kNumStreams = 16;
void encode(const cv::Mat &img, std::vector<uint8_t> *result) {
  // Collect stats for frequency analysis
  std::vector<int16_t> coeffs(img.rows * img.cols, 0);
  assert(coeffs.size() % (256 * 16) == 0);

  int16_t min_coeff = std::numeric_limits<int16_t>::max();
  int16_t max_coeff = std::numeric_limits<int16_t>::min();
  int32_t num_outliers = 0;
  int32_t num_zeros = 0;

#ifdef VERBOSE
  std::cout << std::endl << "First 16 encoded values: ";
#endif // Verbose

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

      int coeff_idx = j*img.cols + i;
#ifdef VERBOSE
      if (coeff_offset <= coeff_idx && coeff_idx < coeff_offset + 16) {
        std::cout << coeff << ", ";
        if (coeff_idx == coeff_offset + 15) {
          std::cout << std::endl;
        }
      }
#endif // VERBOSE

      coeffs[coeff_idx] = coeff;
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
#ifdef VERBOSE
  std::cout << "Total symbols: " << img.cols * img.rows << std::endl;
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
#endif  //  VERBOSE

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
          *output = coeffs[sidx];
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
  assert(num_macroblocks * kNumStreams * ans::kNumEncodedSymbols == 
         static_cast<uint32_t>(result->cols * result->rows));
  std::vector<uint16_t> macroblock_sizes(num_macroblocks);

  for (int i = 0; i < num_macroblocks; ++i) {
    macroblock_sizes[i] = *reinterpret_cast<const uint16_t *>(buf + offset);
    offset += 2;
  }

  std::vector<int16_t> coeffs(result->cols * result->rows);

  int symbol_offset = 0;
  for (uint16_t mb_size : macroblock_sizes) {
    int mb_off = offset + mb_size;

    const uint32_t *states = reinterpret_cast<const uint32_t *>(buf + mb_off) - kNumStreams;
    std::vector<ans::OpenCLCPUDecoder> decoders;
    decoders.reserve(kNumStreams);
    for (uint32_t i = 0; i < kNumStreams; ++i) {
      decoders.push_back(ans::OpenCLCPUDecoder(states[kNumStreams - i - 1], counts));
    }

    int data_sz_bytes = mb_size - 4 * kNumStreams;
    assert(data_sz_bytes % 2 == 0);

    std::vector<uint16_t> mb_data(data_sz_bytes / 2);
    memcpy(mb_data.data(), buf + offset, data_sz_bytes);
    std::reverse(mb_data.begin(), mb_data.end());

    ans::BitReader r(reinterpret_cast<const uint8_t *>(mb_data.data()));
    for (uint32_t sym_idx = 0; sym_idx < ans::kNumEncodedSymbols; sym_idx++) {
      for (uint32_t strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        uint32_t sidx = symbol_offset + (kNumStreams - strm_idx) * ans::kNumEncodedSymbols - sym_idx - 1;
        coeffs[sidx] = static_cast<int16_t>(symbols[decoders[strm_idx].Decode(r)]) - 128;
      }

      for (uint32_t strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        uint32_t idx = symbol_offset + (kNumStreams - strm_idx) * ans::kNumEncodedSymbols - sym_idx - 1;
        if (coeffs[idx] == -128) {
          coeffs[idx] = static_cast<int16_t>(r.ReadBits(16));
        }
      }
    }

    offset = mb_off;
    symbol_offset += kNumStreams * ans::kNumEncodedSymbols;
  }

  // Populate the image properly
  assert(result->type() == CV_16SC1);
  uint32_t coeff_idx = 0;
#ifdef VERBOSE
  std::cout << "First 16 decoded values: ";
#endif
  for (int j = 0; j < result->rows; ++j) {
    for (int i = 0; i < result->cols; ++i) {
#ifdef VERBOSE
      if (coeff_offset <= coeff_idx && coeff_idx < coeff_offset + 16) {
        std::cout << coeffs[coeff_idx] << ", ";
        if (coeff_idx == coeff_offset + 15) {
          std::cout << std::endl;
        }
      }
#endif  // VERBOSE
      result->at<int16_t>(j, i) = coeffs[coeff_idx++];
    }
  }

  return offset;
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
#ifdef VERBOSE
  static int prequantized_copy = 0;
  char fname[256];
  sprintf(fname, "prequantized_%d.png", prequantized_copy);
  cv::imwrite(fname, *dct);
  prequantized_copy++;
#endif // VERBOSE

  for (int j = 0; j < dct->rows / 8; ++j) {
    for (int i = 0; i < dct->cols / 8; ++i) {
      cv::Rect_<int> window(i * 8, j * 8, 8, 8);
      cv::Mat block = (*dct)(window).clone();

      cv::divide(block, is_chroma ? quant_table_chroma : quant_table_luma, block);

      block.copyTo((*dct)(window));
    }
  }

#ifdef VERBOSE
  static int quantized_copy = 0;
  sprintf(fname, "quantized_%d.png", quantized_copy);
  cv::imwrite(fname, *dct);
  quantized_copy++;
#endif // VERBOSE
}

void dequantize(cv::Mat *dct, bool is_chroma) {
#ifdef VERBOSE
  static int predequantized_copy = 0;
  char fname[256];
  sprintf(fname, "predequantized_%d.png", predequantized_copy);
  cv::imwrite(fname, *dct);
  predequantized_copy++;
#endif // VERBOSE

  for (int j = 0; j < dct->rows / 8; ++j) {
    for (int i = 0; i < dct->cols / 8; ++i) {
      cv::Rect_<int> window(i * 8, j * 8, 8, 8);
      cv::Mat block = (*dct)(window).clone();

      cv::multiply(block, is_chroma ? quant_table_chroma : quant_table_luma, block);

      block.copyTo((*dct)(window));
    }
  }

#ifdef VERBOSE
  static int dequantized_copy = 0;
  sprintf(fname, "dequantized_%d.png", dequantized_copy);
  cv::imwrite(fname, *dct);
  dequantized_copy++;
#endif // VERBOSE
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

int decompressChannel(cv::Mat *result, const uint8_t *buf, int width, int height, bool is_chroma) {
  *result = cv::Mat(height, width, CV_16SC1);
  int offset = decode(result, buf);

  dequantize(result, is_chroma);
  dct::RunIDCT(result);

  return offset;
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

#ifdef VERBOSE
  std::cout << "Endpoint Image uncompressed size: " << img.cols * img.rows * 2 << std::endl;
  std::cout << "Endpoint Image compressed size: " << result.size() << std::endl;
#endif

  return std::move(result);
}

cv::Mat decompress(const std::vector<uint8_t> &stream) {
  const uint8_t *stream_buf = stream.data();

  const uint32_t *size_buf = reinterpret_cast<const uint32_t *>(stream_buf);
  int width = size_buf[0];
  int height = size_buf[1];

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

  cv::Mat result(height, width, CV_8UC4);
  cv::cvtColor(img_YCrCb, result, CV_YCrCb2RGB);
  return result;
}

void get_indices_from_block(DXTBlock block, int *indices) {
  for(int i = 0; i < 16; i++) {
    indices[i] = (block.interpolation >> (i*2)) & 3;
  }
}

uint8_t get_gray(const uint8_t color[]) {
  int gray;

  // choose one of the following representations of gray value

  // Average
  gray = static_cast<int> ((color[0]+color[1]+color[2])/3);
  // Lightness
  // gray = static_cast<int> ((std::max({color[0],color[1],color[2]})
  //                          + std::min({color[0],color[1],color[2]}))/2);
  // Luminosity
  // gray = static_cast<int> (0.21*color[0]+0.72*color[1]+0.07*color[2]);
  // Green channel
  // gray = color[1];

  if (gray < 0 || gray > 255) {
    std::cout << "ERROR: ------ Gray value overflow: " << gray << std::endl;
    exit(1);
  }

  uint8_t  eight_bit_gray = gray;
  return eight_bit_gray;
}

void predict_color(const uint8_t diag[], const uint8_t upper[],
                   const uint8_t left[], uint8_t *predicted) {
  uint8_t gray_diag, gray_upper, gray_left;
  gray_diag  = get_gray(diag);
  gray_upper = get_gray(upper);
  gray_left  = get_gray(left);

  uint8_t mb = std::abs(gray_diag - gray_upper);
  uint8_t mc = std::abs(gray_diag - gray_left);
  uint8_t ma = std::abs(mb - mc);

  int temp[3];

  for(int i=0;i<3;i++) {
    if ((ma < 4) && (mb < 4))
      temp[i] = left[i] + upper[i] - diag[i];
    else if (ma < 10)
      temp[i] = (left[i] + upper[i])/2;
    else if (ma < 64) {
      if (mb < mc)
        temp[i] = (3*left[i] + upper[i])/4;
      else
        temp[i] = (left[i] + 3*upper[i])/4;
      }
    else {
      if (mb < mc)
        temp[i] = left[i];
      else
        temp[i] = upper[i];
    }
  } // for

  for(int i=0;i<3;i++) {
    if (temp[i] < 0)
      temp[i] = 0;
    else if (temp[i] > 255)
      temp[i] = 255;

    predicted[i] = temp[i];
  }
}

int distance(uint8_t *colorA, uint8_t *colorB) {
  // abs of gray values
  int gray_a = get_gray(colorA);
  int gray_b = get_gray(colorB);
  int distance;

  // choose one of the following distances:

  // absolute value
  // distance = std::abs(gray_a - gray_b);
  // sum of abs
  // distance = std::abs(colorA[0]-colorB[0])
  //     + std::abs(colorA[1]-colorB[1]) + std::abs(colorA[2]-colorB[2]);
  // sum of sqaures
  distance = (colorA[0]-colorB[0])*(colorA[0]-colorB[0])
      + (colorA[1]-colorB[1])*(colorA[1]-colorB[1])
      + (colorA[2]-colorB[2])*(colorA[2]-colorB[2]);

  return distance;
}

int predict_index(uint8_t *colors, uint8_t *predicted_color) {
  int difference[4];

  for(int i=0;i<4;i++)
    difference[i] = distance(predicted_color, &colors[i*3]);

  int min = difference[0];
  int min_id = 0;
  for(int i=1;i<4;i++) {
    if (min > difference[i]) {
      min = difference[i];
      min_id = i;
    }
  }

  return min_id;
}

std::vector<uint8_t> symbolize_indices(const std::vector<DXTBlock> &blocks,
                                       const std::vector<uint8_t> &indices,
                                       int width, int height) {
  assert(indices.size() == width * height);
  assert(blocks.size() == indices.size() / 16);

  // Operate in 16-block chunks arranged as 4x4 blocks
  assert(width % 16 == 0);
  assert(height % 16 == 0);

  const int num_blocks_x = (width + 3) / 4;

  std::vector<uint8_t> symbols;
  symbols.reserve(indices.size());

  for (int chunk_j = 0; chunk_j < height; chunk_j += 16) {
    for (int chunk_i = 0; chunk_i < width; chunk_i += 16) {
      // For each chunk, go through and leave the top row
      // and left-most column unpredicted
      for (int block_j = 0; block_j < 16; block_j += 4) {
        for (int block_i = 0; block_i < 16; block_i += 4) {
          
          // In each block, push back the symbols one by one..
          for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
              int chunk_coord_y = block_j + j;
              int chunk_coord_x = block_i + i;

              int px = chunk_j + block_j + j;
              int py = chunk_i + block_i + i;

              int pixel_index = py * width + px;
              if (chunk_coord_y == 0 || chunk_coord_x == 0) {
                symbols.push_back(indices[pixel_index]);
              } else {
                uint8_t diag_color[3], top_color[3], left_color[3];
                get_dxt_color_at(blocks, px - 1, py - 1, width, diag_color);
                get_dxt_color_at(blocks, px, py - 1, width, top_color);
                get_dxt_color_at(blocks, px - 1, py, width, left_color);

                uint8_t predicted[3];
                predict_color(diag_color, top_color, left_color, predicted);

                uint8_t palette[12];
                get_dxt_palette(blocks[(py / 4) * (width / 4) + (px / 4)], palette);
                int predicted_index = predict_index(palette, predicted);

                int delta = ((indices[pixel_index] + 4) - predicted_index) % 4;
                symbols.push_back(delta);
              }
            }
          }
        }
      }
    }
  }

  return std::move(symbols);
}

std::vector<uint8_t> entropy_encode_index_symbols(const std::vector<uint8_t> &symbols) {
  // Make sure that we have a multiples of 256 symbols
  assert((symbols.size() % (256 * 16)) == 0);

  // First collect histogram
  std::vector<int> counts(4, 0);
  for (auto symbol : symbols) {
    assert(symbol < 4);
    counts[symbol]++;
  }

  std::vector<uint8_t> output(4 * sizeof(uint16_t));
  uint32_t bytes_written = 0;

  // Write counts to output
  for (size_t i = 0; i < counts.size(); ++i) {
    assert(counts[i] < (1 << 16));
    *(reinterpret_cast<uint16_t *>(output.data()) + i) = static_cast<uint16_t>(counts[i]);
  }

  std::vector<ans::OpenCLEncoder> encoders;
  encoders.reserve(kNumStreams);
  for (uint32_t i = 0; i < kNumStreams; ++i) {
    encoders.push_back(std::move(ans::OpenCLEncoder(counts)));
  }

  std::vector<uint8_t> encoded;
  uint32_t encoded_bytes_written = 0;
  uint32_t last_encoded_bytes_written = 0;

  uint32_t symbol_offset = 0;
  while(symbol_offset < symbols.size()) {
    for (uint32_t sym_idx = 0; sym_idx < ans::kNumEncodedSymbols; sym_idx++) {
      encoded.resize(encoded_bytes_written + 2*kNumStreams);

      for (uint32_t strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
        ans::BitWriter w(encoded.data() + encoded_bytes_written);
        uint32_t sidx = symbol_offset + (strm_idx + 1) * ans::kNumEncodedSymbols - sym_idx - 1;
        uint8_t symbol = symbols[sidx];

        assert(symbol < counts.size());
        assert(counts[symbol] > 0);

        encoders[strm_idx].Encode(symbol, w);
        encoded_bytes_written += w.BytesWritten();
      }
    }

    // Write the encoder states to the encoded stream...
    encoded.resize(encoded_bytes_written + 4*kNumStreams);
    uint32_t *states = reinterpret_cast<uint32_t *>(encoded.data() + encoded_bytes_written);
    for (uint32_t strm_idx = 0; strm_idx < kNumStreams; ++strm_idx) {
      states[strm_idx] = encoders[strm_idx].GetState();
    }

    // Add the offset to the stream...
    uint32_t offset = encoded_bytes_written - last_encoded_bytes_written;
    output.resize(bytes_written + 2);
    *reinterpret_cast<uint16_t *>(output.data() + bytes_written) = static_cast<uint16_t>(offset);
    bytes_written += 2;
    assert(offset <= ((1 << 16) - 1));
    last_encoded_bytes_written = encoded_bytes_written;

    // Get ready for the next symbols...
    symbol_offset += kNumStreams * ans::kNumEncodedSymbols;
  }

  output.resize(bytes_written + encoded_bytes_written);
  output.insert(output.begin() + bytes_written, encoded.begin(), encoded.end());

  return std::move(output);
}

std::vector<uint8_t> compress_indices(const std::vector<DXTBlock> &blocks,
                                      const std::vector<uint8_t> &indices,
                                      int width, int height) {
  std::vector<uint8_t> symbolized_indices = symbolize_indices(blocks, indices, width, height);
  return std::move(entropy_encode_index_symbols(symbolized_indices));
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

  // Decompress into image...
  cv::Mat img_dxt = DecompressDXT(dxt_blocks, img.cols, img.rows);
  cv::imwrite("img_dxt.png", img_dxt);

  std::vector<uint8_t> indices(img.cols * img.rows, 0);
  for (int j = 0; j < img.rows; j+=4) {
    for (int i = 0; i < img.cols; i+=4) {
      int block_idx = (j / 4) * num_blocks_x + (i / 4);

      int idxs[4];
      get_indices_from_block(dxt_blocks[block_idx], idxs);

      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          int pixel_idx = (j + y) * img.cols + (i + x);
          indices[pixel_idx] = idxs[y * 4 + x];
        }
      }
    }
  }

  // Visualize data...
  {
    uint8_t interp_map[4] = { 0, 85, 170, 255 };
    cv::Mat interp_vis(img.rows, img.cols, CV_8UC1);
    for (int j = 0; j < img.rows; j+=4) {
      for (int i = 0; i < img.cols; i+=4) {
        interp_vis.at<uint8_t>(j, i) = interp_map[indices[j*img.rows + i]];
      }
    }
    cv::imwrite("img_dxt_interp.png", interp_vis);
  }

  // Compress indices...
  std::vector<uint8_t> compressed_indices =
    compress_indices(dxt_blocks, indices, img.cols, img.rows);

  cv::Mat img_A(num_blocks_y, num_blocks_x, CV_8UC4);
  cv::Mat img_B(num_blocks_y, num_blocks_x, CV_8UC4);
  for (int j = 0; j < num_blocks_y; j ++) {
    for (int i = 0; i < num_blocks_x; i ++) {
      int block_idx = j * num_blocks_x + i;
      img_A.at<uint32_t>(j, i) = From565(dxt_blocks[block_idx].ep1);
      img_B.at<uint32_t>(j, i) = From565(dxt_blocks[block_idx].ep2);
    }
  }

  cv::imwrite("img_dxtA.png", img_A);
  cv::imwrite("img_dxtB.png", img_B);

  std::vector<uint8_t> strm_A = compress(img_A);
  cv::Mat decomp_A = decompress(strm_A);

  std::vector<uint8_t> strm_B = compress(img_B);
  cv::Mat decomp_B = decompress(strm_B);

  uint32_t total_compressed_sz = strm_A.size() + strm_B.size() + compressed_indices.size();
  uint32_t dxt_sz = dxt_blocks.size() * 8;
  uint32_t uncompressed_sz = img.cols * img.rows * 3;

  std::cout << "Uncompressed size: " << uncompressed_sz << std::endl;
  std::cout << "DXT compressed size: " << dxt_sz << std::endl;
  std::cout << "GTC compressed size: " << total_compressed_sz << std::endl;

  cv::imwrite("img_codec_dxtA.png", decomp_A);
  cv::imwrite("img_codec_dxtB.png", decomp_B);

  for (int j = 0; j < num_blocks_y; j ++) {
    for (int i = 0; i < num_blocks_x; i ++) {
      uint8_t *pixel;
      int block_idx = j * num_blocks_x + i;

      pixel = decomp_A.ptr(j) + i * 3;
      uint16_t e1 = Into565(pixel[0], pixel[1], pixel[2]);

      pixel = decomp_B.ptr(j) + i * 3;
      uint16_t e2 = Into565(pixel[0], pixel[1], pixel[2]);

      if (e1 > e2) {
        dxt_blocks[block_idx].ep1 = e1;
        dxt_blocks[block_idx].ep2 = e2;
      } else if (e2 < e1) {
        dxt_blocks[block_idx].interpolation ^= 0x55555555;

        dxt_blocks[block_idx].ep1 = e2;
        dxt_blocks[block_idx].ep2 = e1;
      } else {
        dxt_blocks[block_idx].ep1 = e1;
        dxt_blocks[block_idx].ep2 = e1;
        dxt_blocks[block_idx].interpolation = 0;
      }
    }
  }  

  cv::Mat img_dxt_codec = DecompressDXT(dxt_blocks, img.cols, img.rows);
  cv::imwrite("img_gtc.png", img_dxt_codec);  

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

  cv::imwrite("img_dxtA_Y.png", channels[0][0]);

  dct::RunDCT(channels[0]);
  cv::imwrite(is_fast ? "img_dxtA_Y_fast_dct.png" : "img_dxtA_Y_dct.png", channels[0][0]);
  dct::RunIDCT(channels[0]);
  cv::imwrite(is_fast ? "img_dxtA_Y_fast_idct.png" : "img_dxtA_Y_idct.png", channels[0][0]);

  cv::imwrite("img_dxtA_Cr.png", channels[0][1]);
  cv::imwrite("img_dxtA_Cb.png", channels[0][2]);

  dct::RunDCT(channels[1]);
  cv::imwrite(is_fast ? "dxt_imgB_Y_fast_dct.png" : "dxt_imgB_Y_dct.png", channels[1][0]);
  dct::RunIDCT(channels[1]);
  cv::imwrite(is_fast ? "dxt_imgB_Y_fast_idct.png" : "dxt_imgB_Y_idct.png", channels[1][0]);

  cv::imwrite("dxt_imgB_Cr.png", channels[1][1]);
  cv::imwrite("dxt_imgB_Cb.png", channels[1][2]);

  return 0;
}
