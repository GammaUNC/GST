#include "dxt_image.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <functional>

static void LerpChannels(uint8_t a[3], uint8_t b[3], uint8_t out[3], int num, int div) {
  for (int i = 0; i < 3; ++i) {
    out[i] = (static_cast<int>(a[i]) * (div - num) + static_cast<int>(b[i]) * num) / div;
  }
  out[3] = 0xFF;
}

static void Decode565(uint16_t x, uint8_t out[4]) {
  uint32_t r = (x >> 11);
  r = (r << 3) | (r >> 2);

  uint32_t g = (x >> 5) & 0x3F;
  g = (g << 2) | (g >> 4);

  uint32_t b = x & 0x1F;
  b = (b << 3) | (b >> 2);

  assert(r < 256);
  assert(g < 256);
  assert(b < 256);

  out[0] = r;
  out[1] = g;
  out[2] = b;
  out[3] = 255;
}

namespace GenTC {

LogicalDXTBlock PhysicalToLogical(const PhysicalDXTBlock &b) {
  LogicalDXTBlock out;

  Decode565(b.ep1, out.ep1);
  Decode565(b.ep2, out.ep2);

  memcpy(out.palette[0], out.ep1, 4);
  memcpy(out.palette[1], out.ep2, 4);

  if (b.ep1 <= b.ep2) {
    LerpChannels(out.ep1, out.ep2, out.palette[2], 1, 2);
    memset(out.palette[3], 0, 4);
  } else {
    LerpChannels(out.ep1, out.ep2, out.palette[2], 1, 3);
    LerpChannels(out.ep1, out.ep2, out.palette[3], 2, 3);
  }

  uint8_t const* bytes = reinterpret_cast<const uint8_t *>(&b.interpolation);
  for (int k = 0; k < 4; ++k) {
    uint8_t packed = bytes[k];

    out.indices[0 + 4 * k] = packed & 0x3;
    out.indices[1 + 4 * k] = (packed >> 2) & 0x3;
    out.indices[2 + 4 * k] = (packed >> 4) & 0x3;
    out.indices[3 + 4 * k] = (packed >> 6) & 0x3;
  }

  return out;
}

static void ChunkBy(int chunk_sz_x, int chunk_sz_y, int sz_x, int sz_y,
                    std::function<void (int x, int y)> func) {
  for (int y = 0; y < sz_y; y += chunk_sz_y) {
    for (int x = 0; x < sz_x; x += chunk_sz_x) {
      func(x, y);
    }
  }
}

static std::vector<LogicalDXTBlock>
PhysicalToLogicalBlocks(const std::vector<PhysicalDXTBlock> &blocks) {
  std::vector<LogicalDXTBlock> out;
  out.reserve(blocks.size());

  for (const auto &b : blocks) {
    out.push_back(PhysicalToLogical(b));
  }

  return std::move(out);
}

DXTImage::DXTImage(const uint8_t *dxt_image, int width, int height)
  : _width(width)
  , _height(height)
  , _blocks_width((width + 3) / 4)
  , _blocks_height((height + 3) / 4)
  , _physical_blocks(
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_image),
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_image)+(_blocks_width * _blocks_height))
  , _logical_blocks(PhysicalToLogicalBlocks(_physical_blocks))
{ }

std::vector<uint8_t> DXTImage::EndpointOneImage() const {
  std::vector<uint8_t> result;
  result.reserve(4 * BlocksWide() * BlocksHigh());

  for (const auto &lb : _logical_blocks) {
    result.push_back(lb.ep1[0]);
    result.push_back(lb.ep1[1]);
    result.push_back(lb.ep1[2]);
    result.push_back(lb.ep1[3]);
  }

  return std::move(result);
}

std::vector<uint8_t> DXTImage::EndpointTwoImage() const {
  std::vector<uint8_t> result;
  size_t img_sz = 4 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &lb : _logical_blocks) {
    result.push_back(lb.ep2[0]);
    result.push_back(lb.ep2[1]);
    result.push_back(lb.ep2[2]);
    result.push_back(lb.ep2[3]);
  }

  assert(result.size() == img_sz);
  return std::move(result);
}

std::vector<uint8_t> DXTImage::DecompressedImage() const {
  std::vector<uint8_t> result;
  const size_t img_sz = 4 * Width() * Height();
  result.reserve(img_sz);

  for (int y = 0; y < Height(); ++y) {
    for (int x = 0; x < Width(); ++x) {
      uint8_t c[4];
      GetColorAt(x, y, c);

      result.push_back(c[0]);
      result.push_back(c[1]);
      result.push_back(c[2]);
      result.push_back(c[3]);
    }
  }

  assert(result.size() == img_sz);
  return std::move(result);
}

std::vector<uint8_t> DXTImage::EndpointOneValues() const {
  std::vector<uint8_t> result;
  const size_t img_sz = 3 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &pb : _physical_blocks) {
    uint32_t x = pb.ep1;
    uint32_t r = (x >> 11);
    uint32_t g = (x >> 5) & 0x3F;
    uint32_t b = x & 0x1F;

    assert(r < (1 << 5));
    assert(g < (1 << 6));
    assert(b < (1 << 5));

    result.push_back(static_cast<uint8_t>(r));
    result.push_back(static_cast<uint8_t>(g));
    result.push_back(static_cast<uint8_t>(b));
  }

  assert(result.size() == img_sz);
  return std::move(result);
}

std::vector<uint8_t> DXTImage::EndpointTwoValues() const {
  std::vector<uint8_t> result;
  const size_t img_sz = 3 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &pb : _physical_blocks) {
    uint32_t x = pb.ep2;
    uint32_t r = (x >> 11);
    uint32_t g = (x >> 5) & 0x3F;
    uint32_t b = x & 0x1F;

    assert(r < (1 << 5));
    assert(g < (1 << 6));
    assert(b < (1 << 5));

    result.push_back(static_cast<uint8_t>(r));
    result.push_back(static_cast<uint8_t>(g));
    result.push_back(static_cast<uint8_t>(b));
  }

  assert(result.size() == img_sz);
  return std::move(result);
}

std::vector<uint8_t> DXTImage::TwoBitValuesToImage(const std::vector<uint8_t> &v) {
  std::vector<uint8_t> values = v;
  const uint8_t two_bit_map[4] = { 0, 85, 170, 255 };
  for (auto &v : values) {
    assert(v < 4);
    v = two_bit_map[v];
  }

  return std::move(values);
}

std::vector<uint8_t> DXTImage::InterpolationValues() const {
  std::vector<uint8_t> values;
  values.reserve(_width * _height);

  for (int y = 0; y < _height; ++y) {
    for (int x = 0; x < _width; ++x) {
      values.push_back(InterpolationValueAt(x, y));
    }
  }

  assert(values.size() == _width * _height);
  return std::move(values);
}

uint8_t DXTImage::InterpolationValueAt(int x, int y) const {
  int block_idx = BlockAt(x, y);
  int pixel_idx = (y % 4) * 4 + (x % 4);
  return _logical_blocks[block_idx].indices[pixel_idx];
}

void DXTImage::GetColorAt(int x, int y, uint8_t out[4]) const {
  const LogicalDXTBlock &b = _logical_blocks[BlockAt(x, y)];
  uint8_t i = InterpolationValueAt(x, y);
  out[0] = b.palette[i][0];
  out[1] = b.palette[i][1];
  out[2] = b.palette[i][2];
  out[3] = b.palette[i][3];
}

uint8_t get_gray(const uint8_t color[]) {
  int gray;

  // choose one of the following representations of gray value

  // Average
  gray = static_cast<int> ((color[0] + color[1] + color[2]) / 3);
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
  gray_diag = get_gray(diag);
  gray_upper = get_gray(upper);
  gray_left = get_gray(left);

  uint8_t mb = std::abs(gray_diag - gray_upper);
  uint8_t mc = std::abs(gray_diag - gray_left);
  uint8_t ma = std::abs(mb - mc);

  int temp[3];

  for (int i = 0; i<3; i++) {
    if ((ma < 4) && (mb < 4))
      temp[i] = left[i] + upper[i] - diag[i];
    else if (ma < 10)
      temp[i] = (left[i] + upper[i]) / 2;
    else if (ma < 64) {
      if (mb < mc)
        temp[i] = (3 * left[i] + upper[i]) / 4;
      else
        temp[i] = (left[i] + 3 * upper[i]) / 4;
    }
    else {
      if (mb < mc)
        temp[i] = left[i];
      else
        temp[i] = upper[i];
    }
  } // for

  for (int i = 0; i<3; i++) {
    if (temp[i] < 0)
      temp[i] = 0;
    else if (temp[i] > 255)
      temp[i] = 255;

    predicted[i] = temp[i];
  }
  predicted[3] = 0xFF;
}

int distance(uint8_t *colorA, const uint8_t *colorB) {
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
  distance = (colorA[0] - colorB[0])*(colorA[0] - colorB[0])
    + (colorA[1] - colorB[1])*(colorA[1] - colorB[1])
    + (colorA[2] - colorB[2])*(colorA[2] - colorB[2]);

  return distance;
}

int predict_index(const uint8_t colors[4][4], uint8_t *predicted_color) {
  int difference[4];

  for (int i = 0; i<4; i++)
    difference[i] = distance(predicted_color, colors[i]);

  int min = difference[0];
  int min_id = 0;
  for (int i = 1; i<4; i++) {
    if (min > difference[i]) {
      min = difference[i];
      min_id = i;
    }
  }

  return min_id;
}

std::vector<uint8_t> DXTImage::PredictIndices() const {
  // Operate in 16-block chunks arranged as 4x4 blocks
  assert(Width() % 16 == 0);
  assert(Height() % 16 == 0);

  std::vector<uint8_t> indices = InterpolationValues();
  std::vector<uint8_t> symbols;
  symbols.reserve(indices.size());

  // Each encoder can encode up to 256 values over 16 threads, this divides
  // the input into chunks that have 256 values each.
  ChunkBy(16, 16, Width(), Height(), [&](int chunk_i, int chunk_j) {

    // For each chunk, we want to linearize the indices...
    ChunkBy(4, 4, 16, 16, [&](int block_i, int block_j) {

      // In each block, push back the symbols one by one. If we're
      // along the top or left row in the chunk, don't predict
      // anything..
      ChunkBy(1, 1, 4, 4, [&](int i, int j) {
        int chunk_coord_y = block_j + j;
        int chunk_coord_x = block_i + i;

        int px = chunk_j + block_j + j;
        int py = chunk_i + block_i + i;

        int pixel_index = py * Width() + px;
        if (chunk_coord_y == 0 || chunk_coord_x == 0) {
          symbols.push_back(indices[pixel_index]);
          return;
        }

        uint8_t diag_color[4], top_color[4], left_color[4];
        GetColorAt(px - 1, py - 1, diag_color);
        GetColorAt(px, py - 1, top_color);
        GetColorAt(px - 1, py, left_color);

        uint8_t predicted[4];
        predict_color(diag_color, top_color, left_color, predicted);

        int predicted_index =
          predict_index(LogicalBlockAt(px, py).palette, predicted);

        int delta = ((indices[pixel_index] + 4) - predicted_index) % 4;
        symbols.push_back(delta);
      });
    });
  });

  assert(symbols.size() == indices.size());
  return std::move(symbols);
}

}  // namespace GenTC
