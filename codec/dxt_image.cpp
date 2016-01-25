#include "dxt_image.h"

#include <algorithm>
#include <cassert>

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

}  // namespace GenTC