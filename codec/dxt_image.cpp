#include "dxt_image.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <functional>
#include <random>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion-null"
#endif
#define STB_DXT_IMPLEMENTATION
#include "stb_dxt.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "crn_decomp.h"
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#endif

#include "vptree/vptree.hh"
#define PREDICT_VPTREE

template <typename T>
static inline T AbsDiff(T a, T b) {
  return a > b ? a - b : b - a;
}

static void LerpChannels(uint8_t a[3], uint8_t b[3], uint8_t out[3], int num, int div) {
  for (int i = 0; i < 3; ++i) {
    out[i] =
      (static_cast<int>(a[i]) * (div - num) + static_cast<int>(b[i]) * num)
      / div;
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

static uint16_t Pack565(const uint8_t in[3]) {
  uint16_t result = 0;
  result |= static_cast<uint16_t>(in[0] & 0xF8) << 8;
  result |= static_cast<uint16_t>(in[1] & 0xFC) << 3;
  result |= static_cast<uint16_t>(in[2] & 0xF8) >> 3;
  return result;
}

// Snap to closest 5-bit value
static uint8_t ToFiveBits(const uint8_t x) {
  uint8_t base = x & 0xF8;
  uint8_t high = base == 255 ? base : base + 0x4;
  uint8_t low = base == 0 ? base : base - 0x4;

  base = base | (base >> 5);
  high = high | (high >> 5);
  low = low | (low >> 5);

  uint8_t diff_base = AbsDiff(x, base);
  uint8_t diff_high = AbsDiff(x, high);
  uint8_t diff_low = AbsDiff(x, low);

  if (diff_base <= diff_high) {
    if (diff_base < diff_low) {
      return base;
    } else {
      return low;
    }
  } 

  assert(diff_high < diff_low);
  return high;
}

// Snap to closest 6-bit value
static uint8_t ToSixBits(const uint8_t x) {
  uint8_t base = x & 0xFC;
  uint8_t high = base == 255 ? base : base + 0x2;
  uint8_t low = base == 0 ? base : base - 0x2;

  base = base | (base >> 6);
  high = high | (high >> 6);
  low = low | (low >> 6);

  uint8_t diff_base = AbsDiff(x, base);
  uint8_t diff_high = AbsDiff(x, high);
  uint8_t diff_low = AbsDiff(x, low);

  if (diff_base <= diff_high) {
    if (diff_base < diff_low) {
      return base;
    } else {
      return low;
    }
  }

  assert(diff_high < diff_low);
  return high;
}

namespace GenTC {

PhysicalDXTBlock LogicalToPhysical(const LogicalDXTBlock &b);
LogicalDXTBlock PhysicalToLogical(const PhysicalDXTBlock &b);

#ifndef NDEBUG
static bool operator==(const PhysicalDXTBlock &a, const PhysicalDXTBlock &b) {
  return a.dxt_block == b.dxt_block;
}
#endif

LogicalDXTBlock PhysicalToLogical(const PhysicalDXTBlock &b) {
  LogicalDXTBlock out;

  Decode565(b.ep1, out.ep1);
  Decode565(b.ep2, out.ep2);

  memcpy(out.palette[0], out.ep1, 4);
  memcpy(out.palette[1], out.ep2, 4);

  if (b.ep1 <= b.ep2) {
    LerpChannels(out.ep1, out.ep2, out.palette[2], 1, 2);
    memset(out.palette[3], 0, 4);
  }
  else {
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

static bool PhysicalBlockNeedsSwap(const LogicalDXTBlock &b) {
  uint16_t p0 = Pack565(b.ep1);
  uint16_t p1 = Pack565(b.ep2);

  bool swap = p0 > p1 && b.palette[3][3] == 0;
  swap = swap || (p0 <= p1 && b.palette[3][3] == 255);
  return swap;
}

PhysicalDXTBlock LogicalToPhysical(const LogicalDXTBlock &b) {
  PhysicalDXTBlock result;
  result.ep1 = Pack565(b.ep1);
  result.ep2 = Pack565(b.ep2);

  bool swap = PhysicalBlockNeedsSwap(b);
  if (swap) {
    std::swap(result.ep1, result.ep2);
  }

  result.interpolation = 0;
  uint8_t *bytes = reinterpret_cast<uint8_t *>(&result.interpolation);
  for (int k = 0; k < 4; ++k) {
    assert(b.indices[0 + 4 * k] < 4);
    bytes[k] |= b.indices[0 + 4 * k];

    assert(b.indices[1 + 4 * k] < 4);
    bytes[k] |= b.indices[1 + 4 * k] << 2;

    assert(b.indices[2 + 4 * k] < 4);
    bytes[k] |= b.indices[2 + 4 * k] << 4;

    assert(b.indices[3 + 4 * k] < 4);
    bytes[k] |= b.indices[3 + 4 * k] << 6;
  }

  if (swap) {
    result.interpolation ^= 0x55555555;
  }

  return result;
}

static void Deinterleave(uint32_t x, uint16_t *i, uint16_t *j) {
  uint64_t w = (static_cast<uint64_t>(x & 0xAAAAAAAA) << 31) | (x & 0x55555555);
  w = (w | (w >> 1)) & 0x3333333333333333ULL;
  w = (w | (w >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
  w = (w | (w >> 4)) & 0x00FF00FF00FF00FFULL;
  w = (w | (w >> 8)) & 0x0000FFFF0000FFFFULL;
  *i = static_cast<uint16_t>(w & 0xFFFF);
  *j = static_cast<uint16_t>((w >> 32) & 0xFFFF);
}

static void ChunkBy(int chunk_sz_x, int chunk_sz_y, int sz_x, int sz_y,
  std::function<void(int x, int y)> func) {
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

  for (const PhysicalDXTBlock &b : blocks) {
    LogicalDXTBlock lb = PhysicalToLogical(b);
    assert(LogicalToPhysical(lb) == b);
    out.push_back(lb);
  }

  return std::move(out);
}

static std::vector<PhysicalDXTBlock>
  LogicalToPhysicalBlocks(const std::vector<LogicalDXTBlock> &blocks) {
  std::vector<PhysicalDXTBlock> out;
  out.reserve(blocks.size());

  for (const LogicalDXTBlock &b : blocks) {
    PhysicalDXTBlock pb = LogicalToPhysical(b);
    assert(PhysicalToLogical(pb) == b);
    out.push_back(pb);
  }

  return std::move(out);
}

struct CompressedBlock {
  std::vector<uint8_t> _uncompressed;
  LogicalDXTBlock _logical;

  size_t Error() const {
    size_t err = 0;
    for (size_t idx = 0; idx < 16; idx++) {
      uint8_t i = _logical.indices[idx];

      uint8_t pixel[3];
      pixel[0] = _logical.palette[i][0];
      pixel[1] = _logical.palette[i][1];
      pixel[2] = _logical.palette[i][2];

      size_t diff_r = AbsDiff(_uncompressed[idx * 3 + 0], pixel[0]);
      size_t diff_g = AbsDiff(_uncompressed[idx * 3 + 1], pixel[1]);
      size_t diff_b = AbsDiff(_uncompressed[idx * 3 + 2], pixel[2]);

      err += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }

    return err / (16 * 3);
  }

  size_t CompareAgainst(const LogicalDXTBlock &other) const {
    CompressedBlock dup = *this;
    dup._logical = other;
    dup.RecalculateEndpoints();
    return dup.Error();
  }

  void AssignIndices(const uint32_t idx) {
    PhysicalDXTBlock pblk = LogicalToPhysical(_logical);
    pblk.interpolation = idx;
    _logical = PhysicalToLogical(pblk);
  }

  void RecalculateEndpoints() {
    // Now that we know the index of each pixel, we can assign the endpoints based
    // on a least squares fit of the clusters. For more information, take a look
    // at this article by NVidia: http://developer.download.nvidia.com/compute/
    // cuda/1.1-Beta/x86_website/projects/dxtc/doc/cuda_dxtc.pdf
    float asq = 0.0, bsq = 0.0, ab = 0.0;
    float ax[3] = { 0.0f, 0.0f, 0.0f };
    float bx[3] = { 0.0f, 0.0f, 0.0f };
    for (size_t i = 0; i < 16; i++) {
      const uint8_t *orig_pixel = _uncompressed.data() + i * 3;

      static const float idx_to_order[4] = { 0.f, 3.f, 1.f, 2.f };
      const float order = idx_to_order[_logical.indices[i]];
      const float fbi = 3.0f - order;
      const float fb = 3.0f;
      const float fi = order;

      const float a = fbi / fb;
      const float b = fi / fb;

      asq += a * a;
      bsq += b * b;
      ab += a * b;

      for (size_t j = 0; j < 3; ++j) {
        ax[j] += static_cast<float>(orig_pixel[j]) * a;
        bx[j] += static_cast<float>(orig_pixel[j]) * b;
      }
    }

    float f = 1.0f / (asq * bsq - ab * ab);

    float p1[3], p2[3];
    for (int i = 0; i < 3; ++i) {
      p1[i] = f * (ax[i] * bsq - bx[i] * ab);
      p2[i] = f * (bx[i] * asq - ax[i] * ab);
    }

    // Quantize the endpoints...
    for (int i = 0; i < 3; ++i) {
      _logical.ep1[i] = std::max(0, std::min(255, static_cast<int32_t>(p1[i] + 0.5f)));
      _logical.ep2[i] = std::max(0, std::min(255, static_cast<int32_t>(p2[i] + 0.5f)));
    }

    _logical.ep1[0] = ToFiveBits(_logical.ep1[0]);
    _logical.ep2[0] = ToFiveBits(_logical.ep2[0]);

    _logical.ep1[1] = ToSixBits(_logical.ep1[1]);
    _logical.ep2[1] = ToSixBits(_logical.ep2[1]);

    _logical.ep1[2] = ToFiveBits(_logical.ep1[2]);
    _logical.ep2[2] = ToFiveBits(_logical.ep2[2]);

    memcpy(_logical.palette[0], _logical.ep1, 4);
    memcpy(_logical.palette[1], _logical.ep2, 4);

    LerpChannels(_logical.ep1, _logical.ep2, _logical.palette[2], 1, 3);
    LerpChannels(_logical.ep1, _logical.ep2, _logical.palette[3], 2, 3);
  }
};

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

  PhysicalDXTBlock result;
  //squish::Compress(block, reinterpret_cast<unsigned char *>(&result.dxt_block), squish::kDxt1);

  stb_compress_dxt_block(reinterpret_cast<unsigned char *>(&result.dxt_block),
    block, 0, STB_DXT_HIGHQUAL);

  return result.dxt_block;
}

DXTImage::DXTImage(const char *orig_fn, const char *cmp_fn) {
  std::string cmp_fname(cmp_fn ? cmp_fn : "");
  if (cmp_fname.substr(cmp_fname.find_last_of(".") + 1) == "crn") {
    std::ifstream ifs(cmp_fname.c_str(), std::ifstream::binary | std::ifstream::ate);
    std::ifstream::pos_type pos = ifs.tellg();

    std::vector<uint8_t> crn(pos);

    ifs.seekg(0, std::ifstream::beg);
    ifs.read(reinterpret_cast<char *>(crn.data()), pos);

    crnd::crn_texture_info tinfo;
    if (!crnd::crnd_get_texture_info(crn.data(), crn.size(), &tinfo)) {
      assert(!"Invalid texture?");
      return;
    }

    assert(tinfo.m_width == static_cast<uint32_t>(_width));
    assert(tinfo.m_height == static_cast<uint32_t>(_height));

    crnd::crnd_unpack_context ctx = crnd::crnd_unpack_begin(crn.data(), crn.size());
    if (!ctx) {
      assert(!"Error beginning crn decoding!");
      return;
    }

    const int num_blocks_x = (tinfo.m_width + 3) / 4;
    const int num_blocks_y = (tinfo.m_height + 3) / 4;
    const int num_blocks = num_blocks_x * num_blocks_y;
    _physical_blocks.resize(num_blocks);

    void *dst = _physical_blocks.data();
    if (!crnd::crnd_unpack_level(ctx, &dst, num_blocks * 8, num_blocks_x * 8, 0)) {
      assert(!"Error decoding crunch texture!");
      return;
    }

    crnd::crnd_unpack_end(ctx);
  } 
  
  // Otherwise, load the file
  stbi_uc *data = stbi_load(orig_fn, &_width, &_height, NULL, 3);
  if (!data) {
    assert(!"Error loading image");
    std::cout << "Error loading image" << orig_fn << std::endl;
    return;
  }

  size_t src_img_sz = _width * _height * 3;
  _src_img.resize(src_img_sz);
  memcpy(_src_img.data(), data, src_img_sz);

  _blocks_width = (_width + 3) / 4;
  _blocks_height = (_height + 3) / 4;
  const int num_blocks = _blocks_width * _blocks_height;

  if (_physical_blocks.size() == 0) {
    // Compress the DXT data
    _physical_blocks.resize(num_blocks);
    for (int physical_idx = 0; physical_idx < num_blocks; ++physical_idx) {
      uint16_t i, j;
      i = static_cast<uint16_t>(physical_idx % _blocks_width);
      j = static_cast<uint16_t>(physical_idx / _blocks_width);

      int block_idx = j * _blocks_width + i;
      const unsigned char *offset_data = data + (j * 4 * _width + i * 4) * 3;
      _physical_blocks[block_idx].dxt_block = CompressRGB(offset_data, _width);
    }
  }

  // Optimize it...
  Reencode();
}

DXTImage::DXTImage(int width, int height, const std::vector<uint8_t> &rgb_data,
                   const std::vector<uint8_t> &dxt_data)
  : _width(width)
  , _height(height)
  , _blocks_width((width + 3) / 4)
  , _blocks_height((height + 3) / 4)
  , _physical_blocks(
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_data.data()),
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_data.data())
    + (_blocks_width * _blocks_height))
  , _logical_blocks(PhysicalToLogicalBlocks(_physical_blocks))
  , _src_img(rgb_data)
{
  Reencode();
}

DXTImage::DXTImage(int width, int height, const std::vector<uint8_t> &dxt_data)
  : _width(width)
  , _height(height)
  , _blocks_width((width + 3) / 4)
  , _blocks_height((height + 3) / 4)
  , _physical_blocks(
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_data.data()),
    reinterpret_cast<const PhysicalDXTBlock *>(dxt_data.data())
    + (_blocks_width * _blocks_height))
  , _logical_blocks(PhysicalToLogicalBlocks(_physical_blocks))
{ }

double DXTImage::PSNR() const {
  if (_src_img.size() == 0) {
    std::cout << "WARNING: Cannot compute PSNR, no original data!" << std::endl;
    assert(false);
    return -1.0;
  }

  // Compute DXT PSNR...
  double orig_mse = 0.0;
  for (int j = 0; j < Height(); ++j) {
    for (int i = 0; i < Width(); ++i) {
      uint8_t pixel[4];
      GetColorAt(i, j, pixel);

      const size_t src_idx = (j * Width() + i) * 3;
      for (int c = 0; c < 3; ++c) {
        double orig = static_cast<double>(_src_img[src_idx + c]);
        double cmp = static_cast<double>(pixel[c]);
        double diff = orig - cmp;
        orig_mse += diff * diff;
      }
    }
  }

  orig_mse /= static_cast<double>(Width() * Height());
  return 10.0 * log10((3.0 * 255.0 * 255.0) / orig_mse);
}

static const int kErrThreshold = 35;
static const size_t kNumPrevLookup = 128;
void DXTImage::Reencode() {
  _logical_blocks = std::move(PhysicalToLogicalBlocks(_physical_blocks));
  std::cout << "DXT Compressed PSNR: " << PSNR() << std::endl;

  assert((_width & 0x3) == 0);
  assert((_height & 0x3) == 0);

  // Now do the dxt compression...
  int last_index = 0;

  const int num_blocks = _blocks_width * _blocks_height;
  for (int physical_idx = 0; physical_idx < num_blocks; ++physical_idx) {
    uint16_t i, j;
    Deinterleave(static_cast<uint32_t>(physical_idx), &i, &j);
    i = static_cast<uint16_t>(physical_idx % _blocks_width);
    j = static_cast<uint16_t>(physical_idx / _blocks_width);

    int block_idx = j * _blocks_width + i;
    assert(block_idx == physical_idx);
    const unsigned char *offset_data = _src_img.data() + (j * 4 * _width + i * 4) * 3;

    CompressedBlock blk;
    blk._logical = _logical_blocks[block_idx];
    blk._uncompressed = std::vector<uint8_t>(48, 0);

    for (int row = 0; row < 4; ++row) {
      uint8_t block_row[12];
      for (int p = 0; p < 4; ++p) {
        block_row[3 * p + 0] = offset_data[(row * _width + p) * 3 + 0];
        block_row[3 * p + 1] = offset_data[(row * _width + p) * 3 + 1];
        block_row[3 * p + 2] = offset_data[(row * _width + p) * 3 + 2];
      }

      memcpy(blk._uncompressed.data() + 12 * row, block_row, sizeof(block_row));
    }

    const int orig_err = static_cast<int>(blk.Error());
    int min_err = std::numeric_limits<int>::max();
    size_t min_err_idx = 0;

    for (size_t idx = 0; idx < std::min<size_t>(kNumPrevLookup - 1, _index_palette.size()); ++idx) {
      uint32_t indices = *(_index_palette.crbegin() + idx);
      CompressedBlock blk2 = blk;
      blk2.AssignIndices(indices);
      blk2.RecalculateEndpoints();

      // !HACK! Check if it flips the indices... There has to be a
      // better way to deal with this... In principle we can just leave
      // them flipped and then reflip them back to the proper value
      // in the decompressor...
      PhysicalDXTBlock maybe_blk = LogicalToPhysical(blk2._logical);
      bool ok = maybe_blk.interpolation == indices;
      ok = ok && blk2._logical.palette[3][3] == 0xFF;
      if (!ok) {
        continue;
      }

      int err = static_cast<int>(blk2.Error());
      int err_diff = err - orig_err;
      if (err_diff < min_err) {
        min_err = err_diff;
        min_err_idx = idx;
        if (err_diff <= 0) {
          break;
        }
      }
    }

    int this_index = -1;
    if (min_err < kErrThreshold) {
      blk.AssignIndices(*(_index_palette.crbegin() + min_err_idx));
      blk.RecalculateEndpoints();
      assert(static_cast<int>(blk.Error()) - orig_err == min_err);
      _logical_blocks[block_idx] = blk._logical;
      _physical_blocks[block_idx] = LogicalToPhysical(blk._logical);
      this_index = static_cast<int>(_index_palette.size() - min_err_idx - 1);
    } else {
      this_index = _index_palette.size();
      _index_palette.push_back(_physical_blocks[block_idx].interpolation);
    }

    int idx_diff = this_index - last_index;
    assert(-128 <= idx_diff && idx_diff < 128);

    // The first index... everyone knows it's zero...
    assert(physical_idx != 0 || 0 == this_index);
    assert(physical_idx != 0 || 0 == last_index);
    assert(physical_idx != 0 || 0 == idx_diff);

    _indices.push_back(idx_diff + 128);
    last_index = this_index;
  }

  std::cout << "Unique index blocks: " << _index_palette.size() << std::endl;
  std::cout << "DXT Optimized PSNR: " << PSNR() << std::endl;
}

std::vector<uint8_t> DXTImage::PaletteData() const {
  std::vector<uint8_t> ret(_index_palette.size() * 4, 0);
  memcpy(ret.data(), _index_palette.data(), ret.size());
  return std::move(ret);
}

std::unique_ptr<RGBAImage> DXTImage::EndpointOneImage() const {
  std::vector<uint8_t> result;
  result.reserve(4 * BlocksWide() * BlocksHigh());

  for (const auto &lb : _logical_blocks) {
    result.push_back(lb.ep1[0]);
    result.push_back(lb.ep1[1]);
    result.push_back(lb.ep1[2]);
    result.push_back(lb.ep1[3]);
  }

  std::unique_ptr<RGBAImage> img
    (new RGBAImage(BlocksWide(), BlocksHigh(), std::move(result)));
  return std::move(img);
}

std::unique_ptr<RGBAImage> DXTImage::EndpointTwoImage() const {
  std::vector<uint8_t> result;
  size_t img_sz = 4 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &lb : _logical_blocks) {
    result.push_back(lb.ep2[0]);
    result.push_back(lb.ep2[1]);
    result.push_back(lb.ep2[2]);
    result.push_back(lb.ep2[3]);
  }

  std::unique_ptr<RGBAImage> img
    (new RGBAImage(BlocksWide(), BlocksHigh(), std::move(result)));
  return std::move(img);
}

std::unique_ptr<RGBAImage> DXTImage::DecompressedImage() const {
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

  std::unique_ptr<RGBAImage> img
    (new RGBAImage(Width(), Height(), std::move(result)));
  return std::move(img);
}

std::unique_ptr<RGB565Image> DXTImage::EndpointOneValues() const {
  std::vector<uint8_t> result;
  const size_t img_sz = 2 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &pb : _physical_blocks) {
    uint32_t x = pb.ep1;
    result.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
    result.push_back(static_cast<uint8_t>(x & 0xFF));
  }

  assert(result.size() == img_sz);

  std::unique_ptr<RGB565Image> img
    (new RGB565Image(BlocksWide(), BlocksHigh(), std::move(result)));
  return std::move(img);
}

std::unique_ptr<RGB565Image> DXTImage::EndpointTwoValues() const {
  std::vector<uint8_t> result;
  const size_t img_sz = 2 * BlocksWide() * BlocksHigh();
  result.reserve(img_sz);

  for (const auto &pb : _physical_blocks) {
    uint32_t x = pb.ep2;
    result.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
    result.push_back(static_cast<uint8_t>(x & 0xFF));
  }

  assert(result.size() == img_sz);

  std::unique_ptr<RGB565Image> img
    (new RGB565Image(BlocksWide(), BlocksHigh(), std::move(result)));
  return std::move(img);
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

  assert(values.size() == static_cast<size_t>(_width * _height));
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

void predict_color_wennersten(const uint8_t diag[], const uint8_t upper[],
                              const uint8_t left[], uint8_t *predicted) {
  int16_t temp[3];

  for (int i = 0; i<3; i++) {
    int16_t l = static_cast<uint16_t>(left[i]);
    int16_t u = static_cast<uint16_t>(upper[i]);
    int16_t d = static_cast<uint16_t>(diag[i]);

    int16_t mb = AbsDiff(d, u);
    int16_t mc = AbsDiff(d, l);
    int16_t ma = AbsDiff(mb, mc);

    if ((ma < 4) && (mb < 4))
      temp[i] = l + u - d;
    else if (ma < 10)
      temp[i] = (l + u) / 2;
    else if (ma < 64) {
      if (mb < mc)
        temp[i] = (3 * l + u) / 4;
      else
        temp[i] = (l + 3 * u) / 4;
    }
    else {
      if (mb < mc)
        temp[i] = l;
      else
        temp[i] = u;
    }
  } // for

  for (int i = 0; i<3; i++) {
    if (temp[i] < 0)
      temp[i] = 0;
    else if (temp[i] > 255)
      temp[i] = 255;

    predicted[i] = static_cast<uint8_t>(temp[i]);
  }
  predicted[3] = 0xFF;
}

void predict_color_med(const uint8_t diag[], const uint8_t upper[],
                       const uint8_t left[], uint8_t *predicted) {
  int16_t temp[3];

  for (int i = 0; i<3; i++) {
    int16_t l = static_cast<uint16_t>(left[i]);
    int16_t u = static_cast<uint16_t>(upper[i]);
    int16_t d = static_cast<uint16_t>(diag[i]);

    int16_t mxa = std::max(l, u);
    int16_t mna = std::min(l, u);

    if ( d >= mxa )
      temp[i] = mna;
    else if ( d <= mna )
      temp[i] = mxa;
    else {
      temp[i] = l + u - d;
    }
  } // for

  for (int i = 0; i<3; i++) {
    if (temp[i] < 0)
      temp[i] = 0;
    else if (temp[i] > 255)
      temp[i] = 255;

    predicted[i] = static_cast<uint8_t>(temp[i]);
  }
  predicted[3] = 0xFF;
}

void predict_color(const uint8_t diag[], const uint8_t upper[],
                   const uint8_t left[], uint8_t *predicted) {
  predict_color_med(diag, upper, left, predicted);
}

int distance(uint8_t *colorA, const uint8_t *colorB) {
  // abs of gray values
  // int gray_a = get_gray(colorA);
  // int gray_b = get_gray(colorB);
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

uint8_t predict_index(const uint8_t colors[4][4], uint8_t *predicted_color) {
  int difference[4];

  for (int i = 0; i<4; i++)
    difference[i] = distance(predicted_color, colors[i]);

  int min = difference[0];
  int min_id = 0;
  for (int i = 1; i<4; i++) {
    if (difference[i] <= min) {
      min = difference[i];
      min_id = i;
    }
  }

  return min_id;
}

uint8_t compute_prediction_delta(uint8_t idx, uint8_t orig_idx) {
  // The indices are ordered like 0, 3, 1, 2
  static const uint8_t idx_to_order[4] = { 0, 3, 1, 2 };

  assert(orig_idx < 4);
  assert(idx < 4);
  uint8_t orig_order = idx_to_order[orig_idx];
  uint8_t pred_order = idx_to_order[idx];

  return ((orig_order + 4) - pred_order) % 4;
}

#ifdef PREDICT_VPTREE

static double BlockDist(const LogicalDXTBlock &p1, const LogicalDXTBlock &p2) {
  // The indices are ordered like 0, 3, 1, 2
  static const uint8_t idx_to_order[4] = { 0, 3, 1, 2 };

  double dist = 0.0;
  for (size_t i = 0; i < 16; ++i) {
    double x = static_cast<double>(idx_to_order[p1.indices[i]]);
    double y = static_cast<double>(idx_to_order[p2.indices[i]]);

    double err = x - y;
    dist += err * err;
  }

  return dist;
}

static std::vector<std::pair<uint32_t, size_t> > CountBlocks(
  const std::vector<PhysicalDXTBlock> &blocks) {
  typedef std::pair<uint32_t, size_t> Res;
  std::vector<Res> result;
  result.reserve(blocks.size());

  for (const auto &b : blocks) {
    bool found = false;
    for (auto &r : result) {
      if (r.first == b.interpolation) {
        r.second++;
        found = true;
        break;
      }
    }

    if (!found) {
      result.push_back(std::make_pair(b.interpolation, 1));
    }
  }

  std::sort(result.begin(), result.end(), [](const Res &a, const Res &b) {
    return std::greater<size_t>()(a.second, b.second);
  });

  return std::move(result);
}

static std::vector<LogicalDXTBlock> KMeansBlocks(const std::vector<PhysicalDXTBlock> &blocks,
                                                 size_t num_clusters) {
  // Generate num_clusters random logical blocks
  std::vector<LogicalDXTBlock> clusters;
  clusters.reserve(num_clusters);

  std::default_random_engine gen(
    std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<unsigned> dist(0, blocks.size() - 1);

  std::vector<std::pair<uint32_t, size_t> > counted_indices = CountBlocks(blocks);
  std::cout << "Num unique index blocks: " << counted_indices.size() << std::endl;

  size_t num_duplicated = counted_indices.size();
  for (size_t i = 0; i < counted_indices.size(); ++i) {
    if (counted_indices[i].second == 1) {
      num_duplicated = i;
      break;
    }
  }

  std::cout << "Num duplicated indices: " << num_duplicated << std::endl;

  while (clusters.size() < num_clusters) {

    LogicalDXTBlock blk;
    memset(&blk, 0, sizeof(blk));

    //////////
    /// Choose cluster samples as first most-used 
    ///
    PhysicalDXTBlock pblk;
    pblk.interpolation = counted_indices[clusters.size()].first;
    memcpy(blk.indices, PhysicalToLogical(pblk).indices, sizeof(blk.indices));

    //////////
    /// Choose cluster samples as sampling of existing index values
    ///
    // memcpy(blk.indices, PhysicalToLogical(blocks[dist(gen)]).indices, sizeof(blk.indices));

    //////////
    /// Choose a stratified initial distribution of clusters
    ///
    //for (int i = 0; i < 8; ++i) {
    //  uint8_t blk0[2] = { 0xFF, 0x00 };
    //  uint8_t blk1[2] = { 0x00, 0xFF };

    //  if ((clusters.size() >> i) & 0x1) {
    //    blk.indices[2 * i + 0] = blk0[0];
    //    blk.indices[2 * i + 1] = blk0[1];
    //  } else {
    //    blk.indices[2 * i + 0] = blk1[0];
    //    blk.indices[2 * i + 1] = blk1[1];
    //  }
    //}

    //////////
    /// Choose a random initial distribution of clusters
    ///
    //for (int i = 0; i < 16; ++i) {
    //  blk.indices[i] = dist(gen);
    //  assert(blk.indices[i] < 4);
    //}

    //// If we already have this seeded cluster, then generate a new one...
    //if (std::find(clusters.begin(), clusters.end(), blk) != clusters.end()) {
    //  continue;
    //}

    clusters.push_back(blk);
  }

  // Allocate data for k-means
  std::vector<std::vector<double> > old_clusters
    (clusters.size(), std::vector<double>(16, 0.0));
  std::vector<std::vector<double> > new_clusters
    (clusters.size(), std::vector<double>(16, 0.0));
  std::vector<size_t> cluster_idx(blocks.size(), 0);
  std::vector<double> tmp(16, 0.0);

  // Get old clusters from initial clusters...
  for (size_t i = 0; i < num_clusters; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      old_clusters[i][j] = static_cast<double>(clusters[i].indices[j]);
    }
  }

  // Do until fixed point...
  for (;;) {

    // Assign each block to a cluster
    EuclideanVPTree vp_tree;
    vp_tree.addMany(old_clusters.begin(), old_clusters.end());

    for (size_t i = 0; i < blocks.size(); ++i) {
      for (size_t j = 0; j < 16; ++j) {
        tmp[j] = static_cast<double>(PhysicalToLogical(blocks[i]).indices[j]);
      }

      const std::vector<double> *closest = vp_tree.nearestNeighbors(tmp)[0];

      auto closest_itr = std::find(old_clusters.begin(), old_clusters.end(), *closest);
      size_t idx = closest_itr - old_clusters.begin();
      assert(idx < blocks.size());
      cluster_idx[i] = idx;
    }

    // Generate new clusters based on the index of each cluster...
    size_t cidx = 0;
    for (size_t c = 0; c < clusters.size(); ++c) {
      tmp.assign(16, 0.0);
      double cluster_sz = 0.0;
      for (size_t b = 0; b < blocks.size(); ++b) {
        if (cluster_idx[b] == c) {
          for (size_t i = 0; i < 16; ++i) {
            tmp[i] += static_cast<double>(PhysicalToLogical(blocks[b]).indices[i]);
          }

          cluster_sz += 1.0;
        }
      }

      if (0.0 != cluster_sz) {
        for (size_t i = 0; i < 16; ++i) {
          tmp[i] /= cluster_sz;
        }

        new_clusters[cidx].assign(tmp.begin(), tmp.end());
        cidx++;
      }
    }

    bool fixed_point = cidx == old_clusters.size();
    for (size_t i = 0; i < cidx && fixed_point; ++i) {
      fixed_point = old_clusters[i] == new_clusters[i];
    }

    if (fixed_point) {
      break;
    }

    // Not fixed point, so set each old cluster to a new cluster...
    for (size_t i = 0; i < cidx; ++i) {
      old_clusters[i].assign(new_clusters[i].begin(), new_clusters[i].end());
    }
    old_clusters.resize(cidx);
  }

  // Create discrete clusters by rounding continuous clusters
  for (size_t i = 0; i < old_clusters.size(); ++i) {
    for (size_t j = 0; j < 16; ++j) {
      clusters[i].indices[j] = static_cast<uint8_t>(old_clusters[i][j] + 0.5);
      assert(0 <= clusters[i].indices[j] && clusters[i].indices[j] < 4);
    }
  }

  clusters.resize(old_clusters.size());
  return std::move(clusters);
}

class IndexVPTree : public VPTree<LogicalDXTBlock> {
 protected:
  double distance(const LogicalDXTBlock &p1, const LogicalDXTBlock &p2) override {
    return BlockDist(p1, p2);
  }
};

std::vector<uint8_t>
DXTImage::PredictIndices(int chunk_width, int chunk_height) const {
  // Operate in 16-block chunks arranged as 4x4 blocks
  assert(Width() % 16 == 0);
  assert(Height() % 16 == 0);

  std::vector<LogicalDXTBlock> blocks = std::move(KMeansBlocks(PhysicalBlocks(), 256));
  std::cout << "Number of block clusters: " << blocks.size() << std::endl;

  IndexVPTree vptree;
  vptree.addMany(blocks.begin(), blocks.end());

  std::vector<uint8_t> symbols;
  symbols.reserve(Height() * Width());

  for (int py = 0; py < Height(); ++py) {
    for (int px = 0; px < Width(); ++px) {
      const LogicalDXTBlock &blk = LogicalBlockAt(px, py);

      int local_idx = (py % 4) * 4 + (px % 4);
      uint8_t predicted_index = vptree.nearestNeighbors(blk)[0]->indices[local_idx];
      uint8_t predicted_delta = compute_prediction_delta(predicted_index, blk.indices[local_idx]);

      symbols.push_back(predicted_delta);
    }
  }

  return std::move(symbols);
}

#else

std::vector<uint8_t>
DXTImage::PredictIndices(int chunk_width, int chunk_height) const {
  // Operate in 16-block chunks arranged as 4x4 blocks
  assert(Width() % 16 == 0);
  assert(Height() % 16 == 0);

  std::vector<uint8_t> indices = InterpolationValues();
  std::vector<uint8_t> symbols;
  symbols.reserve(indices.size());

  for (int py = 0; py < Height(); ++py) {
    for (int px = 0; px < Width(); ++px) {
      int pixel_index = py * Width() + px;
      if ((px % chunk_width) == 0 || (py % chunk_height) == 0) {
        symbols.push_back(indices[pixel_index]);
        continue;
      }

      uint8_t diag_color[4], top_color[4], left_color[4];
      GetColorAt(px - 1, py - 1, diag_color);
      GetColorAt(px, py - 1, top_color);
      GetColorAt(px - 1, py, left_color);

      uint8_t predicted[4];
      predict_color(diag_color, top_color, left_color, predicted);

      uint8_t predicted_index = predict_index(LogicalBlockAt(px, py).palette, predicted);
      uint8_t predicted_delta = compute_prediction_delta(predicted_index, indices[pixel_index]);

      symbols.push_back(predicted_delta);
    }
  }

  assert(symbols.size() == indices.size());
  return std::move(symbols);
}

#endif

std::vector<uint8_t>
DXTImage::PredictIndicesLinearize(int chunk_width, int chunk_height) const {
  std::vector<uint8_t> predicted = PredictIndices(chunk_width, chunk_height);
  std::vector<uint8_t> symbols;
  symbols.reserve(predicted.size());

  int sym_idx = 0;

  // Each encoder can encode up to 256 values over 16 threads, this divides
  // the input into chunks that have 256 values each.
  ChunkBy(chunk_width, chunk_height, Width(), Height(), [&](int chunk_i, int chunk_j) {
    // For each chunk, we want to linearize the indices...
    ChunkBy(4, 4, 16, 16, [&](int block_i, int block_j) {
      // In each block, push back the symbols one by one. If we're
      // along the top or left row in the chunk, don't predict
      // anything..
      ChunkBy(1, 1, 4, 4, [&](int i, int j) {
          symbols.push_back(predicted[sym_idx++]);
      });
    });
  });

  assert(symbols.size() == predicted.size());
  return std::move(symbols);
}

void DXTImage::ReassignIndices(int mse_threshold) {
  if (_src_img.size() == 0) {
    std::cout << "WARNING: Cannot reassign DXT indices without source data" << std::endl;
    assert(false);
    return;
  }

  std::vector<std::pair<uint32_t, size_t> > counted_indices = CountBlocks(PhysicalBlocks());

  // Collect compressed blocks
  std::vector<CompressedBlock> blocks;
  blocks.resize(LogicalBlocks().size());

  for (int y = 0; y < _blocks_height; ++y) {
    for (int x = 0; x < _blocks_width; ++x) {
      int block_idx = y * _blocks_width + x;
      CompressedBlock &blk = blocks[block_idx];

      for (int row = 0; row < 4; ++row) {
        int row_idx = ((4 * y + row) * _width + (4 * x)) * 3;
        blk._uncompressed.insert(
          blk._uncompressed.end(),
          _src_img.begin() + row_idx,
          _src_img.begin() + row_idx + 12);
      }

      blk._logical = LogicalBlocks()[block_idx];
    }
  }

  // For each block, see if we can reassign its index: search through
  // all of the other indices and measure it against the current index. If we
  // find one that's not as bad w.r.t. the current one, then switch the index...

  int idx = 0;
  for (auto &block : blocks) {
    idx++;
    size_t min_MSE = mse_threshold;
    std::pair<uint32_t, size_t> *cnt_ptr = nullptr;
    std::pair<uint32_t, size_t> *orig_cnt_ptr = nullptr;

    PhysicalDXTBlock pb = LogicalToPhysical(block._logical);
    assert(PhysicalToLogical(pb) == block._logical);
    
    for (auto &cnt : counted_indices) {
      if (cnt.second == 0) {
        continue;
      }

      if (pb.interpolation == cnt.first) {
        orig_cnt_ptr = &cnt;
        continue;
      }

      PhysicalDXTBlock npb = pb;
      npb.interpolation = cnt.first;
      LogicalDXTBlock lb = PhysicalToLogical(npb);

      size_t mse = block.CompareAgainst(lb);
      if (mse < min_MSE) {
        block._logical = lb;
        min_MSE = mse;
        cnt_ptr = &cnt;
      }
    }

    assert(orig_cnt_ptr);

    if (min_MSE < static_cast<size_t>(mse_threshold)) {
      assert(cnt_ptr);

      orig_cnt_ptr->second--;
      cnt_ptr->second++;
      block.RecalculateEndpoints();
    }
  }

  // Reassign blocks
  for (size_t i = 0; i < _logical_blocks.size(); ++i) {
    _logical_blocks[i] = blocks[i]._logical;
  }

  _physical_blocks = LogicalToPhysicalBlocks(_logical_blocks);
}

}  // namespace GenTC
