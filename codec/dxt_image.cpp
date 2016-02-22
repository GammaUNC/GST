#include "dxt_image.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <functional>
#include <random>

#include "vptree/vptree.hh"
#define PREDICT_VPTREE

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

template <typename T>
static inline T AbsDiff(T a, T b) {
  return a > b ? a - b : b - a;
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

static bool operator==(const LogicalDXTBlock &p1, const LogicalDXTBlock &p2) {
  return memcmp(&p1, &p2, sizeof(p1)) == 0;
}

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

static std::vector<LogicalDXTBlock> KMeansBlocks(const std::vector<LogicalDXTBlock> &blocks,
                                                 size_t num_clusters) {
  // Generate num_clusters random logical blocks
  std::vector<LogicalDXTBlock> clusters;
  clusters.reserve(num_clusters);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<unsigned> dist(0, 3);

  while (clusters.size() < num_clusters) {

    LogicalDXTBlock blk;
    memset(&blk, 0, sizeof(blk));

    for (int i = 0; i < 16; ++i) {
      blk.indices[i] = dist(gen);
      assert(blk.indices[i] < 4);
    }

    // If we already have this seeded cluster, then generate a new one...
    if (std::find(clusters.begin(), clusters.end(), blk) != clusters.end()) {
      continue;
    }

    clusters.push_back(blk);
  }

  // Allocate data for k-means
  std::vector<std::vector<double> > old_clusters(clusters.size(), std::vector<double>(16, 0.0));
  std::vector<size_t> cluster_idx(blocks.size(), 0);
  std::vector<std::vector<double> > new_clusters(clusters.size(), std::vector<double>(16, 0.0));
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
        tmp[j] = static_cast<double>(blocks[i].indices[j]);
      }

      const std::vector<double> *closest = vp_tree.nearestNeighbors(tmp)[0];

      auto closest_itr = std::find(old_clusters.begin(), old_clusters.end(), *closest);
      size_t idx = closest_itr - old_clusters.begin();
      assert(idx < blocks.size());
      cluster_idx[i] = idx;
    }

    // Generate new clusters based on the index of each cluster...
    for (size_t c = 0; c < clusters.size(); ++c) {
      tmp.assign(16, 0.0);
      double cluster_sz = 0.0;
      for (size_t b = 0; b < blocks.size(); ++b) {
        if (cluster_idx[b] == c) {
          for (size_t i = 0; i < 16; ++i) {
            tmp[i] += static_cast<double>(blocks[b].indices[i]);
          }

          cluster_sz += 1.0;
        }
      }

      for (size_t i = 0; i < 16; ++i) {
        tmp[i] /= cluster_sz;
      }

      new_clusters[c].assign(tmp.begin(), tmp.end());
    }

    bool fixed_point = true;
    for (size_t i = 0; i < new_clusters.size(); ++i) {
      if (old_clusters[i] != new_clusters[i]) {
        continue;
      }
    }

    if (fixed_point) {
      break;
    }

    // Not fixed point, so set each old cluster to a new cluster...
    for (size_t i = 0; i < clusters.size(); ++i) {
      old_clusters[i].assign(new_clusters[i].begin(), new_clusters[i].end());
    }
  }

  // Create discrete clusters by rounding continuous clusters
  for (size_t i = 0; i < clusters.size(); ++i) {
    for (size_t j = 0; j < 16; ++j) {
      clusters[i].indices[j] = static_cast<uint8_t>(old_clusters[i][j] + 0.5);
      assert(0 <= clusters[i].indices[j] && clusters[i].indices[j] < 4);
    }
  }

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

  std::vector<LogicalDXTBlock> blocks = std::move(KMeansBlocks(LogicalBlocks(), 256));

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

}  // namespace GenTC
