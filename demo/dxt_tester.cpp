#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

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

struct Pixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;

  static Pixel From565(uint16_t x) {
    Pixel p;
    p.r = static_cast<uint8_t>((x >> 11) & 0x1F);
    p.r = (p.r << 3) | (p.r >> 2);

    p.g = static_cast<uint8_t>((x >> 5) & 0x3F);
    p.g = (p.g << 2) | (p.g >> 4);

    p.b = static_cast<uint8_t>(x & 0x1F);
    p.b = (p.b << 3) | (p.b >> 2);
    return p;
  }
};

enum EColorSpace {
  eColorSpace_RGB,
  eColorSpace_YCoCg
};

struct Color {
  EColorSpace color_space;
  float r;
  float g;
  float b;

  Color()
    : color_space(eColorSpace_RGB)
    , r(0.0f)
    , g(0.0f)
    , b(0.0f) { }

  void ConvertTo(EColorSpace cs) {
    Color c = *this;
    if (c.color_space == cs) {
      return;
    }

    switch(cs) {
      case eColorSpace_YCoCg:
      {
        c.r = 0.25 * r + 0.5 * g + 0.25 * b;
        c.g = -0.25 * r + 0.5 * g - 0.25 * b;
        c.b = 0.5 * r + 0.5 * b;
        c.color_space = eColorSpace_YCoCg;
      }
      break;

      case eColorSpace_RGB:
      {
        c.r = r - g + b;
        c.g = r + g;
        c.b = r - g - b;
        c.color_space = eColorSpace_RGB;
      }
      break;
    }

    *this = c;
  }

  Pixel ToPixel() const {
    Pixel x;
    if (color_space == eColorSpace_RGB) {
      x.r = static_cast<uint8_t>((r * 255.0f) + 0.5f);
      x.g = static_cast<uint8_t>((g * 255.0f) + 0.5f);
      x.b = static_cast<uint8_t>((b * 255.0f) + 0.5f);
    } else {
      x.r = static_cast<uint8_t>(r * 255.0f + 0.5f);
      x.g = static_cast<uint8_t>((g + 0.5) * 255.0f + 0.5f);
      x.b = static_cast<uint8_t>((b + 0.5) * 255.0f + 0.5f);
    }
    return x;
  }

  static Color FromPixel(const Pixel &p) {
    Color result;
    result.r = static_cast<float>(p.r) / 255.0f;
    result.g = static_cast<float>(p.g) / 255.0f;
    result.b = static_cast<float>(p.b) / 255.0f;
    result.color_space = eColorSpace_RGB; // !KLUDGE!
    return result;
  }

  static Color From565(uint16_t x) {
    return FromPixel(Pixel::From565(x));
  }
};

union DXTBlock {
  struct {
    uint16_t ep1;
    uint16_t ep2;
    uint32_t interpolation;
  };
  uint64_t dxt_block;
};

void SplitChannels(const std::vector<Pixel> &in,
                   std::vector<uint8_t> *r,
                   std::vector<uint8_t> *g,
                   std::vector<uint8_t> *b) {
  assert(r); assert(g); assert(b);

  r->clear();
  r->reserve(in.size());

  g->clear();
  g->reserve(in.size());

  b->clear();
  b->reserve(in.size());

  for (const auto &p : in) {
    r->push_back(p.r);
    g->push_back(p.g);
    b->push_back(p.b);
  }
}

int main(int argc, char **argv) {
  // Make sure that we have the proper number of arguments...
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  // Otherwise, load the file
  int w = 0, h = 0, channels = 0;
  unsigned char *data = stbi_load(argv[1], &w, &h, &channels, 0);
  if (!data) {
    std::cerr << "Error loading image: " << argv[1] << std::endl;
    return 1;
  }

  assert((w & 0x3) == 0);
  assert((h & 0x3) == 0);

  const int num_blocks_x = (w + 3) / 4;
  const int num_blocks_y = (h + 3) / 4;
  const int num_blocks = num_blocks_x * num_blocks_y;

  // Now do the dxt compression...
  std::vector<DXTBlock> dxt_blocks(num_blocks);

  for (int j = 0; j < h; j += 4) {
    for (int i = 0; i < w; i += 4) {
      int block_idx = (j / 4) * num_blocks_x + (i / 4);
      unsigned char *offset_data = data + (j*w + i) * channels;
      if (3 == channels) {
        dxt_blocks[block_idx].dxt_block = CompressRGB(offset_data, w);
      } else if (4 == channels) {
        dxt_blocks[block_idx].dxt_block = CompressRGBA(offset_data, w);
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
  std::vector<uint8_t> interp_vis(w*h);
  for (int j = 0; j < h; j+=4) {
    for (int i = 0; i < w; i+=4) {
      int block_idx = (j / 4) * num_blocks_x + (i / 4);
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          int pixel_idx = ((j + y) * w) + i + x;
          int local_pixel_idx = y * 4 + x;
          int interp = (dxt_blocks[block_idx].interpolation >> local_pixel_idx) & 0x3;
          interp_vis[pixel_idx] = interp_map[interp];
        }
      }
    }
  }

  std::vector<Pixel> img_A(num_blocks_x * num_blocks_y);
  std::vector<Pixel> img_B(num_blocks_x * num_blocks_y);
  for (int j = 0; j < num_blocks_y; j ++) {
    for (int i = 0; i < num_blocks_x; i ++) {
      int block_idx = j * num_blocks_x + i;
      img_A[block_idx] = Pixel::From565(dxt_blocks[block_idx].ep1);
      img_B[block_idx] = Pixel::From565(dxt_blocks[block_idx].ep2);
    }
  }

  stbi_write_png("dxt_interp_vis.png", w, h, 1, interp_vis.data(), w);
  stbi_write_png("dxt_img_A.png", num_blocks_x, num_blocks_y, 3, img_A.data(), num_blocks_x * sizeof(Pixel));
  stbi_write_png("dxt_img_B.png", num_blocks_x, num_blocks_y, 3, img_B.data(), num_blocks_x * sizeof(Pixel));

  std::transform(img_A.begin(), img_A.end(), img_A.begin(),
                 [](const Pixel &p) {
                   Color c = Color::FromPixel(p);
                   c.ConvertTo(eColorSpace_YCoCg);
                   return c.ToPixel();
                 });

  std::vector<uint8_t> y_channel;
  std::vector<uint8_t> co_channel;
  std::vector<uint8_t> cg_channel;

  SplitChannels(img_A, &y_channel, &co_channel, &cg_channel);

  stbi_write_png("dxt_img_A_Y.png", num_blocks_x, num_blocks_y, 1, y_channel.data(), num_blocks_x);
  stbi_write_png("dxt_img_A_Co.png", num_blocks_x, num_blocks_y, 1, co_channel.data(), num_blocks_x);
  stbi_write_png("dxt_img_A_Cg.png", num_blocks_x, num_blocks_y, 1, cg_channel.data(), num_blocks_x);

  std::transform(img_B.begin(), img_B.end(), img_B.begin(),
                 [](const Pixel &p) {
                   Color c = Color::FromPixel(p);
                   c.ConvertTo(eColorSpace_YCoCg);
                   return c.ToPixel();
                 });

  SplitChannels(img_B, &y_channel, &co_channel, &cg_channel);

  stbi_write_png("dxt_img_B_Y.png", num_blocks_x, num_blocks_y, 1, y_channel.data(), num_blocks_x);
  stbi_write_png("dxt_img_B_Co.png", num_blocks_x, num_blocks_y, 1, co_channel.data(), num_blocks_x);
  stbi_write_png("dxt_img_B_Cg.png", num_blocks_x, num_blocks_y, 1, cg_channel.data(), num_blocks_x);

  stbi_image_free(data);
  return 0;
}
