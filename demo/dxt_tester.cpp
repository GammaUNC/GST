#include <cassert>
#include <cstdint>
#include <iostream>

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

  const int num_blocks_x = (w + 3) / 4;
  const int num_blocks_y = (h + 3) / 4;
  const int num_blocks = num_blocks_x * num_blocks_y;

  // Now do the dxt compression...
  union DXTBlock {
    struct {
      uint16_t ep1;
      uint16_t ep2;
      uint32_t interpolation;
    };
    uint64_t dxt_block;
  } *dxt_blocks = new DXTBlock[num_blocks];

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
  uint8_t *interp_vis = new uint8_t[w*h];
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
  
  stbi_write_png("dxt_tester_vis.png", w, h, 1, interp_vis, w);

  delete interp_vis;
  stbi_image_free(data);
  delete dxt_blocks;
  return 0;
}
