#ifndef __TCAR_DXT_IMAGE_H__
#define __TCAR_DXT_IMAGE_H__

#include <array>
#include <cstdint>
#include <vector>

namespace GenTC {

  union PhysicalDXTBlock {
    struct {
      uint16_t ep1;
      uint16_t ep2;
      uint32_t interpolation;
    };
    uint64_t dxt_block;
  };

  struct LogicalDXTBlock {
    uint8_t ep1[4];
    uint8_t ep2[4];
    uint8_t palette[4][4];
    uint8_t indices[16];
  };

  class DXTImage {
   public:
    DXTImage(const uint8_t *dxt_image, int width, int height);

    int Width() const { return _width;  }
    int Height() const { return _height;  }

    int BlocksWide() const { return _blocks_width;  }
    int BlocksHigh() const { return _blocks_height; }

    // RGBA image
    std::vector<uint8_t> EndpointOneImage() const;
    std::vector<uint8_t> EndpointTwoImage() const;
    std::vector<uint8_t> DecompressedImage() const;

    // RGB 565 separated into bytes
    std::vector<uint8_t> EndpointOneValues() const;
    std::vector<uint8_t> EndpointTwoValues() const;

    static std::vector<uint8_t> TwoBitValuesToImage(const std::vector<uint8_t> &v);

    // Byte-wise image where each byte takes on of four values in [0, 255]
    std::vector<uint8_t> InterpolationImage() const {
      return std::move(TwoBitValuesToImage(InterpolationValues()));
    }

    // Original interpolation values
    std::vector<uint8_t> InterpolationValues() const;

    const std::vector<PhysicalDXTBlock> &PhysicalBlocks() const {
      return _physical_blocks;
    }

    const std::vector<LogicalDXTBlock> &LogicalBlocks() const {
      return _logical_blocks;
    }

    const LogicalDXTBlock &LogicalBlockAt(int x, int y) const {
      return _logical_blocks[BlockAt(x, y)];
    }

    const PhysicalDXTBlock &PhysicalBlockAt(int x, int y) const {
      return _physical_blocks[BlockAt(x, y)];
    }

    uint8_t InterpolationValueAt(int x, int y) const;
    void DXTImage::GetColorAt(int x, int y, uint8_t out[4]) const;

    std::vector<uint8_t> DXTImage::PredictIndices() const;

  private:
    uint32_t BlockAt(int x, int y) const {
      return (y / 4) * _blocks_width + (x / 4);
    }

    const int _width;
    const int _height;
    const int _blocks_width;
    const int _blocks_height;

    const std::vector<PhysicalDXTBlock> _physical_blocks;
    const std::vector<LogicalDXTBlock> _logical_blocks;
  };

}  // namespace GenTC

#endif