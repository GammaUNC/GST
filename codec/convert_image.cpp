#include "convert_image.h"

namespace GenTC {

std::unique_ptr<RGBImage>
ConvertRGBtoRGB565::Run(const std::unique_ptr<RGB565Image> &img) const {
  std::vector<uint8_t> result;
  result.reserve(img->Width() * img->Height() * 3);

  for (size_t j = 0; j < img->Height(); ++j) {
    for (size_t i = 0; i < img->Width(); ++i) {
      std::array<uint32_t, 3> pixel = img->At(i, j);
      uint32_t r = (pixel[0] << 3) | (pixel[0] >> 2);
      result.push_back(static_cast<uint8_t>(r));

      uint32_t g = (pixel[1] << 2) | (pixel[1] >> 4);
      result.push_back(static_cast<uint8_t>(g));

      uint32_t b = (pixel[2] << 3) | (pixel[2] >> 2);
      result.push_back(static_cast<uint8_t>(b));
    }
  }

  return std::move(std::unique_ptr<RGBImage>
                   (new RGBImage(img->Width(), img->Height(), std::move(result))));
}

}  // namespace GenTC
