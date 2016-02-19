#include "image_utils.h"

#include <chrono>
#include <random>
#include <string>
#include <sstream>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

namespace GenTC {

DropAlpha::Base::ReturnType DropAlpha::Run(const DropAlpha::Base::ArgType &in) const {
  std::vector<uint8_t> img_data;
  img_data.reserve(in->Width() * in->Height() * 3);

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      auto pixel = in->GetAt(i, j);
      for (size_t ch = 0; ch < 3; ++ch) {
        assert(pixel[ch] < 256);
        img_data.push_back(static_cast<uint8_t>(pixel[ch]));
      }
    }
  }

  RGBImage *img = new RGBImage(in->Width(), in->Height(), std::move(img_data));
  return std::move(std::unique_ptr<RGBImage>(img));
}

void WriteAlphaImage(const std::string &fname, size_t w, size_t h,
                     std::vector<uint8_t> &&pixels) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());
  std::stringstream ss;
  ss << dist(gen) << "-" << fname << ".png";
  
  stbi_write_png(ss.str().c_str(), w, h, 1, pixels.data(), w);
}

}  // namespace GenTC
