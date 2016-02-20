#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

#include "image_utils.h"

#include <chrono>
#include <random>
#include <string>
#include <sstream>

namespace GenTC {

DropAlpha::Base::ReturnType DropAlpha::Run(const DropAlpha::Base::ArgType &in) const {
  RGBImage *img = new RGBImage(in->Width(), in->Height());

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      RGBA in_pixel = in->GetAt(i, j);
      RGB out_pixel;
      std::get<0>(out_pixel) = std::get<0>(in_pixel);
      std::get<1>(out_pixel) = std::get<1>(in_pixel);
      std::get<2>(out_pixel) = std::get<2>(in_pixel);

      img->SetAt(i, j, out_pixel);
    }
  }

  return std::move(std::unique_ptr<RGBImage>(img));
}

void WriteAlphaImage(const std::string &fname, size_t w, size_t h,
                     std::vector<uint8_t> &&pixels) {
  unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());
  std::stringstream ss;
  ss << dist(gen) << "-" << fname << ".png";
  
  stbi_write_png(ss.str().c_str(), static_cast<int>(w),
	  static_cast<int>(h), 1, pixels.data(), static_cast<int>(w));
}

}  // namespace GenTC
