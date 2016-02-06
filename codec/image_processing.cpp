#include "image_processing.h"

namespace GenTC {

typename RGBtoYCrCb::Base::ReturnType
RGBtoYCrCb::Run(const typename RGBtoYCrCb::Base::ArgType &in) const {
  YCbCrImage  *ret = new YCbCrImage(in->Width(), in->Height());

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      auto pixel = std::move(in->GetAt(i, j));
      double r = static_cast<double>(pixel[0]);
      double g = static_cast<double>(pixel[1]);
      double b = static_cast<double>(pixel[2]);

      double y = 0.299 * r + 0.587 * g + 0.114 * b;
      double cr = (r - y) * 0.713 + 128.0;
      double cb = (b - y) * 0.564 + 128.0;

      pixel[0] = static_cast<uint32_t>(std::max(0.0, std::min(255.0, y + 0.5)));
      pixel[1] = static_cast<uint32_t>(std::max(0.0, std::min(255.0, cr + 0.5)));
      pixel[2] = static_cast<uint32_t>(std::max(0.0, std::min(255.0, cb + 0.5)));
      ret->SetAt(i, j, std::move(pixel));
    }
  }

  return std::move(std::unique_ptr<YCbCrImage>(ret));
}

typename YCrCbtoRGB::Base::ReturnType
YCrCbtoRGB::Run(const typename YCrCbtoRGB::Base::ArgType &in) const {
  // We need to pack the data ourselves here...
  std::vector<uint8_t> img_data;
  img_data.reserve(3 * in->Width() * in->Height());

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      auto pixel = std::move(in->GetAt(i, j));
      double y = static_cast<double>(pixel[0]);
      double cr = static_cast<double>(pixel[1]);
      double cb = static_cast<double>(pixel[2]);

      double r = y + 1.403 * (cr - 128.0);
      double g = y - 0.714 * (cr - 128.0) - 0.344 * (cb - 128.0);
      double b = y + 1.773 * (cb - 128.0);

      img_data.push_back(static_cast<uint8_t>(std::max(0.0, std::min(255.0, r + 0.5))));
      img_data.push_back(static_cast<uint8_t>(std::max(0.0, std::min(255.0, g + 0.5))));
      img_data.push_back(static_cast<uint8_t>(std::max(0.0, std::min(255.0, b + 0.5))));
    }
  }

  RGBImage *img = new RGBImage(in->Width(), in->Height(), std::move(img_data));
  return std::move(std::unique_ptr<RGBImage>(img));
}

std::unique_ptr<RGBImage> Expand565::Run(const std::unique_ptr<RGB565Image> &in) const {
  const size_t w = in->Width();
  const size_t h = in->Height();

  std::vector<uint8_t> result;
  result.reserve(w * h * 3);

  for (size_t j = 0; j < h; ++j) {
    for (size_t i = 0; i < w; ++i) {
      auto pixel = in->GetAt(i, j);

      uint32_t r = (pixel[0] << 3) | (pixel[0] >> 2);
      result.push_back(static_cast<uint8_t>(r));

      uint32_t g = (pixel[1] << 2) | (pixel[1] >> 4);
      result.push_back(static_cast<uint8_t>(g));

      uint32_t b = (pixel[2] << 3) | (pixel[2] >> 2);
      result.push_back(static_cast<uint8_t>(b));
    }
  }

  RGBImage *img = new RGBImage(w, h, std::move(result));
  return std::move(std::unique_ptr<RGBImage>(img));
}

}  // namespace GenTC
