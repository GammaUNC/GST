#include "image_processing.h"

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////
//
// Static helper functions
//

static void rgb565_to_ycocg667(const int8_t *in, int8_t *out) {
  int8_t r = in[0];
  int8_t g = in[1];
  int8_t b = in[2];

  assert(0 <= r && r < 32);
  assert(0 <= g && g < 64);
  assert(0 <= b && b < 32);

  out[1] = r - b;
  int8_t t = r + b + (b >> 4);
  out[2] = g - t;
  out[0] = t + (out[2] / 2);

  assert(0 <= out[0] && out[0] < 64);
  assert(-31 <= out[1] && out[1] < 32);
  assert(-63 <= out[2] && out[2] < 64);
}

static void ycocg667_to_rgb565(const int8_t *in, int8_t *out) {
  int8_t y = in[0];
  int8_t co = in[1];
  int8_t cg = in[2];

  assert(0 <= y && y < 64);
  assert(-31 <= co && co < 32);
  assert(-63 <= cg && cg < 64);

  int8_t t = y - (cg / 2);
  out[1] = cg + t;
  out[2] = (t - co) / 2;
  out[0] = out[2] + co;

  assert(0 <= out[0] && out[0] < 32);
  assert(0 <= out[1] && out[1] < 64);
  assert(0 <= out[2] && out[2] < 32);
}

////////////////////////////////////////////////////////////////////////////////
//
// Implementation
//

namespace GenTC {

RGBtoYCrCb::Base::ReturnType RGBtoYCrCb::Run(const RGBtoYCrCb::Base::ArgType &in) const {
  YCbCrImage  *ret = new YCbCrImage(in->Width(), in->Height());

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      RGB pixel = in->GetAt(i, j);
      double r = static_cast<double>(pixel.r);
      double g = static_cast<double>(pixel.g);
      double b = static_cast<double>(pixel.b);

      double y = 0.299 * r + 0.587 * g + 0.114 * b;
      double cr = (r - y) * 0.713 + 128.0;
      double cb = (b - y) * 0.564 + 128.0;

      pixel.r = static_cast<uint32_t>(std::max(0.0, std::min(255.0, y + 0.5)));
      pixel.g = static_cast<uint32_t>(std::max(0.0, std::min(255.0, cr + 0.5)));
      pixel.b = static_cast<uint32_t>(std::max(0.0, std::min(255.0, cb + 0.5)));
      ret->SetAt(i, j, pixel);
    }
  }

  return std::move(std::unique_ptr<YCbCrImage>(ret));
}

YCrCbtoRGB::Base::ReturnType YCrCbtoRGB::Run(const YCrCbtoRGB::Base::ArgType &in) const {
  // We need to pack the data ourselves here...
  std::vector<uint8_t> img_data;
  img_data.reserve(3 * in->Width() * in->Height());

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Width(); ++i) {
      auto pixel = in->GetAt(i, j);
      double y = static_cast<double>(pixel.r);
      double cr = static_cast<double>(pixel.g);
      double cb = static_cast<double>(pixel.b);

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
      RGB565 pixel = in->GetAt(i, j);

      uint32_t r = (pixel.r << 3) | (pixel.r >> 2);
      result.push_back(static_cast<uint8_t>(r));

      uint32_t g = (pixel.g << 2) | (pixel.g >> 4);
      result.push_back(static_cast<uint8_t>(g));

      uint32_t b = (pixel.b << 3) | (pixel.b >> 2);
      result.push_back(static_cast<uint8_t>(b));
    }
  }

  RGBImage *img = new RGBImage(w, h, std::move(result));
  return std::move(std::unique_ptr<RGBImage>(img));
}

std::unique_ptr<YCoCg667Image> RGB565toYCoCg667::Run(const std::unique_ptr<RGB565Image> &in) const {
  const size_t w = in->Width();
  const size_t h = in->Height();

  YCoCg667Image *img = new YCoCg667Image(w, h);

  for (size_t j = 0; j < h; ++j) {
    for (size_t i = 0; i < w; ++i) {
      auto pixel = in->GetAt(i, j);

      assert(0 <= pixel.r && pixel.r < 32);
      assert(0 <= pixel.g && pixel.g < 64);
      assert(0 <= pixel.b && pixel.b < 32);

      int8_t rgb[3] = {
        static_cast<int8_t>(pixel.r),
        static_cast<int8_t>(pixel.g),
        static_cast<int8_t>(pixel.b)
      };

      int8_t ycocg[3];
      rgb565_to_ycocg667(rgb, ycocg);

      YCoCg667 x;
      x.r = ycocg[0];
      x.g = ycocg[1];
      x.b = ycocg[2];

      img->SetAt(i, j, x);
    }
  }

  return std::move(std::unique_ptr<YCoCg667Image>(img));
}

std::unique_ptr<RGB565Image> YCoCg667toRGB565::Run(const std::unique_ptr<YCoCg667Image> &in) const {
  std::vector<uint8_t> data;
  data.reserve(in->Width() * in->Height() * 2);

  for (size_t j = 0; j < in->Height(); ++j) {
    for (size_t i = 0; i < in->Height(); ++i) {
      auto pixel = in->GetAt(i, j);

      assert(0 <= pixel.r && pixel.r < 64);
      assert(-31 <= pixel.g && pixel.g < 32);
      assert(-63 <= pixel.b && pixel.b < 64);

      int8_t ycocg[3] = { pixel.r, pixel.g, pixel.b };
      int8_t rgb[3];
      ycocg667_to_rgb565(ycocg, rgb);

      assert(0 <= rgb[0] && rgb[0] < 32);
      assert(0 <= rgb[1] && rgb[1] < 64);
      assert(0 <= rgb[2] && rgb[2] < 32);

      // Pack it in...
      uint16_t x = 0;
      x |= static_cast<uint16_t>(rgb[0]);
      x <<= 6;
      x |= static_cast<uint16_t>(rgb[1]);
      x <<= 5;
      x |= static_cast<uint16_t>(rgb[2]);

      data.push_back((x >> 8) & 0xFF);
      data.push_back(x & 0xFF);
    }
  }

  RGB565Image *img = new RGB565Image(in->Width(), in->Height(), std::move(data));
  return std::move(std::unique_ptr<RGB565Image>(img));
}

}  // namespace GenTC
