#ifndef __TCAR_IMAGE_PROCESSING_H__
#define __TCAR_IMAGE_PROCESSING_H__

#include "pipeline.h"
#include "image.h"

namespace GenTC {

class RGBtoYCrCb: public PipelineUnit<RGBImage, YCbCrImage> {
 public:
  typedef PipelineUnit<RGBImage, YCbCrImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new RGBtoYCrCb); }
  typename Base::ReturnType Run(const typename Base::ArgType &) const override;
};

class YCrCbtoRGB: public PipelineUnit<YCbCrImage, RGBImage> {
 public:
  typedef PipelineUnit<YCbCrImage, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new YCrCbtoRGB); }
  typename Base::ReturnType Run(const typename Base::ArgType &) const override;
};

class Expand565
: public PipelineUnit<RGB565Image, RGBImage> {
 public:
  typedef PipelineUnit<RGB565Image, RGBImage> Base;
  static std::unique_ptr<Base> New() { return std::unique_ptr<Base>(new Expand565); }
  std::unique_ptr<RGBImage> Run(const std::unique_ptr<RGB565Image> &) const override;
};

}  // namespace GenTC

#endif  // __TCAR_IMAGE_PROCESSING_H__
