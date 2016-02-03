#ifndef __CODEC_CONVERT_IMAGE_H__
#define __CODEC_CONVERT_IMAGE_H__

#include "image.h"
#include "pipeline.h"

namespace GenTC {

class ConvertRGBtoRGB565
: public PipelineUnit<RGB565Image, RGBImage> {
 public:
   ConvertRGBtoRGB565() : PipelineUnit<RGB565Image, RGBImage>() { }
   virtual ~ConvertRGBtoRGB565() { }

  static std::unique_ptr<PipelineUnit<RGB565Image, RGBImage> > New() {
    return std::unique_ptr<PipelineUnit<RGB565Image, RGBImage> >(new ConvertRGBtoRGB565());
  }

  std::unique_ptr<RGBImage> Run(const std::unique_ptr<RGB565Image> &) const override;
};

}  // namespace GenTC {

#endif  // __CODEC_CONVERT_IMAGE_H__
