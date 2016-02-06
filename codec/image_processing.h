#ifndef __CODEC_IMAGE_PROCESSING_H__
#define __CODEC_IMAGE_PROCESSING_H__

#include "pipeline.h"
#include "image.h"

namespace GenTC {

class RGBtoYCrCb: public PipelineUnit<RGBImage, YCbCrImage> {
 public:
  typedef PipelineUnit<RGBImage, YCbCrImage> Base;
  typename Base::ReturnType Run(const typename Base::ArgType &) const override;
};

class YCrCbtoRGB: public PipelineUnit<YCbCrImage, RGBImage> {
 public:
  typedef PipelineUnit<YCbCrImage, RGBImage> Base;
  typename Base::ReturnType Run(const typename Base::ArgType &) const override;
};

}  // namespace GenTC

#endif  // __CODEC_IMAGE_PROCESSING_H__
