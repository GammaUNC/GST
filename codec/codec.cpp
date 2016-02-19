#include "fast_dct.h"
#include "codec.h"
#include "data_stream.h"
#include "dxt_image.h"
#include "image.h"
#include "image_processing.h"
#include "image_utils.h"
#include "pipeline.h"
#include "entropy.h"

#include <iostream>

#include "ans_ocl.h"

namespace GenTC {

std::vector<uint8_t> CompressDXT(const uint8_t *dxt, int width, int height) {
  std::cout << "Original DXT size: " << (width * height / 2) << std::endl;

  DXTImage dxt_img(dxt, width, height);

  auto endpoint_one = dxt_img.EndpointOneImage();
  auto endpoint_two = dxt_img.EndpointTwoImage();

  assert(endpoint_one->Width() == endpoint_two->Width());
  assert(endpoint_one->Height() == endpoint_two->Height());

  auto initial_endpoint_pipeline =
    Pipeline<RGBAImage, RGBImage>
    ::Create(DropAlpha::New())
    ->Chain(std::move(RGBtoYCrCb::New()))
    ->Chain(std::move(YCrCbSplitter::New()));

  auto y_pipeline =
    Pipeline<UnpackedAlphaImage, UnpackedSixteenBitImage>
    ::Create(ForwardDCT<Alpha>::New())
    ->Chain(InspectGrayscale<int16_t, SingleChannel<16>>::New("Y-dct"))
    ->Chain(Quantize8x8<int16_t, SingleChannel<16> >::QuantizeJPEGLuma())
    ->Chain(InspectGrayscale<int16_t, SingleChannel<16>>::New("Y-quantized"))
    ->Chain(Linearize<int16_t, SingleChannel<16> >::New())
    ->Chain(RearrangeStream<int16_t>::New(endpoint_one->Width(), 32))
    ->Chain(RearrangeStream<int16_t>::New(32, 4))
    ->Chain(ShortEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  auto chroma_pipeline =
    Pipeline<UnpackedAlphaImage, UnpackedSixteenBitImage>
    ::Create(ForwardDCT<Alpha>::New())
    ->Chain(InspectGrayscale<int16_t, SingleChannel<16>>::New("Chroma-dct"))
    ->Chain(Quantize8x8<int16_t, SingleChannel<16> >::QuantizeJPEGChroma())
    ->Chain(InspectGrayscale<int16_t, SingleChannel<16>>::New("Chroma-quantized"))
    ->Chain(Linearize<int16_t, SingleChannel<16> >::New())
    ->Chain(RearrangeStream<int16_t>::New(endpoint_one->Width(), 32))
    ->Chain(RearrangeStream<int16_t>::New(32, 4))
    ->Chain(ShortEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  std::unique_ptr<std::array<UnpackedAlphaImage, 3> > ep1_planes =
    initial_endpoint_pipeline->Run(endpoint_one);

  auto ep1_y = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep1_planes->at(0)));
  auto ep1_cr = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep1_planes->at(1)));
  auto ep1_cb = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep1_planes->at(2)));

  std::unique_ptr<std::array<UnpackedAlphaImage, 3> > ep2_planes =
    initial_endpoint_pipeline->Run(endpoint_two);

  auto ep2_y = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep2_planes->at(0)));
  auto ep2_cr = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep2_planes->at(1)));
  auto ep2_cb = std::unique_ptr<UnpackedAlphaImage>(new UnpackedAlphaImage(ep2_planes->at(2)));

  DataStream out;

  auto ep1_y_cmp = y_pipeline->Run(ep1_y);
  out.WriteInt(static_cast<uint32_t>(ep1_y_cmp->size()));

  auto ep1_cr_cmp = chroma_pipeline->Run(ep1_cr);
  out.WriteInt(static_cast<uint32_t>(ep1_cr_cmp->size()));

  auto ep1_cb_cmp = chroma_pipeline->Run(ep1_cb);
  out.WriteInt(static_cast<uint32_t>(ep1_cb_cmp->size()));

  auto ep2_y_cmp = y_pipeline->Run(ep2_y);
  out.WriteInt(static_cast<uint32_t>(ep2_y_cmp->size()));

  auto ep2_cr_cmp = chroma_pipeline->Run(ep2_cr);
  out.WriteInt(static_cast<uint32_t>(ep2_cr_cmp->size()));

  auto ep2_cb_cmp = chroma_pipeline->Run(ep2_cb);
  out.WriteInt(static_cast<uint32_t>(ep2_cb_cmp->size()));

  std::vector<uint8_t> result = out.GetData();
  result.insert(result.end(), ep1_y_cmp->begin(), ep1_y_cmp->end());
  result.insert(result.end(), ep1_cr_cmp->begin(), ep1_cr_cmp->end());
  result.insert(result.end(), ep1_cb_cmp->begin(), ep1_cb_cmp->end());
  result.insert(result.end(), ep2_y_cmp->begin(), ep2_y_cmp->end());
  result.insert(result.end(), ep2_cr_cmp->begin(), ep2_cr_cmp->end());
  result.insert(result.end(), ep2_cb_cmp->begin(), ep2_cb_cmp->end());

  std::cout << "Compressed DXT size: " << result.size() << std::endl;

  return std::move(result);
}

}
