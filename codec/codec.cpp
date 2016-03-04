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

template <typename T>
std::unique_ptr<std::vector<uint8_t> > RunDXTEndpointPipeline(const std::unique_ptr<Image<T> > &img) {
  static_assert(PixelTraits::NumChannels<T>::value,
    "This should operate on each DXT endpoing channel separately");

  static const size_t kNumBits = PixelTraits::BitsUsed<T>::value;
  typedef typename PixelTraits::SignedTypeForBits<kNumBits+2>::Ty WaveletSignedTy;
  typedef typename PixelTraits::UnsignedForSigned<WaveletSignedTy>::Ty WaveletUnsignedTy;

  auto pipeline = Pipeline<Image<T>, Image<WaveletSignedTy> >
    ::Create(FWavelet2D<T, 32>::New())
    ->Chain(MakeUnsigned<WaveletSignedTy>::New())
    ->Chain(Linearize<WaveletUnsignedTy>::New())
    ->Chain(RearrangeStream<WaveletUnsignedTy>::New(img->Width(), 32))
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New())
    ->Chain(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  return std::move(pipeline->Run(img));
}

std::vector<uint8_t> CompressDXT(const char *orig_fn, const char *cmp_fn, int width, int height) {
  std::cout << "Original DXT size: " << (width * height / 2) << std::endl;

  DXTImage dxt_img(width, height, orig_fn, cmp_fn);

  auto endpoint_one = dxt_img.EndpointOneValues();
  auto endpoint_two = dxt_img.EndpointTwoValues();

  assert(endpoint_one->Width() == endpoint_two->Width());
  assert(endpoint_one->Height() == endpoint_two->Height());

  auto initial_endpoint_pipeline =
    Pipeline<RGB565Image, YCoCg667Image>
    ::Create(RGB565toYCoCg667::New())
    ->Chain(std::move(ImageSplit<YCoCg667>::New()));

  auto ep1_planes = initial_endpoint_pipeline->Run(endpoint_one);
  auto ep2_planes = initial_endpoint_pipeline->Run(endpoint_two);

  DataStream out;

  std::cout << "Compressing Y plane for EP 1... ";
  auto ep1_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_y_cmp->size()));
  std::cout << "Done." << std::endl;

  std::cout << "Compressing Co plane for EP 1... ";
  auto ep1_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_co_cmp->size()));
  std::cout << "Done." << std::endl;

  std::cout << "Compressing Cg plane for EP 1... ";
  auto ep1_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_cg_cmp->size()));
  std::cout << "Done." << std::endl;

  std::cout << "Compressing Y plane for EP 2... ";
  auto ep2_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_y_cmp->size()));
  std::cout << "Done." << std::endl;

  std::cout << "Compressing Co plane for EP 2... ";
  auto ep2_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_co_cmp->size()));
  std::cout << "Done." << std::endl;

  std::cout << "Compressing Cg plane for EP 2... ";
  auto ep2_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_cg_cmp->size()));
  std::cout << "Done." << std::endl;

  // !FIXME! Do something with the index data...
  auto index_pipeline =
    Pipeline<std::vector<uint8_t>, std::vector<uint8_t> >
    ::Create(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  //std::unique_ptr<std::vector<uint8_t> > idx_img(
  //  new std::vector<uint8_t>(dxt_img.PredictIndicesLinearize(16, 16)));
  std::unique_ptr<std::vector<uint8_t> > idx_img(
    new std::vector<uint8_t>(dxt_img.InterpolationValues()));
  std::cout << "Compressing symbols... ";
  auto idx_cmp = index_pipeline->Run(idx_img);
  out.WriteInt(static_cast<uint32_t>(idx_cmp->size()));
  std::cout << "Done." << std::endl;

  std::vector<uint8_t> result = out.GetData();
  result.insert(result.end(), ep1_y_cmp->begin(), ep1_y_cmp->end());
  result.insert(result.end(), ep1_co_cmp->begin(), ep1_co_cmp->end());
  result.insert(result.end(), ep1_cg_cmp->begin(), ep1_cg_cmp->end());
  result.insert(result.end(), ep2_y_cmp->begin(), ep2_y_cmp->end());
  result.insert(result.end(), ep2_co_cmp->begin(), ep2_co_cmp->end());
  result.insert(result.end(), ep2_cg_cmp->begin(), ep2_cg_cmp->end());

  result.insert(result.end(), idx_cmp->begin(), idx_cmp->end());

  std::cout << "Interpolation value stats:" << std::endl;
  std::cout << "Uncompressed Size of 2-bit symbols: " << (idx_img->size() * 2) / 8 << std::endl;

  std::vector<size_t> F(256, 0);
  for (auto it = idx_img->begin(); it != idx_img->end(); ++it) {
    F[*it]++;
  }
  size_t M = std::accumulate(F.begin(), F.end(), 0ULL);

  double H = 0;
  for (auto f : F) {
    if (f == 0)
      continue;

    double Ps = static_cast<double>(f);
    H -= Ps * log2(Ps);
  }
  H = log2(static_cast<double>(M)) + (H / static_cast<double>(M));

  std::cout << "H: " << H << std::endl;
  std::cout << "Expected num bytes: " << H*(idx_img->size() / 8) << std::endl;
  std::cout << "Actual num bytes: " << idx_cmp->size() << std::endl;

  double bpp = static_cast<double>(result.size() * 8) / static_cast<double>(width * height);
  std::cout << "Compressed DXT size: " << result.size()
            << " (" << bpp << " bpp)" << std::endl;

  return std::move(result);
}

}
