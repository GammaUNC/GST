#include "fast_dct.h"

#include "encoder.h"

#include "codec_base.h"
#include "data_stream.h"
#include "image.h"
#include "image_processing.h"
#include "image_utils.h"
#include "pipeline.h"
#include "entropy.h"

#include <atomic>
#include <iostream>

#include "ans.h"

namespace GenTC {

template <typename T> std::unique_ptr<std::vector<uint8_t> >
RunDXTEndpointPipeline(const std::unique_ptr<Image<T> > &img) {
  static_assert(PixelTraits::NumChannels<T>::value,
    "This should operate on each DXT endpoing channel separately");

  const bool kIsSixBits = PixelTraits::BitsUsed<T>::value == 6;
  typedef typename WaveletResultTy<T, kIsSixBits>::DstTy WaveletSignedTy;
  typedef typename PixelTraits::UnsignedForSigned<WaveletSignedTy>::Ty WaveletUnsignedTy;

  auto pipeline = Pipeline<Image<T>, Image<WaveletSignedTy> >
    ::Create(FWavelet2D<T, kWaveletBlockDim>::New())
    ->Chain(MakeUnsigned<WaveletSignedTy>::New())
    ->Chain(Linearize<WaveletUnsignedTy>::New())
    ->Chain(RearrangeStream<WaveletUnsignedTy>::New(img->Width(), kWaveletBlockDim))
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New());

  return std::move(pipeline->Run(img));
}

static std::vector<uint8_t> CompressDXTImage(const DXTImage &dxt_img) {
  // Otherwise we can't really compress this...
  assert((dxt_img.Width() % 128) == 0);
  assert((dxt_img.Height() % 128) == 0);

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

  std::cout << "Processing Y plane for EP 1... ";
  auto ep1_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Processing Co plane for EP 1... ";
  auto ep1_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Processing Cg plane for EP 1... ";
  auto ep1_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Processing Y plane for EP 2... ";
  auto ep2_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Processing Co plane for EP 2... ";
  auto ep2_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Processing Cg plane for EP 2... ";
  auto ep2_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  auto cmp_pipeline =
    Pipeline<std::vector<uint8_t>, std::vector<uint8_t> >
    ::Create(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  // Concatenate Y planes
  ep1_y_cmp->insert(ep1_y_cmp->end(), ep2_y_cmp->begin(), ep2_y_cmp->end());
  std::cout << "Compressing luma planes (" << ep1_y_cmp->size() << " bytes)...";
  auto y_planes = cmp_pipeline->Run(ep1_y_cmp);
  std::cout << "Done. (" << y_planes->size() << " bytes)" << std::endl;

  // Concatenate Chroma planes
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep1_cg_cmp->begin(), ep1_cg_cmp->end());
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep2_co_cmp->begin(), ep2_co_cmp->end());
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep2_cg_cmp->begin(), ep2_cg_cmp->end());
  std::cout << "Compressing chroma planes (" << ep1_co_cmp->size() << " bytes)...";
  auto chroma_planes = cmp_pipeline->Run(ep1_co_cmp);
  std::cout << "Done. (" << chroma_planes->size() << " bytes)" << std::endl;

  std::unique_ptr<std::vector<uint8_t> > palette_data(
    new std::vector<uint8_t>(std::move(dxt_img.PaletteData())));
  size_t palette_data_size = palette_data->size();
  std::cout << "Original palette data size: " << palette_data_size << std::endl;
  static const size_t f =
    ans::ocl::kNumEncodedSymbols * ans::ocl::kThreadsPerEncodingGroup;
  size_t padding = ((palette_data_size + (f - 1)) / f) * f;
  std::cout << "Padded palette data size: " << padding << std::endl;
  palette_data->resize(padding, 0);

  std::cout << "Compressing index palette... ";
  auto palette_cmp = cmp_pipeline->Run(palette_data);
  std::cout << "Done: " << palette_cmp->size() << " bytes" << std::endl;

  std::unique_ptr<std::vector<uint8_t> > idx_data(
    new std::vector<uint8_t>(dxt_img.IndexDiffs()));

  std::cout << "Original index differences size: " << idx_data->size() << std::endl;
  std::cout << "Compressing index differences... ";
  auto idx_cmp = cmp_pipeline->Run(idx_data);
  std::cout << "Done: " << idx_cmp->size() << " bytes" << std::endl;

  GenTCHeader hdr;
  hdr.width = dxt_img.Width();
  hdr.height = dxt_img.Height();
  hdr.palette_bytes = static_cast<uint32_t>(palette_data->size());
  hdr.y_cmp_sz = static_cast<uint32_t>(y_planes->size()) - 512;
  hdr.chroma_cmp_sz = static_cast<uint32_t>(chroma_planes->size()) - 512;
  hdr.palette_sz = static_cast<uint32_t>(palette_cmp->size()) - 512;
  hdr.indices_sz = static_cast<uint32_t>(idx_cmp->size()) - 512;

  std::vector<uint8_t> result(sizeof(hdr), 0);
  memcpy(result.data(), &hdr, sizeof(hdr));

  // Input the frequencies first
  result.insert(result.end(), y_planes->begin(), y_planes->begin() + 512);
  result.insert(result.end(), chroma_planes->begin(), chroma_planes->begin() + 512);
  result.insert(result.end(), palette_cmp->begin(), palette_cmp->begin() + 512);
  result.insert(result.end(), idx_cmp->begin(), idx_cmp->begin() + 512);
  
  // Input the compressed streams next
  result.insert(result.end(), y_planes->begin() + 512, y_planes->end());
  result.insert(result.end(), chroma_planes->begin() + 512, chroma_planes->end());
  result.insert(result.end(), palette_cmp->begin() + 512, palette_cmp->end());
  result.insert(result.end(), idx_cmp->begin() + 512, idx_cmp->end());

#if 0
  std::cout << "Interpolation value stats:" << std::endl;
  std::cout << "Uncompressed Size of 2-bit symbols: " <<
    (idx_img->size() * 2) / 8 << std::endl;

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
#endif

  double bpp = static_cast<double>(result.size() * 8) /
    static_cast<double>(dxt_img.Width() * dxt_img.Height());
  std::cout << "Original DXT size: " <<
	  (dxt_img.Width() * dxt_img.Height()) / 2 << std::endl;
  std::cout << "Compressed DXT size: " << result.size()
            << " (" << bpp << " bpp)" << std::endl;

  return std::move(result);
}

std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn) {
  DXTImage dxt_img(filename, cmp_fn);
  return std::move(CompressDXTImage(dxt_img));
}

std::vector<uint8_t> CompressDXT(int width, int height, const std::vector<uint8_t> &rgb_data,
                                 const std::vector<uint8_t> &dxt_data) {
  DXTImage dxt_img(width, height, rgb_data, dxt_data);
  return std::move(CompressDXTImage(dxt_img));
}

std::vector<uint8_t> CompressDXT(const DXTImage &dxt_img) {
  return std::move(CompressDXTImage(dxt_img));
}

}
