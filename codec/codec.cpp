#include "fast_dct.h"
#include "codec.h"
#include "data_stream.h"
#include "image.h"
#include "image_processing.h"
#include "image_utils.h"
#include "pipeline.h"
#include "entropy.h"

#include <iostream>

#include "ans_config.h"
#include "ans_ocl.h"
#include "gpu.h"

using gpu::GPUContext;

static const size_t kWaveletBlockDim = 32;

static inline cl_mem_flags GetHostReadOnlyFlags() {
#ifdef CL_VERSION_1_2
  return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
#else
  return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
#endif
}

struct AnsTableEntry {
  cl_ushort freq;
  cl_ushort cum_freq;
  cl_uchar  symbol;
};

struct CLKernelResult {
  cl_mem output;
  cl_event output_event;
};

static CLKernelResult DecodeANS(GPUContext *gpu_ctx, const cl_uchar *data, const size_t num_symbols) {
  cl_kernel build_table_kernel = gpu_ctx->GetOpenCLKernel(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table");

  cl_context ctx = gpu_ctx->GetOpenCLContext();

  // First get the number of frequencies...
  cl_uint num_freqs = static_cast<cl_uint>(data[0]);
  data++;

  // Load all of the frequencies
  cl_uint freqs[256];
  for (cl_uint i = 0; i < num_freqs; ++i) {
    memcpy(freqs + i, data, sizeof(*freqs));
    data += sizeof(*freqs);
  }

  cl_mem_flags flags = GetHostReadOnlyFlags();

  // Note: we could do this on the GPU as well, but the array size here is almost never more than
  // about 256, so the CPU is actually much better at doing it. We can also stick it in constant
  // memory, which makes the upload not that bad...
  cl_uint cum_freqs[256];
  std::partial_sum(freqs, freqs + num_freqs - 1, cum_freqs + 1);

  size_t M = static_cast<size_t>(cum_freqs[num_freqs - 1] + freqs[num_freqs]);

  cl_int errCreateBuffer;
  cl_mem table = clCreateBuffer(gpu_ctx->GetOpenCLContext(),
    CL_MEM_READ_WRITE, M * sizeof(AnsTableEntry), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem freqs_buffer = clCreateBuffer(ctx, flags, num_freqs * sizeof(freqs[0]), freqs, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  const size_t cum_freqs_buf_size = num_freqs * sizeof(cum_freqs[0]);
  cl_mem cum_freqs_buffer = clCreateBuffer(ctx, flags, cum_freqs_buf_size, cum_freqs, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel, 0, sizeof(freqs_buffer), &freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 1, sizeof(cum_freqs_buffer), &cum_freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 2, sizeof(cl_uint), &num_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel, 3, sizeof(table), &table);

  cl_event build_table_event;
  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), build_table_kernel,
                                   1, NULL, &M, NULL, 0, NULL, &build_table_event);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);

  cl_kernel decode_kernel = gpu_ctx->GetOpenCLKernel(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode");

  // First, just set our table buffers...
  CHECK_CL(clSetKernelArg, decode_kernel, 0, sizeof(table), &table);

  // Load all of the offsets to the different data streams...
  cl_uint num_offsets = static_cast<cl_uint>(data[0]);
  data++;

#ifndef NDEBUG
  // Make sure that each offset is a multiple of four, otherwise we won't be getting
  // the values we're expecting in our kernel.
  const cl_uint *debug_offsets = reinterpret_cast<const cl_uint *>(data);
  for (cl_uint i = 0; i < num_offsets; ++i) {
    assert((debug_offsets[i] & 0x3) == 0);
  }
#endif

  cl_uint data_sz = reinterpret_cast<const cl_uint *>(data)[num_offsets - 1];
  cl_mem data_buf = clCreateBuffer(ctx, GetHostReadOnlyFlags(), data_sz,
    const_cast<cl_uchar *>(data), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(data_buf), &data_buf);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif

  // Allocate 256 * num interleaved slots for result
  size_t total_encoded =
    num_offsets * ans::ocl::kThreadsPerEncodingGroup * ans::ocl::kNumEncodedSymbols;
  const size_t total_streams = total_encoded / ans::ocl::kNumEncodedSymbols;
  const size_t streams_per_work_group = ans::ocl::kThreadsPerEncodingGroup;

  CLKernelResult result;
  result.output = clCreateBuffer(ctx, out_flags, total_encoded, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(result.output), &result.output);

  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), decode_kernel,
                                   1, NULL, &total_streams, &streams_per_work_group,
                                   1, &build_table_event, &result.output_event);

  CHECK_CL(clReleaseMemObject, table);
  CHECK_CL(clReleaseMemObject, data_buf);

  return result;
}

static CLKernelResult InverseWavelet(GPUContext *gpu_ctx, CLKernelResult img, int width, int height) {

  CLKernelResult result;
  return result;
}

namespace GenTC {

template <typename T> std::unique_ptr<std::vector<uint8_t> >
RunDXTEndpointPipeline(const std::unique_ptr<Image<T> > &img) {
  static_assert(PixelTraits::NumChannels<T>::value,
    "This should operate on each DXT endpoing channel separately");

  static const size_t kNumBits = PixelTraits::BitsUsed<T>::value;
  typedef typename PixelTraits::SignedTypeForBits<kNumBits+2>::Ty
    WaveletSignedTy;
  typedef typename PixelTraits::UnsignedForSigned<WaveletSignedTy>::Ty
    WaveletUnsignedTy;

  auto pipeline = Pipeline<Image<T>, Image<WaveletSignedTy> >
    ::Create(FWavelet2D<T, kWaveletBlockDim>::New())
    ->Chain(MakeUnsigned<WaveletSignedTy>::New())
    ->Chain(Linearize<WaveletUnsignedTy>::New())
    ->Chain(RearrangeStream<WaveletUnsignedTy>::New(img->Width(), kWaveletBlockDim))
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New())
    ->Chain(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  return std::move(pipeline->Run(img));
}

static DXTImage DecompressDXTImage(const std::vector<uint8_t> &dxt_img) {
  std::cout << std::endl;
  std::cout << "Decompressing DXT Image..." << std::endl;

  DataStream in(dxt_img);
  uint32_t width = in.ReadInt();
  std::cout << "Width: " << width << std::endl;

  uint32_t height = in.ReadInt();
  std::cout << "Height: " << height << std::endl;

  uint32_t ep1_y_cmp_sz = in.ReadInt();
  std::cout << "Endpoint One Y compressed size: " << ep1_y_cmp_sz << std::endl;
  uint32_t ep1_co_cmp_sz = in.ReadInt();
  std::cout << "Endpoint One Co compressed size: " << ep1_co_cmp_sz << std::endl;
  uint32_t ep1_cg_cmp_sz = in.ReadInt();
  std::cout << "Endpoint One Cg compressed size: " << ep1_cg_cmp_sz << std::endl;

  uint32_t ep2_y_cmp_sz = in.ReadInt();
  std::cout << "Endpoint Two Y compressed size: " << ep2_y_cmp_sz << std::endl;
  uint32_t ep2_co_cmp_sz = in.ReadInt();
  std::cout << "Endpoint Two Co compressed size: " << ep2_co_cmp_sz << std::endl;
  uint32_t ep2_cg_cmp_sz = in.ReadInt();
  std::cout << "Endpoint Two Cg compressed size: " << ep2_cg_cmp_sz << std::endl;

  uint32_t palette_sz = in.ReadInt();
  std::cout << "Palette size: " << palette_sz << std::endl;
  uint32_t palette_cmp_sz = in.ReadInt();
  std::cout << "Palette size compressed: " << palette_cmp_sz << std::endl;
  uint32_t indices_cmp_sz = in.ReadInt();
  std::cout << "Palette index deltas compressed: " << indices_cmp_sz << std::endl;

  exit(0);
  return DXTImage(width, height, std::vector<uint8_t>());
}

static std::vector<uint8_t> CompressDXTImage(const DXTImage &dxt_img) {
  std::cout << "Original DXT size: " <<
    (dxt_img.Width() * dxt_img.Height()) / 2 << std::endl;
  std::cout << "Half original DXT size: " <<
    (dxt_img.Width() * dxt_img.Height()) / 4 << std::endl;

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
  out.WriteInt(dxt_img.Width());
  out.WriteInt(dxt_img.Height());

  std::cout << "Compressing Y plane for EP 1... ";
  auto ep1_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_y_cmp->size()));
  std::cout << "Done: " << ep1_y_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Co plane for EP 1... ";
  auto ep1_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_co_cmp->size()));
  std::cout << "Done: " << ep1_co_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Cg plane for EP 1... ";
  auto ep1_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep1_planes));
  out.WriteInt(static_cast<uint32_t>(ep1_cg_cmp->size()));
  std::cout << "Done: " << ep1_cg_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Y plane for EP 2... ";
  auto ep2_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_y_cmp->size()));
  std::cout << "Done: " << ep2_y_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Co plane for EP 2... ";
  auto ep2_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_co_cmp->size()));
  std::cout << "Done: " << ep2_co_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Cg plane for EP 2... ";
  auto ep2_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep2_planes));
  out.WriteInt(static_cast<uint32_t>(ep2_cg_cmp->size()));
  std::cout << "Done: " << ep2_cg_cmp->size() << " bytes" << std::endl;

  auto index_pipeline =
    Pipeline<std::vector<uint8_t>, std::vector<uint8_t> >
    ::Create(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  std::unique_ptr<std::vector<uint8_t> > palette_data(
    new std::vector<uint8_t>(std::move(dxt_img.PaletteData())));
  size_t palette_data_size = palette_data->size();
  std::cout << "Original palette data size: " << palette_data_size << std::endl;
  static const size_t f =
    ans::ocl::kNumEncodedSymbols * ans::ocl::kThreadsPerEncodingGroup;
  size_t padding = ((palette_data_size + (f - 1)) / f) * f;
  std::cout << "Padded palette data size: " << padding << std::endl;
  palette_data->resize(padding, 0);

  std::unique_ptr<std::vector<uint8_t> > idx_data(
    new std::vector<uint8_t>(dxt_img.IndexDiffs()));

  std::cout << "Compressing index palette... ";
  auto palette_cmp = index_pipeline->Run(palette_data);
  out.WriteInt(static_cast<uint32_t>(padding));
  out.WriteInt(static_cast<uint32_t>(palette_cmp->size()));
  std::cout << "Done: " << palette_cmp->size() << " bytes" << std::endl;

  std::cout << "Original index differences size: " << idx_data->size() << std::endl;
  std::cout << "Compressing index differences... ";
  auto idx_cmp = index_pipeline->Run(idx_data);
  out.WriteInt(static_cast<uint32_t>(idx_cmp->size()));
  std::cout << "Done: " << idx_cmp->size()<< " bytes" << std::endl;
  
  std::vector<uint8_t> result = out.GetData();
  result.insert(result.end(), ep1_y_cmp->begin(), ep1_y_cmp->end());
  result.insert(result.end(), ep1_co_cmp->begin(), ep1_co_cmp->end());
  result.insert(result.end(), ep1_cg_cmp->begin(), ep1_cg_cmp->end());
  result.insert(result.end(), ep2_y_cmp->begin(), ep2_y_cmp->end());
  result.insert(result.end(), ep2_co_cmp->begin(), ep2_co_cmp->end());
  result.insert(result.end(), ep2_cg_cmp->begin(), ep2_cg_cmp->end());

  result.insert(result.end(), palette_cmp->begin(), palette_cmp->end());
  result.insert(result.end(), idx_cmp->begin(), idx_cmp->end());

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
  std::cout << "Compressed DXT size: " << result.size()
            << " (" << bpp << " bpp)" << std::endl;

  return std::move(result);
}

std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn,
                                 int width, int height) {
  DXTImage dxt_img(width, height, filename, cmp_fn);
  return std::move(CompressDXTImage(dxt_img));
}

DXTImage DecompressDXT(const std::vector<uint8_t> &cmp_data) {
  return std::move(DecompressDXTImage(cmp_data));
}

void TestDXT(const char *filename, const char *cmp_fn, int width, int height) {
  DXTImage dxt_img(width, height, filename, cmp_fn);
  std::vector<uint8_t> cmp_img(std::move(CompressDXTImage(dxt_img)));

  DXTImage decmp_img = DecompressDXTImage(cmp_img);

  // Check that both dxt_img and decomp_img match...
}

}
