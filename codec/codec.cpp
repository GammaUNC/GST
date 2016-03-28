#include "fast_dct.h"
#include "codec.h"
#include "codec_config.h"
#include "data_stream.h"
#include "image.h"
#include "image_processing.h"
#include "image_utils.h"
#include "pipeline.h"
#include "entropy.h"

#include <iostream>

#include "ans_config.h"
#include "ans_ocl.h"

using gpu::GPUContext;

static const size_t kWaveletBlockDim = 32;
static_assert((kWaveletBlockDim % 2) == 0, "Wavelet dimension must be power of two!");

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
  cl_uint num_events;
  cl_event output_events[8];
};

static CLKernelResult DecodeANS(const std::unique_ptr<GPUContext> &gpu_ctx,
                                const cl_uchar *data) {
  cl_kernel build_table_kernel = gpu_ctx->GetOpenCLKernel(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table");

  cl_context ctx = gpu_ctx->GetOpenCLContext();

  // First get the number of frequencies...
  cl_uint num_freqs = static_cast<cl_uint>(data[0]);

  // Check for overflowed (i.e. the max number of symbols)
  if (num_freqs == 0) {
    num_freqs = 256;
  }

  // Advance the pointer
  data++;

  // Load all of the frequencies
  cl_uint freqs[256];
  for (cl_uint i = 0; i < num_freqs; ++i) {
    memcpy(freqs + i, data, sizeof(*freqs));
    data += sizeof(*freqs);
  }

  cl_mem_flags flags = GetHostReadOnlyFlags();

  // Note: we could do this on the GPU as well, but the array size here is
  // almost never more than about 256, so the CPU is actually much better
  // at doing it. We can also stick it in constant memory, which makes the
  // upload not that bad...
  cl_uint cum_freqs[256];
  std::partial_sum(freqs, freqs + num_freqs - 1, cum_freqs + 1);

  size_t M = static_cast<size_t>(cum_freqs[num_freqs - 1] + freqs[num_freqs - 1]);
  assert(M == ans::ocl::kANSTableSize);

  cl_int errCreateBuffer;
  cl_mem table = clCreateBuffer(gpu_ctx->GetOpenCLContext(),
    CL_MEM_READ_WRITE, M * sizeof(AnsTableEntry), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem freqs_buffer = clCreateBuffer(ctx, flags,
                                       num_freqs * sizeof(freqs[0]),
                                       freqs, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  const size_t cum_freqs_buf_size = num_freqs * sizeof(cum_freqs[0]);
  cl_mem cum_freqs_buffer = clCreateBuffer(ctx, flags, cum_freqs_buf_size, cum_freqs, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel, 0,
                           sizeof(freqs_buffer), &freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 1,
                           sizeof(cum_freqs_buffer), &cum_freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 2, sizeof(cl_uint), &num_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel, 3, sizeof(table), &table);

  cl_event build_table_event;
  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), build_table_kernel,
                                   1, NULL, &M, NULL, 0, NULL, &build_table_event);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);

  cl_kernel decode_kernel = gpu_ctx->GetOpenCLKernel(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode");

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

  // Make sure that we have enough constant memory to allocate here...
  assert(data_sz + 8 * num_freqs + M * sizeof(AnsTableEntry) <
    gpu_ctx->GetDeviceInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE));

  // Allocate 256 * num interleaved slots for result
  size_t total_encoded =
    num_offsets * ans::ocl::kThreadsPerEncodingGroup * ans::ocl::kNumEncodedSymbols;
  const size_t total_streams = total_encoded / ans::ocl::kNumEncodedSymbols;
  const size_t streams_per_work_group = ans::ocl::kThreadsPerEncodingGroup;

  CLKernelResult result;
  result.num_events = 1;
  result.output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_encoded,
                                 NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // First, just set our table buffers...
  CHECK_CL(clSetKernelArg, decode_kernel, 0, sizeof(table), &table);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(data_buf), &data_buf);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(result.output), &result.output);

  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), decode_kernel,
                                   1, NULL, &total_streams, &streams_per_work_group,
                                   1, &build_table_event, result.output_events);

  CHECK_CL(clReleaseMemObject, table);
  CHECK_CL(clReleaseMemObject, data_buf);

  return result;
}

static CLKernelResult InverseWavelet(const std::unique_ptr<GPUContext> &gpu_ctx,
                                     CLKernelResult img, cl_int offset,
                                     int width, int height) {
  assert(width % kWaveletBlockDim == 0);
  assert(height % kWaveletBlockDim == 0);

  cl_int errCreateBuffer;
  cl_context ctx = gpu_ctx->GetOpenCLContext();
  size_t out_sz = width * height;
  size_t local_mem_sz = 8 * kWaveletBlockDim * kWaveletBlockDim;

  cl_kernel wavelet_kernel = gpu_ctx->GetOpenCLKernel(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_InverseWavelet], "inv_wavelet");

  // First, setup our inputs
  CHECK_CL(clSetKernelArg, wavelet_kernel, 0, sizeof(img.output), &img.output);
  CHECK_CL(clSetKernelArg, wavelet_kernel, 1, sizeof(offset), &offset);
  CHECK_CL(clSetKernelArg, wavelet_kernel, 2, local_mem_sz, NULL);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_READ_WRITE;
#endif

  CLKernelResult result;
  result.num_events = 1;
  result.output = clCreateBuffer(ctx, out_flags, out_sz, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, wavelet_kernel, 3, sizeof(result.output), &result.output);

#ifndef NDEBUG
  // One thread per pixel, kWaveletBlockDim * kWaveletBlockDim threads
  // per group...
  size_t threads_per_group = (kWaveletBlockDim / 2) * kWaveletBlockDim;

  // Make sure that we can launch enough kernels per group
  assert(threads_per_group <=
    gpu_ctx->GetDeviceInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE));

  // I don't know of a GPU implementation that uses more than 3 dims..
  assert(3 ==
    gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));

  struct WorkGroupSizes {
    size_t sizes[3];
  };
  WorkGroupSizes wgsz =
    gpu_ctx->GetDeviceInfo<WorkGroupSizes>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  assert( threads_per_group <= wgsz.sizes[0] );
#endif

  size_t global_work_size[2] = {
    static_cast<size_t>(width / 2),
    static_cast<size_t>(height) };

  size_t local_work_size[2] = {
    static_cast<size_t>(kWaveletBlockDim / 2),
    static_cast<size_t>(kWaveletBlockDim) };

  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), wavelet_kernel,
                                   2, NULL, global_work_size, local_work_size,
                                   img.num_events, img.output_events,
                                   result.output_events);

  // No longer need image buffer
  CHECK_CL(clReleaseMemObject, img.output);

  return result;
}

namespace GenTC {

static CLKernelResult DecompressEndpoints(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          const std::vector<uint8_t> &cmp_data,
                                          int32_t data_sz, size_t *data_offset,
                                          cl_int val_offset, int width, int height) {
  const cl_uchar *ep_cmp_data = reinterpret_cast<const cl_uchar *>(cmp_data.data());
  ep_cmp_data += *data_offset;
  *data_offset += data_sz;

  CLKernelResult ep_cmp = DecodeANS(gpu_ctx, ep_cmp_data);
  return InverseWavelet(gpu_ctx, ep_cmp, val_offset, width, height);
}

static cl_event CollectEndpoints(const std::unique_ptr<GPUContext> &gpu_ctx,
                                 cl_mem dst, size_t num_pixels, const CLKernelResult &y,
                                 const CLKernelResult &co, const CLKernelResult &cg,
                                 const cl_uint endpoint_index) {
  cl_kernel ep_kernel = gpu_ctx->GetOpenCLKernel(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_Endpoints], "collect_endpoints");

  // First, setup our inputs
  CHECK_CL(clSetKernelArg, ep_kernel, 0, sizeof(y.output), &y.output);
  CHECK_CL(clSetKernelArg, ep_kernel, 1, sizeof(co.output), &co.output);
  CHECK_CL(clSetKernelArg, ep_kernel, 2, sizeof(cg.output), &cg.output);
  CHECK_CL(clSetKernelArg, ep_kernel, 3, sizeof(dst), &dst);
  CHECK_CL(clSetKernelArg, ep_kernel, 4, sizeof(endpoint_index), &endpoint_index);

  cl_event wait_events[] = {
    y.output_events[0], co.output_events[0], cg.output_events[0]
  };
  const size_t num_wait_events = sizeof(wait_events) / sizeof(wait_events[0]);

  cl_event e;
  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), ep_kernel,
                                   1, NULL, &num_pixels, NULL,
                                   num_wait_events, wait_events, &e);
  return e;
}

static cl_event DecodeIndices(const std::unique_ptr<GPUContext> &gpu_ctx, cl_mem dst,
                              const std::vector<uint8_t> &cmp_data, size_t offset,
                              size_t num_pixels, uint32_t palette_cmp_sz) {

  const cl_uchar *palette_cmp_data = reinterpret_cast<const cl_uchar *>(cmp_data.data());
  palette_cmp_data += offset;

  const cl_uchar *indices_cmp_data = palette_cmp_data + palette_cmp_sz;

  CLKernelResult palette_cmp = DecodeANS(gpu_ctx, palette_cmp_data);
  CLKernelResult indices_cmp = DecodeANS(gpu_ctx, indices_cmp_data);

  cl_kernel idx_kernel = gpu_ctx->GetOpenCLKernel(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices");

  CHECK_CL(clSetKernelArg, idx_kernel, 0, sizeof(palette_cmp.output), &palette_cmp.output);
  CHECK_CL(clSetKernelArg, idx_kernel, 1, sizeof(indices_cmp.output), &indices_cmp.output);
  CHECK_CL(clSetKernelArg, idx_kernel, 2, sizeof(dst), &dst);

  cl_event wait_events[] = {
    palette_cmp.output_events[0],
    indices_cmp.output_events[0],
  };

  const size_t num_wait_events = sizeof(wait_events) / sizeof(wait_events[0]);

  cl_event e;
  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), idx_kernel,
                                   1, NULL, &num_pixels, NULL,
                                   num_wait_events, wait_events, &e);
  return e;
}

static CLKernelResult DecompressDXTImage(const std::unique_ptr<GPUContext> &gpu_ctx,
                                         const std::vector<uint8_t> &dxt_img) {
  std::cout << std::endl;
  std::cout << "Decompressing DXT Image..." << std::endl;

  DataStream in(dxt_img);
  uint32_t width = in.ReadInt();
  uint32_t height = in.ReadInt();

  uint32_t ep1_y_cmp_sz = in.ReadInt();
  uint32_t ep1_co_cmp_sz = in.ReadInt();
  uint32_t ep1_cg_cmp_sz = in.ReadInt();

  uint32_t ep2_y_cmp_sz = in.ReadInt();
  uint32_t ep2_co_cmp_sz = in.ReadInt();
  uint32_t ep2_cg_cmp_sz = in.ReadInt();

#ifndef NDEBUG
  uint32_t palette_sz = in.ReadInt();
#else
  in.ReadInt();
#endif

  uint32_t palette_cmp_sz = in.ReadInt();

#ifndef NDEBUG
  uint32_t indices_cmp_sz = in.ReadInt();
#else
  in.ReadInt();
#endif

#ifndef NDEBUG
  std::cout << "Width: " << width << std::endl;
  std::cout << "Height: " << height << std::endl;
  std::cout << "Endpoint One Y compressed size: " << ep1_y_cmp_sz << std::endl;
  std::cout << "Endpoint One Co compressed size: " << ep1_co_cmp_sz << std::endl;
  std::cout << "Endpoint One Cg compressed size: " << ep1_cg_cmp_sz << std::endl;
  std::cout << "Endpoint Two Y compressed size: " << ep2_y_cmp_sz << std::endl;
  std::cout << "Endpoint Two Co compressed size: " << ep2_co_cmp_sz << std::endl;
  std::cout << "Endpoint Two Cg compressed size: " << ep2_cg_cmp_sz << std::endl;
  std::cout << "Palette size: " << palette_sz << std::endl;
  std::cout << "Palette size compressed: " << palette_cmp_sz << std::endl;
  std::cout << "Palette index deltas compressed: " << indices_cmp_sz << std::endl;
#endif

  const std::vector<uint8_t> &cmp_data = in.GetData();
  size_t offset = in.BytesRead();

  assert((width & 0x3) == 0);
  assert((height & 0x3) == 0);

  const int blocks_x = static_cast<int>(width) >> 2;
  const int blocks_y = static_cast<int>(height) >> 2;

  CLKernelResult ep1_y = DecompressEndpoints(gpu_ctx, cmp_data, ep1_y_cmp_sz,
                                             &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep1_co = DecompressEndpoints(gpu_ctx, cmp_data, ep1_co_cmp_sz,
                                              &offset, -64, blocks_x, blocks_y);
  CLKernelResult ep1_cg = DecompressEndpoints(gpu_ctx, cmp_data, ep1_cg_cmp_sz,
                                              &offset, -128, blocks_x, blocks_y);

  CLKernelResult ep2_y = DecompressEndpoints(gpu_ctx, cmp_data, ep2_y_cmp_sz,
                                             &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep2_co = DecompressEndpoints(gpu_ctx, cmp_data, ep2_co_cmp_sz,
                                              &offset, -64, blocks_x, blocks_y);
  CLKernelResult ep2_cg = DecompressEndpoints(gpu_ctx, cmp_data, ep2_cg_cmp_sz,
                                              &offset, -128, blocks_x, blocks_y);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif
  
  size_t dxt_size = blocks_x * blocks_y * 8;
  cl_int errCreateBuffer;

  CLKernelResult result;
  result.num_events = 3;
  result.output = clCreateBuffer(gpu_ctx->GetOpenCLContext(),
                                 out_flags, dxt_size, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  size_t num_blocks = blocks_x * blocks_y;
  result.output_events[0] =
    CollectEndpoints(gpu_ctx, result.output, num_blocks, ep1_y, ep1_co, ep1_cg, 0);
  result.output_events[1] =
    CollectEndpoints(gpu_ctx, result.output, num_blocks, ep2_y, ep2_co, ep2_cg, 1);

  result.output_events[2] = DecodeIndices(gpu_ctx, result.output, cmp_data,
                                          offset, num_blocks, palette_cmp_sz);

  return result;
}

template <typename T> std::unique_ptr<std::vector<uint8_t> >
RunDXTEndpointPipeline(const std::unique_ptr<Image<T> > &img) {
  static_assert(PixelTraits::NumChannels<T>::value,
    "This should operate on each DXT endpoing channel separately");

  const bool kIsSigned = PixelTraits::IsSigned<T>::value;
  typedef typename WaveletResultTy<T, kIsSigned>::DstTy WaveletSignedTy;
  typedef typename PixelTraits::UnsignedForSigned<WaveletSignedTy>::Ty WaveletUnsignedTy;

  auto pipeline = Pipeline<Image<T>, Image<WaveletSignedTy> >
    ::Create(FWavelet2D<T, kWaveletBlockDim>::New())
    ->Chain(MakeUnsigned<WaveletSignedTy>::New())
    ->Chain(Linearize<WaveletUnsignedTy>::New())
    ->Chain(RearrangeStream<WaveletUnsignedTy>::New(img->Width(), kWaveletBlockDim))
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New())
    ->Chain(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  return std::move(pipeline->Run(img));
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


  
std::vector<uint8_t>  DecompressDXTBuffer(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          const std::vector<uint8_t> &cmp_data,
                                          uint32_t width, uint32_t height) {
  CLKernelResult decmp = DecompressDXTImage(gpu_ctx, cmp_data);

  size_t dxt_size = (width * height) / 2;
  std::vector<uint8_t> decmp_data(dxt_size);
  CHECK_CL(clEnqueueReadBuffer, gpu_ctx->GetCommandQueue(),
                                decmp.output, true,
                                0, dxt_size, decmp_data.data(),
                                decmp.num_events, decmp.output_events,
                                NULL);

  return std::move(decmp_data);
}

DXTImage DecompressDXT(const std::unique_ptr<GPUContext> &gpu_ctx,
                       const std::vector<uint8_t> &cmp_data) {
  DataStream in(cmp_data);
  uint32_t width = in.ReadInt();
  uint32_t height = in.ReadInt();

  std::vector<uint8_t> decmp_data =
    std::move(DecompressDXTBuffer(gpu_ctx, cmp_data, width, height));
  return DXTImage(width, height, decmp_data);
}

bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
             const char *filename, const char *cmp_fn, int width, int height) {
  DXTImage dxt_img(width, height, filename, cmp_fn);
  
  std::vector<uint8_t> cmp_img = std::move(CompressDXTImage(dxt_img));
  std::vector<uint8_t> decmp_data =
    std::move(DecompressDXTBuffer(gpu_ctx, cmp_img, width, height));
  
  // Check that both dxt_img and decomp_img match...
  return 0 == memcmp(dxt_img.PhysicalBlocks().data(),
                     decmp_data.data(), decmp_data.size());
}

}
