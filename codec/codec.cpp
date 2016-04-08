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

template<typename T>
static std::vector<T> ReadBuffer(cl_command_queue queue, cl_mem buffer, size_t num_elements, cl_event e) {
  std::vector<T> host_mem(num_elements);
  CHECK_CL(clEnqueueReadBuffer, queue, buffer, true, 0, num_elements * sizeof(T), host_mem.data(), 1, &e, NULL);
  return std::move(host_mem);
}

static CLKernelResult DecodeANS(const std::unique_ptr<GPUContext> &gpu_ctx,
                                cl_event init_event, const cl_uchar *data) {
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
  memset(cum_freqs, 0, sizeof(cum_freqs));
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
                                   1, NULL, &M, NULL, 1, &init_event, &build_table_event);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);

  cl_kernel decode_kernel = gpu_ctx->GetOpenCLKernel(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode");

  // Load all of the offsets to the different data streams...
  cl_uint num_offsets = static_cast<cl_uint>(data[0]);
  data++;

#ifndef NDEBUG
  // Make sure we have enough space to use constant buffer...
  assert( static_cast<cl_ulong>(M * sizeof(AnsTableEntry)) <
          gpu_ctx->GetDeviceInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) );

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
  CHECK_CL(clReleaseEvent, build_table_event);

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
  for (cl_uint i = 0; i < img.num_events; ++i) {
    CHECK_CL(clReleaseEvent, img.output_events[i]);
  }

  return result;
}

namespace GenTC {

static CLKernelResult DecompressEndpoints(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          const std::vector<uint8_t> &cmp_data,
                                          int32_t data_sz, size_t *data_offset,
                                          cl_event init_event, cl_int val_offset,
                                          int width, int height) {
  const cl_uchar *ep_cmp_data = reinterpret_cast<const cl_uchar *>(cmp_data.data());
  ep_cmp_data += *data_offset;
  *data_offset += data_sz;

  return InverseWavelet(gpu_ctx, DecodeANS(gpu_ctx, init_event, ep_cmp_data),
                        val_offset, width, height);
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

  assert(y.num_events == 1);
  assert(co.num_events == 1);
  assert(cg.num_events == 1);

  cl_event wait_events[] = {
    y.output_events[0], co.output_events[0], cg.output_events[0]
  };
  const size_t num_wait_events = sizeof(wait_events) / sizeof(wait_events[0]);

  cl_event e;
  CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), ep_kernel,
                                   1, NULL, &num_pixels, NULL,
                                   num_wait_events, wait_events, &e);

  // Release events we don't need anymore
  for (size_t i = 0; i < num_wait_events; ++i) {
    CHECK_CL(clReleaseEvent, wait_events[i]);
  }
  CHECK_CL(clReleaseMemObject, y.output);
  CHECK_CL(clReleaseMemObject, co.output);
  CHECK_CL(clReleaseMemObject, cg.output);

  return e;
}

static cl_event DecodeIndices(const std::unique_ptr<GPUContext> &gpu_ctx, cl_mem dst,
                              cl_event init_event,
                              const std::vector<uint8_t> &cmp_data, size_t offset,
                              size_t num_pixels, uint32_t palette_cmp_sz) {
  const cl_uchar *palette_cmp_data =
    reinterpret_cast<const cl_uchar *>(cmp_data.data()) + offset;
  const cl_uchar *indices_cmp_data = palette_cmp_data + palette_cmp_sz;

  CLKernelResult palette = DecodeANS(gpu_ctx, init_event, palette_cmp_data);
  CLKernelResult indices = DecodeANS(gpu_ctx, init_event, indices_cmp_data);

  cl_kernel idx_kernel = gpu_ctx->GetOpenCLKernel(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices");

  CHECK_CL(clSetKernelArg, idx_kernel, 0, sizeof(indices.output), &indices.output);
  CHECK_CL(clSetKernelArg, idx_kernel, 2, sizeof(dst), &dst);

  static const size_t kLocalScanSz = 128;
  static const size_t kLocalScanSzLog = 7;
  
  // !SPEED! We don't really need to allocate here...
  std::vector<cl_event> e;
  cl_uint event_idx = 0;
  for (size_t i = 0; i < indices.num_events; ++i) {
    e.push_back(indices.output_events[i]);
  }

  cl_int stage = -1;
  size_t num_vals = num_pixels;
  while (num_vals > 1) {
    stage++;
    CHECK_CL(clSetKernelArg, idx_kernel, 1, sizeof(stage), &stage);

    cl_event next_event;
    cl_uint num_events = static_cast<cl_uint>(e.size()) - event_idx;
    size_t local_work_sz = std::min(num_vals, kLocalScanSz);

#ifndef NDEBUG
    assert((num_vals % local_work_sz) == 0);
    assert(local_work_sz <= gpu_ctx->GetKernelWGInfo<size_t>(
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",
      CL_KERNEL_WORK_GROUP_SIZE));
#endif

    CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), idx_kernel,
                                     1, NULL, &num_vals, &local_work_sz,
                                     num_events, e.data() + event_idx,
                                     &next_event);

    num_vals >>= kLocalScanSzLog;
    event_idx += num_events;
    e.push_back(next_event);
  }

  cl_kernel final_kernel = gpu_ctx->GetOpenCLKernel(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "collect_indices");

  CHECK_CL(clSetKernelArg, final_kernel, 0, sizeof(palette.output), &palette.output);
  CHECK_CL(clSetKernelArg, final_kernel, 2, sizeof(dst), &dst);
  for (size_t i = 0; i < palette.num_events; ++i) {
    e.push_back(palette.output_events[i]);
  }

  num_vals = num_pixels;
  while ((num_vals >> kLocalScanSzLog) > 1) {
    num_vals >>= kLocalScanSzLog;
  }

  while (stage >= 0) {
    num_vals = std::min(num_pixels, num_vals << kLocalScanSzLog);

    cl_event next_event;
    cl_uint num_events = static_cast<cl_uint>(e.size()) - event_idx;
    CHECK_CL(clSetKernelArg, final_kernel, 1, sizeof(stage), &stage);
    CHECK_CL(clEnqueueNDRangeKernel, gpu_ctx->GetCommandQueue(), final_kernel,
                                     1, NULL, &num_vals, &kLocalScanSz,
                                     num_events, e.data() + event_idx,
                                     &next_event);
    event_idx += num_events;
    e.push_back(next_event);
    stage--;
  }

  // Don't need these events anymore...
  for (size_t i = 0; i < event_idx; ++i) {
    CHECK_CL(clReleaseEvent, e[i]);
  }
  CHECK_CL(clReleaseMemObject, indices.output);
  CHECK_CL(clReleaseMemObject, palette.output);

  assert(e.size() - event_idx == 1);
  return e[event_idx];
}

static void DecompressDXTImage(const std::unique_ptr<GPUContext> &gpu_ctx,
                               const std::vector<uint8_t> &dxt_img,
                               cl_event init_event, CLKernelResult *result) {
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
                                             &offset, init_event, -128, blocks_x, blocks_y);
  CLKernelResult ep1_co = DecompressEndpoints(gpu_ctx, cmp_data, ep1_co_cmp_sz,
                                              &offset, init_event, -128, blocks_x, blocks_y);
  CLKernelResult ep1_cg = DecompressEndpoints(gpu_ctx, cmp_data, ep1_cg_cmp_sz,
                                              &offset, init_event, -128, blocks_x, blocks_y);

  CLKernelResult ep2_y = DecompressEndpoints(gpu_ctx, cmp_data, ep2_y_cmp_sz,
                                             &offset, init_event, -128, blocks_x, blocks_y);
  CLKernelResult ep2_co = DecompressEndpoints(gpu_ctx, cmp_data, ep2_co_cmp_sz,
                                              &offset, init_event, -128, blocks_x, blocks_y);
  CLKernelResult ep2_cg = DecompressEndpoints(gpu_ctx, cmp_data, ep2_cg_cmp_sz,
                                              &offset, init_event, -128, blocks_x, blocks_y);

  result->num_events = 3;
  const size_t num_blocks = blocks_x * blocks_y;
  result->output_events[0] =
    CollectEndpoints(gpu_ctx, result->output, num_blocks, ep1_y, ep1_co, ep1_cg, 0);
  result->output_events[1] =
    CollectEndpoints(gpu_ctx, result->output, num_blocks, ep2_y, ep2_co, ep2_cg, 1);

  result->output_events[2] = DecodeIndices(gpu_ctx, result->output, init_event, cmp_data,
                                           offset, num_blocks, palette_cmp_sz);
}

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
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New())
    ->Chain(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  return std::move(pipeline->Run(img));
}

static std::vector<uint8_t> CompressDXTImage(const DXTImage &dxt_img) {
  // Otherwise we can't really compress this...
  assert((dxt_img.Width() % 128) == 0);
  assert((dxt_img.Height() % 128) == 0);

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

  std::cout << "Compressing index palette... ";
  auto palette_cmp = index_pipeline->Run(palette_data);
  out.WriteInt(static_cast<uint32_t>(padding));
  out.WriteInt(static_cast<uint32_t>(palette_cmp->size()));
  std::cout << "Done: " << palette_cmp->size() << " bytes" << std::endl;

  std::unique_ptr<std::vector<uint8_t> > idx_data(
    new std::vector<uint8_t>(dxt_img.IndexDiffs()));

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

std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn) {
  DXTImage dxt_img(filename, cmp_fn);
  return std::move(CompressDXTImage(dxt_img));
}

std::vector<uint8_t> CompressDXT(int width, int height, const std::vector<uint8_t> &rgb_data,
                                 const std::vector<uint8_t> &dxt_data) {
  DXTImage dxt_img(width, height, rgb_data, dxt_data);
  return std::move(CompressDXTImage(dxt_img));  
}
  
std::vector<uint8_t>  DecompressDXTBuffer(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          const std::vector<uint8_t> &cmp_data,
                                          uint32_t width, uint32_t height) {
  cl_int errCreateBuffer;
  CLKernelResult decmp;

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_READ_WRITE;
#endif
  size_t dxt_size = (width * height) / 2;
  decmp.output = clCreateBuffer(gpu_ctx->GetOpenCLContext(),
                                out_flags, dxt_size, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Set a dummy event
  cl_event init_event;
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueMarkerWithWaitList, gpu_ctx->GetCommandQueue(), 0, NULL, &init_event);
#else
  CHECK_CL(clEnqueueMarker, gpu_ctx->GetCommandQueue(), &init_event);
#endif

  // Queue the decompression...
  DecompressDXTImage(gpu_ctx, cmp_data, init_event, &decmp);

  // Block on read
  std::vector<uint8_t> decmp_data(dxt_size, 0xFF);
  CHECK_CL(clEnqueueReadBuffer, gpu_ctx->GetCommandQueue(),
                                decmp.output, CL_TRUE, 0, dxt_size, decmp_data.data(),
                                decmp.num_events, decmp.output_events, NULL);

  for (size_t i = 0; i < decmp.num_events; ++i) {
    CHECK_CL(clReleaseEvent, decmp.output_events[i]);
  }
  CHECK_CL(clReleaseMemObject, decmp.output);
  CHECK_CL(clReleaseEvent, init_event);
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
             const char *filename, const char *cmp_fn) {
  DXTImage dxt_img(filename, cmp_fn);
  int width = dxt_img.Width();
  int height = dxt_img.Height();
  
  std::vector<uint8_t> cmp_img = std::move(CompressDXTImage(dxt_img));
  std::vector<uint8_t> decmp_data =
    std::move(DecompressDXTBuffer(gpu_ctx, cmp_img, width, height));

  const std::vector<PhysicalDXTBlock> &blks = dxt_img.PhysicalBlocks();
  for (size_t i = 0; i < blks.size(); ++i) {
    const PhysicalDXTBlock *blk =
      reinterpret_cast<const PhysicalDXTBlock *>(decmp_data.data() + i * 8);
    if (memcmp(&blks[i], blk, 8) != 0) {
      std::cout << "Bad block: " << i << std::endl;
      printf("Original block: 0x%lx\n", blks[i].dxt_block);
      printf("Compressed block: 0x%lx\n", blk->dxt_block);
      return false;
    }
  }
  
  return true;
}

void LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                       const std::vector<uint8_t> &cmp_data, cl_event *e,
                       GLuint texID) {
  CLKernelResult decmp;

  DataStream in(cmp_data);
  uint32_t width = in.ReadInt();
  uint32_t height = in.ReadInt();
  size_t dxt_size = (width * height) / 2;

  // Generate a buffer to plop our data into
  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dxt_size, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // Get an OpenCL handle to pbo memory
  cl_int errCreateFromGL;
  decmp.output = clCreateFromGLBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE,
                                      pbo, &errCreateFromGL);
  CHECK_CL((cl_int), errCreateFromGL);

  // Acquire lock on GL objects...
  cl_event acquire_event;
  CHECK_CL(clEnqueueAcquireGLObjects, gpu_ctx->GetCommandQueue(),
                                      1, &decmp.output, 0, NULL, &acquire_event);

  // Queue the decompression...
  DecompressDXTImage(gpu_ctx, cmp_data, acquire_event, &decmp);

  // Release lock on GL objects...
  cl_event release_event;
  CHECK_CL(clEnqueueReleaseGLObjects, gpu_ctx->GetCommandQueue(),
                                      1, &decmp.output,
                                      decmp.num_events, decmp.output_events,
                                      &release_event);

  // !SPEED! No reason to block here...
  CHECK_CL(clWaitForEvents, 1, &release_event);

  // Copy to the newly created texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBindTexture(GL_TEXTURE_2D, texID);
  glCompressedTexImage2D(
    GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, width, height, 0, dxt_size, 0);
  
#ifndef NDEBUG
  GLint query;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &query);
  assert ( query == GL_TRUE );

  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &query);
  assert ( static_cast<size_t>(query) == dxt_size );

  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &query);
  assert ( query == GL_COMPRESSED_RGB_S3TC_DXT1_EXT );
#endif

  // Unbind the textures
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Don't need the PBO anymore
  glDeleteBuffers(1, &pbo);

  for (size_t i = 0; i < decmp.num_events; ++i) {
    CHECK_CL(clReleaseEvent, decmp.output_events[i]);
  }
  CHECK_CL(clReleaseMemObject, decmp.output);
  CHECK_CL(clReleaseEvent, acquire_event);
  CHECK_CL(clReleaseEvent, release_event);
}

}
