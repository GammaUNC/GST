#include "fast_dct.h"
#include "codec.h"
#include "codec_config.h"
#include "data_stream.h"
#include "image.h"
#include "image_processing.h"
#include "image_utils.h"
#include "pipeline.h"
#include "entropy.h"

#include <atomic>
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

struct CLSubRegion {
  size_t origin;
  size_t size;
};

template<typename T>
static std::vector<T> ReadBuffer(cl_command_queue queue, cl_mem buffer, size_t num_elements, cl_event e) {
  std::vector<T> host_mem(num_elements);
  CHECK_CL(clEnqueueReadBuffer, queue, buffer, true, 0, num_elements * sizeof(T), host_mem.data(), 1, &e, NULL);
  return std::move(host_mem);
}

static const int kCommandQueueListSz = 8;
typedef cl_command_queue CommandQueueList[kCommandQueueListSz];

namespace GenTC {

////////////////////////////////////////////////////////////////////////////////
//
// Encoder
//
////////////////////////////////////////////////////////////////////////////////

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

static GenTCHeader::rANSInfo GetCompressedInfo(const std::unique_ptr<std::vector<uint8_t> > &cmp) {
  const uint32_t *data_as_ints = reinterpret_cast<const uint32_t *>(cmp->data());

  GenTCHeader::rANSInfo info;
  info.sz = static_cast<uint32_t>(cmp->size());
  info.num_freqs = data_as_ints[0];
  info.num_offsets = data_as_ints[1 + info.num_freqs];
  return info;
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

  std::cout << "Compressing Y plane for EP 1... ";
  auto ep1_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep1_planes));
  std::cout << "Done: " << ep1_y_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Co plane for EP 1... ";
  auto ep1_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep1_planes));
  std::cout << "Done: " << ep1_co_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Cg plane for EP 1... ";
  auto ep1_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep1_planes));
  std::cout << "Done: " << ep1_cg_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Y plane for EP 2... ";
  auto ep2_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep2_planes));
  std::cout << "Done: " << ep2_y_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Co plane for EP 2... ";
  auto ep2_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep2_planes));
  std::cout << "Done: " << ep2_co_cmp->size() << " bytes" << std::endl;

  std::cout << "Compressing Cg plane for EP 2... ";
  auto ep2_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep2_planes));
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
  std::cout << "Done: " << palette_cmp->size() << " bytes" << std::endl;

  std::unique_ptr<std::vector<uint8_t> > idx_data(
    new std::vector<uint8_t>(dxt_img.IndexDiffs()));

  std::cout << "Original index differences size: " << idx_data->size() << std::endl;
  std::cout << "Compressing index differences... ";
  auto idx_cmp = index_pipeline->Run(idx_data);
  std::cout << "Done: " << idx_cmp->size() << " bytes" << std::endl;

  GenTCHeader hdr;
  hdr.width = dxt_img.Width();
  hdr.height = dxt_img.Height();
  hdr.ep1_y = GetCompressedInfo(ep1_y_cmp);
  hdr.ep1_co = GetCompressedInfo(ep1_co_cmp);
  hdr.ep1_cg = GetCompressedInfo(ep1_cg_cmp);
  hdr.ep2_y = GetCompressedInfo(ep2_y_cmp);
  hdr.ep2_co = GetCompressedInfo(ep2_co_cmp);
  hdr.ep2_cg = GetCompressedInfo(ep2_cg_cmp);
  hdr.palette = GetCompressedInfo(palette_cmp);
  hdr.indices = GetCompressedInfo(idx_cmp);

  std::vector<uint8_t> result(sizeof(hdr), 0);
  memcpy(result.data(), &hdr, sizeof(hdr));
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

////////////////////////////////////////////////////////////////////////////////
//
// Decoder
//
////////////////////////////////////////////////////////////////////////////////

static CLKernelResult DecodeANS(const std::unique_ptr<GPUContext> &gpu_ctx, cl_command_queue queue,
                                cl_mem input, const size_t offset, const GenTCHeader::rANSInfo &info) {
  // First get the number of frequencies...
  cl_uint num_freqs = info.num_freqs;
  const size_t M = ans::ocl::kANSTableSize;
  size_t build_table_local_work_size = 256;
  assert(build_table_local_work_size <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",
    CL_KERNEL_WORK_GROUP_SIZE));

  cl_int errCreateBuffer;
  cl_mem table = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE,
                                M * sizeof(AnsTableEntry), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CLSubRegion freqs_sub_region;
  freqs_sub_region.origin = offset;
  freqs_sub_region.size = (num_freqs + 1) * 4;

  assert((0x7 & gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN)) == 0);
  assert((freqs_sub_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem freqs_buffer = clCreateSubBuffer(input, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                          &freqs_sub_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  gpu_ctx->EnqueueOpenCLKernel<1>(
    // Queue to run on
    queue,

    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",

    &M, &build_table_local_work_size,
    
    // No events
    0, NULL, NULL,
    
    freqs_buffer, table);

  CHECK_CL(clReleaseMemObject, freqs_buffer);

  // Load all of the offsets to the different data streams...
  cl_uint num_offsets = info.num_offsets;

  // Make sure we have enough space to use constant buffer...
  assert( static_cast<cl_ulong>(M * sizeof(AnsTableEntry)) <
          gpu_ctx->GetDeviceInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) );

  CLSubRegion data_sub_region;
  data_sub_region.origin = offset + (num_freqs + 2) * 4;
  data_sub_region.origin = ((data_sub_region.origin + 511) / 512) * 512;
  data_sub_region.size = (offset + info.sz) - data_sub_region.origin;

  assert((data_sub_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem data_buf = clCreateSubBuffer(input, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                      &data_sub_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Allocate 256 * num interleaved slots for result
  size_t total_encoded =
    num_offsets * ans::ocl::kThreadsPerEncodingGroup * ans::ocl::kNumEncodedSymbols;
  const size_t total_streams = total_encoded / ans::ocl::kNumEncodedSymbols;
  const size_t streams_per_work_group = ans::ocl::kThreadsPerEncodingGroup;

  CLKernelResult result;
  result.num_events = 1;
  result.output = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE,
                                 total_encoded, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  gpu_ctx->EnqueueOpenCLKernel<1>(
    // Queue to run on
    queue,

    // Kernel to run...
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode",

    // Work size (global and local)
    &total_streams, &streams_per_work_group,

    // Events to depend on and return
    0, NULL, result.output_events,

    // Kernel arguments
    table, data_buf, result.output);

  CHECK_CL(clReleaseMemObject, table);
  CHECK_CL(clReleaseMemObject, data_buf);

  return result;
}

static CLKernelResult InverseWavelet(const std::unique_ptr<GPUContext> &gpu_ctx,
                                     cl_command_queue queue, cl_mem img, cl_int offset,
                                     int width, int height) {
  assert(width % kWaveletBlockDim == 0);
  assert(height % kWaveletBlockDim == 0);

  cl_int errCreateBuffer;
  cl_context ctx = gpu_ctx->GetOpenCLContext();
  size_t out_sz = width * height;
  size_t local_mem_sz = 8 * kWaveletBlockDim * kWaveletBlockDim;

 CLKernelResult result;
  result.num_events = 1;
  result.output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_sz, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

#ifndef NDEBUG
  // One thread per pixel, kWaveletBlockDim * kWaveletBlockDim threads
  // per group...
  size_t threads_per_group = (kWaveletBlockDim / 2) * (kWaveletBlockDim / 2);

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
    static_cast<size_t>(height / 2) };

  size_t local_work_size[2] = {
    static_cast<size_t>(kWaveletBlockDim / 2),
    static_cast<size_t>(kWaveletBlockDim / 2) };

  gpu::GPUContext::LocalMemoryKernelArg local_mem;
  local_mem._local_mem_sz = local_mem_sz;

  gpu_ctx->EnqueueOpenCLKernel<2>(
    // Queue to run on
    queue,

    // Kernel to run...
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_InverseWavelet], "inv_wavelet",

    // Work size (global and local)
    global_work_size, local_work_size,

    // Events to depend on and return
    0, NULL, result.output_events,

    // Kernel arguments
    img, offset, local_mem, result.output);

  // No longer need image buffer
  CHECK_CL(clReleaseMemObject, img);

  return result;
}

static CLKernelResult DecompressEndpoints(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          cl_command_queue command_queue,
                                          cl_mem cmp_data, const GenTCHeader::rANSInfo &rANS_info,
                                          size_t *data_offset, cl_int val_offset,
                                          int width, int height) {
  size_t offset = *data_offset;
  *data_offset += rANS_info.sz;
  CLKernelResult decoded_ans = DecodeANS(gpu_ctx, command_queue, cmp_data, offset, rANS_info);
  CLKernelResult result = InverseWavelet(gpu_ctx, command_queue, decoded_ans.output, val_offset, width, height);
  for (cl_uint i = 0; i < decoded_ans.num_events; ++i) {
    CHECK_CL(clReleaseEvent, decoded_ans.output_events[i]);
  }
  return result;
}

static cl_event CollectEndpoints(const std::unique_ptr<GPUContext> &gpu_ctx,
                                 cl_command_queue command_queue,
                                 cl_mem dst, size_t num_pixels, const CLKernelResult &y,
                                 const CLKernelResult &co, const CLKernelResult &cg,
                                 const cl_uint endpoint_index) {
  assert(y.num_events == 1);
  assert(co.num_events == 1);
  assert(cg.num_events == 1);

  cl_event wait_events[] = {
    y.output_events[0], co.output_events[0], cg.output_events[0]
  };
  const size_t num_wait_events = sizeof(wait_events) / sizeof(wait_events[0]);

  cl_event e;
  gpu_ctx->EnqueueOpenCLKernel<1>(
    // Queue to run on
    command_queue,

    // Kernel to run...
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_Endpoints], "collect_endpoints",

    // Work size (global and local)
    &num_pixels, NULL,

    // Events to depend on and return
    num_wait_events, wait_events, &e,

    // Kernel arguments
    y.output, co.output, cg.output, dst, endpoint_index);

  // Release events we don't need anymore
  for (size_t i = 0; i < num_wait_events; ++i) {
    CHECK_CL(clReleaseEvent, wait_events[i]);
  }
  CHECK_CL(clReleaseMemObject, y.output);
  CHECK_CL(clReleaseMemObject, co.output);
  CHECK_CL(clReleaseMemObject, cg.output);

  return e;
}

static cl_event DecodeIndices(const std::unique_ptr<GPUContext> &gpu_ctx, cl_command_queue queue,
                              cl_mem dst, cl_mem cmp_buf, size_t offset, size_t num_pixels,
                              const GenTCHeader::rANSInfo &palette_info,
                              const GenTCHeader::rANSInfo &indices_info) {

  static std::atomic_int queue_order_counter(0);
  int queue_order = (queue_order_counter++) % 2;

  CLKernelResult palette = DecodeANS(gpu_ctx, queue, cmp_buf, offset, palette_info);
  CLKernelResult indices = DecodeANS(gpu_ctx, queue, cmp_buf, offset + palette_info.sz, indices_info);

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

    cl_event next_event;
    cl_uint num_events = static_cast<cl_uint>(e.size()) - event_idx;
    size_t local_work_sz = std::min(num_vals, kLocalScanSz);

#ifndef NDEBUG
    assert((num_vals % local_work_sz) == 0);
    assert(local_work_sz <= gpu_ctx->GetKernelWGInfo<size_t>(
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",
      CL_KERNEL_WORK_GROUP_SIZE));
#endif

    gpu_ctx->EnqueueOpenCLKernel<1>(
      // Queue to run on
      queue,

      // Kernel to run...
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",

      // Work size (global and local)
      &num_vals, &local_work_sz,

      // Events to depend on and return
      num_events, e.data() + event_idx, &next_event,

      // Kernel arguments
      indices.output, stage, dst);

    num_vals >>= kLocalScanSzLog;
    event_idx += num_events;
    e.push_back(next_event);
  }

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

    gpu_ctx->EnqueueOpenCLKernel<1>(
      // Queue to run on
      queue,

      // Kernel to run...
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "collect_indices",

      // Work size (global and local)
      &num_vals, &kLocalScanSz,

      // Events to depend on and return
      num_events, e.data() + event_idx, &next_event,

      // Kernel arguments
      palette.output, stage, dst);

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
                               const GenTCHeader &hdr, cl_command_queue queue, cl_mem cmp_buf,
                               cl_event init_event, CLKernelResult *result) {
  assert((hdr.width & 0x3) == 0);
  assert((hdr.height & 0x3) == 0);
  const int blocks_x = static_cast<int>(hdr.width) >> 2;
  const int blocks_y = static_cast<int>(hdr.height) >> 2;

  size_t offset = 0;
  CLKernelResult ep1_y = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep1_y,
                                             &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep1_co = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep1_co,
                                              &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep1_cg = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep1_cg,
                                              &offset, -128, blocks_x, blocks_y);

  CLKernelResult ep2_y = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep2_y,
                                             &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep2_co = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep2_co,
                                              &offset, -128, blocks_x, blocks_y);
  CLKernelResult ep2_cg = DecompressEndpoints(gpu_ctx, queue, cmp_buf, hdr.ep2_cg,
                                              &offset, -128, blocks_x, blocks_y);

  static std::atomic_int queue_order_counter(0);
  int queue_order = (queue_order_counter++) % 3;

  result->num_events = 3;
  const size_t num_blocks = blocks_x * blocks_y;
  result->output_events[0] =
    CollectEndpoints(gpu_ctx, queue, result->output, num_blocks, ep1_y, ep1_co, ep1_cg, 0);
  result->output_events[1] =
    CollectEndpoints(gpu_ctx, queue, result->output, num_blocks, ep2_y, ep2_co, ep2_cg, 1);

  result->output_events[2] = DecodeIndices(gpu_ctx, queue, result->output, cmp_buf,
                                           offset, num_blocks, hdr.palette, hdr.indices);
}

cl_mem UploadData(const std::unique_ptr<GPUContext> &gpu_ctx,
                  const std::vector<uint8_t> &cmp_data, GenTCHeader *hdr) {
  hdr->LoadFrom(cmp_data.data());

  // Upload everything but the header
  cl_int errCreateBuffer;
  static const size_t kHeaderSz = sizeof(*hdr);
  cl_mem cmp_buf = clCreateBuffer(gpu_ctx->GetOpenCLContext(), GetHostReadOnlyFlags(),
                                  cmp_data.size() - kHeaderSz, const_cast<uint8_t *>(cmp_data.data() + kHeaderSz),
                                  &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  return cmp_buf;
}
  
std::vector<uint8_t>  DecompressDXTBuffer(const std::unique_ptr<GPUContext> &gpu_ctx,
                                          const std::vector<uint8_t> &cmp_data) {
  cl_command_queue queue = gpu_ctx->GetNextQueue();

  GenTCHeader hdr;
  cl_mem cmp_buf = UploadData(gpu_ctx, cmp_data, &hdr);

  // Setup output
#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_READ_WRITE;
#endif
  CLKernelResult decmp;

  cl_int errCreateBuffer;
  size_t dxt_size = (hdr.width * hdr.height) / 2;
  decmp.output = clCreateBuffer(gpu_ctx->GetOpenCLContext(),
                                out_flags, dxt_size, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Set a dummy event
  cl_event init_event;
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueMarkerWithWaitList, queue, 0, NULL, &init_event);
#else
  CHECK_CL(clEnqueueMarker, queue, &init_event);
#endif

  // Queue the decompression...
  DecompressDXTImage(gpu_ctx, hdr, queue, cmp_buf, init_event, &decmp);

  // Block on read
  std::vector<uint8_t> decmp_data(dxt_size, 0xFF);
  CHECK_CL(clEnqueueReadBuffer, queue, decmp.output, CL_TRUE, 0, dxt_size, decmp_data.data(),
                                decmp.num_events, decmp.output_events, NULL);

  CHECK_CL(clReleaseMemObject, cmp_buf);
  for (size_t i = 0; i < decmp.num_events; ++i) {
    CHECK_CL(clReleaseEvent, decmp.output_events[i]);
  }
  CHECK_CL(clReleaseMemObject, decmp.output);
  CHECK_CL(clReleaseEvent, init_event);
  return std::move(decmp_data);
}

DXTImage DecompressDXT(const std::unique_ptr<GPUContext> &gpu_ctx,
                       const std::vector<uint8_t> &cmp_data) {
  GenTCHeader hdr;
  hdr.LoadFrom(cmp_data.data());

  std::vector<uint8_t> decmp_data = std::move(DecompressDXTBuffer(gpu_ctx, cmp_data));
  return DXTImage(hdr.width, hdr.height, decmp_data);
}

bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
             const char *filename, const char *cmp_fn) {
  DXTImage dxt_img(filename, cmp_fn);
  
  std::vector<uint8_t> cmp_img = std::move(CompressDXTImage(dxt_img));
  std::vector<uint8_t> decmp_data = std::move(DecompressDXTBuffer(gpu_ctx, cmp_img));

  const std::vector<PhysicalDXTBlock> &blks = dxt_img.PhysicalBlocks();
  for (size_t i = 0; i < blks.size(); ++i) {
    const PhysicalDXTBlock *blk =
      reinterpret_cast<const PhysicalDXTBlock *>(decmp_data.data() + i * 8);
    if (memcmp(&blks[i], blk, 8) != 0) {
      std::cout << "Bad block: " << i << std::endl;
      printf("Original block: 0x%lx\n", static_cast<unsigned long>(blks[i].dxt_block));
      printf("Compressed block: 0x%lx\n", static_cast<unsigned long>(blk->dxt_block));
      return false;
    }
  }
  
  return true;
}

std::vector<cl_event> LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                                        const GenTCHeader &hdr, cl_command_queue queue,
                                        cl_mem cmp_data, cl_mem output,
                                        cl_event *init) {
  // Set a dummy event
  cl_event init_event;
  if (NULL == init) {
#ifdef CL_VERSION_1_2
    CHECK_CL(clEnqueueMarkerWithWaitList, queue, 0, NULL, &init_event);
#else
    CHECK_CL(clEnqueueMarker, queue, &init_event);
#endif
  } else {
    // If there's an actual event, switch to it instead
    init_event = *init;
  }

  // Queue the decompression...
  CLKernelResult decmp;
  decmp.output = output;
  DecompressDXTImage(gpu_ctx, hdr, queue, cmp_data, init_event, &decmp);
  if (NULL == init) {
    CHECK_CL(clReleaseEvent, init_event);
  }

  // Send back the events...
  std::vector<cl_event> events;
  events.reserve(decmp.num_events);
  for (size_t i = 0; i < decmp.num_events; ++i) {
    events.push_back(decmp.output_events[i]);
  }

  return std::move(events);
}

bool InitializeDecoder(const std::unique_ptr<gpu::GPUContext> &gpu_ctx) {
  bool ok = true;

  // At least make sure that the work group size needed for each kernel is met...
  ok = ok && 256 <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && ans::ocl::kThreadsPerEncodingGroup <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && (kWaveletBlockDim * kWaveletBlockDim / 4) <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_InverseWavelet], "inv_wavelet",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && 1 <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_Endpoints], "collect_endpoints",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && 128 <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && 128 <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "collect_indices",
    CL_KERNEL_WORK_GROUP_SIZE);

  return ok;
}

void GenTCHeader::Print() const {
  std::cout << "Width: " << width << std::endl;
  std::cout << "Height: " << height << std::endl;
  std::cout << "Endpoint One Y compressed size: " << ep1_y.sz << std::endl;
  std::cout << "Endpoint One Co compressed size: " << ep1_co.sz << std::endl;
  std::cout << "Endpoint One Cg compressed size: " << ep1_cg.sz << std::endl;
  std::cout << "Endpoint Two Y compressed size: " << ep2_y.sz << std::endl;
  std::cout << "Endpoint Two Co compressed size: " << ep2_co.sz << std::endl;
  std::cout << "Endpoint Two Cg compressed size: " << ep2_cg.sz << std::endl;
  std::cout << "Palette size compressed: " << palette.sz << std::endl;
  std::cout << "Palette index deltas compressed: " << indices.sz << std::endl;
}

void GenTCHeader::LoadFrom(const uint8_t *buf) {
  // Read the header
  memcpy(this, buf, sizeof(*this));
#ifndef NDEBUG
  Print();
#endif
}

}
