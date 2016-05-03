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
    ->Chain(ReducePrecision<WaveletUnsignedTy, uint8_t>::New());

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

  std::cout << "Compressing Y plane for EP 1... ";
  auto ep1_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Compressing Co plane for EP 1... ";
  auto ep1_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Compressing Cg plane for EP 1... ";
  auto ep1_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep1_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Compressing Y plane for EP 2... ";
  auto ep2_y_cmp = RunDXTEndpointPipeline(std::get<0>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Compressing Co plane for EP 2... ";
  auto ep2_co_cmp = RunDXTEndpointPipeline(std::get<1>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  std::cout << "Compressing Cg plane for EP 2... ";
  auto ep2_cg_cmp = RunDXTEndpointPipeline(std::get<2>(*ep2_planes));
  std::cout << "Done. " << std::endl;

  auto cmp_pipeline =
    Pipeline<std::vector<uint8_t>, std::vector<uint8_t> >
    ::Create(ByteEncoder::Encoder(ans::ocl::kNumEncodedSymbols));

  // Concatenate Y planes
  ep1_y_cmp->insert(ep1_y_cmp->end(), ep2_y_cmp->begin(), ep2_y_cmp->end());
  auto y_planes = cmp_pipeline->Run(ep1_y_cmp);

  // Concatenate Chroma planes
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep1_cg_cmp->begin(), ep1_cg_cmp->end());
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep2_co_cmp->begin(), ep2_co_cmp->end());
  ep1_co_cmp->insert(ep1_co_cmp->end(), ep2_cg_cmp->begin(), ep2_cg_cmp->end());
  auto chroma_planes = cmp_pipeline->Run(ep1_co_cmp);

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

static std::vector<cl_event> DecompressDXTImage(const std::unique_ptr<GPUContext> &gpu_ctx,
                                                const std::vector<GenTCHeader> &hdrs, cl_command_queue queue,
                                                cl_mem cmp_data, cl_event init_event, cl_mem output) {
  // Queue the decompression...
  cl_int errCreateBuffer;

  size_t blocks_x = hdrs[0].width / 4;
  size_t blocks_y = hdrs[0].height / 4;
  size_t num_vals = blocks_x * blocks_y;

  size_t scratch_mem_sz = 0;
  for (const auto &hdr : hdrs) {
    // If the images don't match in each dimension, then our inverse wavelet calculation
    // doesn't do a good job. =(
    assert(hdr.width / 4 == blocks_x);
    assert(hdr.height / 4 == blocks_y);

    scratch_mem_sz += 4 * ans::ocl::kANSTableSize * sizeof(AnsTableEntry);
    scratch_mem_sz += 13 * num_vals;
    scratch_mem_sz += hdr.palette_bytes;
  }
  cl_mem scratch = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE, scratch_mem_sz, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  assert((0x7 & gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN)) == 0);

  // First get the number of frequencies...
  const size_t M = ans::ocl::kANSTableSize;
  const size_t build_table_global_work_size[2] = { M, 4 * hdrs.size() };
  const size_t build_table_local_work_size[2] = { 256, 1 };
  assert(build_table_local_work_size[0] <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",
    CL_KERNEL_WORK_GROUP_SIZE));

  cl_buffer_region freqs_sub_region;
  freqs_sub_region.origin = 0;
  freqs_sub_region.size = 4 * 512 * hdrs.size();

  assert((0x7 & gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN)) == 0);
  assert((freqs_sub_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem freqs_buffer = clCreateSubBuffer(cmp_data, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                          &freqs_sub_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_buffer_region table_sub_region;
  table_sub_region.origin = 0;
  table_sub_region.size = hdrs.size() * 4 * ans::ocl::kANSTableSize * sizeof(AnsTableEntry);
  assert((table_sub_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);
  cl_mem table_region = clCreateSubBuffer(scratch, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                          &table_sub_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_event build_table_event;
  gpu_ctx->EnqueueOpenCLKernel<2>(
    // Queue to run on
    queue,

    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",

    build_table_global_work_size, build_table_local_work_size,

    // Events
    1, &init_event, &build_table_event,

    freqs_buffer, table_region);
  CHECK_CL(clReleaseMemObject, freqs_buffer);

  // Setup ANS input
  std::vector<cl_uint> input_offsets(hdrs.size() * 4);

  cl_uint input_offset = 0;
  for (size_t i = 0; i < hdrs.size(); ++i) {
    input_offsets[4 * i + 0] = input_offset; input_offset += hdrs[i].y_cmp_sz;
    input_offsets[4 * i + 1] = input_offset; input_offset += hdrs[i].chroma_cmp_sz;
    input_offsets[4 * i + 2] = input_offset; input_offset += hdrs[i].palette_sz;
    input_offsets[4 * i + 3] = input_offset; input_offset += hdrs[i].indices_sz;
  }

  cl_buffer_region input_region;
  input_region.origin = 4 * 512 * hdrs.size();
  input_region.size = input_offset;
  assert((input_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem input_buf = clCreateSubBuffer(cmp_data, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                       &input_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Setup ANS output
  std::vector<cl_uint> output_offsets (4 * hdrs.size());

  cl_uint output_offset = 0;
  for (size_t i = 0; i < hdrs.size(); ++i) {
    cl_uint nvals = static_cast<cl_uint>(num_vals);

    output_offsets[4 * i + 0] = output_offset;
    output_offset += 2 * nvals; // Y planes
    
    output_offsets[4 * i + 1] = output_offset;
    output_offset += 4 * nvals; // Chroma planes
    
    output_offsets[4 * i + 2] = output_offset;
    output_offset += static_cast<cl_uint>(hdrs[i].palette_bytes); // Palette
    
    output_offsets[4 * i + 3] = output_offset;
    output_offset += nvals; // Indices
  }
  assert(output_offset % ans::ocl::kNumEncodedSymbols == 0);

  cl_buffer_region decmp_region;
  decmp_region.origin = 4 * ans::ocl::kANSTableSize * sizeof(AnsTableEntry) * hdrs.size();
  decmp_region.size = output_offset;
  assert((decmp_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem decmp_buf = clCreateSubBuffer(scratch, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                       &decmp_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Setup OpenCL buffers for input and output offsets
  cl_mem ans_input_offsets = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            input_offsets.size() * sizeof(input_offsets[0]),
                                            input_offsets.data(), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem ans_output_offsets = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             output_offsets.size() * sizeof(output_offsets[0]),
                                             output_offsets.data(), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Allocate 256 * num interleaved slots for result
  const size_t rANS_global_work = output_offset / ans::ocl::kNumEncodedSymbols;
  const size_t rANS_local_work = ans::ocl::kThreadsPerEncodingGroup;
  assert(rANS_global_work % rANS_local_work == 0);

  cl_uint num_offsets = static_cast<cl_uint>(input_offsets.size());
  assert(num_offsets == output_offsets.size());

  cl_event decode_ans_event;
  gpu_ctx->EnqueueOpenCLKernel<1>(
    // Queue to run on
    queue,

    // Kernel to run...
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode_multiple",

    // Work size (global and local)
    &rANS_global_work, &rANS_local_work,

    // Events to depend on and return
    1, &build_table_event, &decode_ans_event,

    // Kernel arguments
    table_region, num_offsets, ans_input_offsets, ans_output_offsets, input_buf, decmp_buf);
  CHECK_CL(clReleaseEvent, build_table_event);
  CHECK_CL(clReleaseMemObject, table_region);
  CHECK_CL(clReleaseMemObject, input_buf);

  // Run inverse wavelet
  assert(blocks_x % kWaveletBlockDim == 0);
  assert(blocks_y % kWaveletBlockDim == 0);
  size_t local_mem_sz = 8 * kWaveletBlockDim * kWaveletBlockDim;

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
  assert(threads_per_group <= wgsz.sizes[0]);
#endif

  size_t inv_wavelet_global_work_size[3] = {
    static_cast<size_t>(blocks_x / 2),
    static_cast<size_t>(blocks_y / 2),
    6 * hdrs.size()
  };

  size_t inv_wavelet_local_work_size[3] = {
    static_cast<size_t>(kWaveletBlockDim / 2),
    static_cast<size_t>(kWaveletBlockDim / 2),
    1
  };

  cl_buffer_region inv_wavelet_output_region;
  inv_wavelet_output_region.origin =
    4 * hdrs.size() * sizeof(AnsTableEntry) * ans::ocl::kANSTableSize + output_offset;
  inv_wavelet_output_region.size = 6 * num_vals * hdrs.size();
  assert((inv_wavelet_output_region.origin % (gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8)) == 0);

  cl_mem inv_wavelet_output = clCreateSubBuffer(scratch, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                                &inv_wavelet_output_region, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  gpu::GPUContext::LocalMemoryKernelArg local_mem;
  local_mem._local_mem_sz = local_mem_sz;

  cl_event inv_wavelet_event;
  gpu_ctx->EnqueueOpenCLKernel<3>(
    // Queue to run on
    queue,

    // Kernel to run...
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_InverseWavelet], "inv_wavelet",

    // Work size (global and local)
    inv_wavelet_global_work_size, inv_wavelet_local_work_size,

    // Events to depend on and return
    1, &decode_ans_event, &inv_wavelet_event,

    // Kernel arguments
    decmp_buf, ans_output_offsets, local_mem, inv_wavelet_output);

  static const size_t kLocalScanSz = 128;
  static const size_t kLocalScanSzLog = 7;

  // !SPEED! We don't really need to allocate here...
  cl_event decode_event = decode_ans_event;

  size_t num_decode_indices_vals = num_vals;
  cl_uint total_num_indices = static_cast<cl_uint>(num_vals);

  cl_int stage = -1;
  while (num_decode_indices_vals > 1) {
    stage++;

    cl_event next_event;
    size_t decode_indices_global_work_sz[2] = {
      num_decode_indices_vals,
      hdrs.size()
    };

    size_t decode_indices_local_work_sz[2] = {
      std::min(num_decode_indices_vals, kLocalScanSz),
      1
    };

#ifndef NDEBUG
    assert((num_vals % decode_indices_local_work_sz[0]) == 0);
    assert(decode_indices_local_work_sz[0] <= gpu_ctx->GetKernelWGInfo<size_t>(
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",
      CL_KERNEL_WORK_GROUP_SIZE));
#endif

    gpu_ctx->EnqueueOpenCLKernel<2>(
      // Queue to run on
      queue,

      // Kernel to run...
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "decode_indices",

      // Work size (global and local)
      decode_indices_global_work_sz, decode_indices_local_work_sz,

      // Events to depend on and return
      1, &decode_event, &next_event,

      // Kernel arguments
      decmp_buf, ans_output_offsets, stage, total_num_indices, output);

    CHECK_CL(clReleaseEvent, decode_event);
    decode_event = next_event;

    num_decode_indices_vals >>= kLocalScanSzLog;
  }

  num_decode_indices_vals = num_vals;
  while ((num_decode_indices_vals >> kLocalScanSzLog) > 1) {
    num_decode_indices_vals >>= kLocalScanSzLog;
  }

  while (stage > 0) {
    num_decode_indices_vals = std::min<size_t>(num_vals, num_decode_indices_vals << kLocalScanSzLog);

    size_t collect_indices_global_work_sz[2] = {
      num_decode_indices_vals,
      hdrs.size()
    };

    size_t collect_indices_local_work_sz[2] = {
      kLocalScanSz,
      1
    };

    cl_event next_event;
    gpu_ctx->EnqueueOpenCLKernel<2>(
      // Queue to run on
      queue,

      // Kernel to run...
      GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_DecodeIndices], "collect_indices",

      // Work size (global and local)
      collect_indices_global_work_sz, collect_indices_local_work_sz,

      // Events to depend on and return
      1, &decode_event, &next_event,

      // Kernel arguments
      stage, total_num_indices, output);

    CHECK_CL(clReleaseEvent, decode_event);
    decode_event = next_event;

    stage--;
  }

  size_t collect_endpoints_global_work_size[3] = {
    static_cast<size_t>(blocks_x * blocks_y),
    1,
    hdrs.size(), // Number of textures
  };

  cl_event assembly_events[2] = { inv_wavelet_event, decode_event };
  cl_event assembly_event;
  gpu_ctx->EnqueueOpenCLKernel<3>(
    // Queue to run on
    queue,

    // Kernel to run...
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_Assemble], "assemble_dxt",

    // Work size (global and local)
    collect_endpoints_global_work_size, NULL,

    // Events to depend on and return
    2, assembly_events, &assembly_event,

    // Kernel arguments
    decmp_buf, ans_output_offsets, inv_wavelet_output, output);
  CHECK_CL(clReleaseEvent, decode_event);
  CHECK_CL(clReleaseEvent, inv_wavelet_event);
  CHECK_CL(clReleaseMemObject, inv_wavelet_output);
  CHECK_CL(clReleaseMemObject, decmp_buf);
  CHECK_CL(clReleaseMemObject, ans_input_offsets);
  CHECK_CL(clReleaseMemObject, ans_output_offsets);
  CHECK_CL(clReleaseMemObject, scratch);

  // Send back the events...
  return { assembly_event };
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
  cl_int errCreateBuffer;
  size_t dxt_size = (hdr.width * hdr.height) / 2;
  cl_mem dxt_output = clCreateBuffer(gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                                     dxt_size, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Set a dummy event
  cl_event init_event;
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueMarkerWithWaitList, queue, 0, NULL, &init_event);
#else
  CHECK_CL(clEnqueueMarker, queue, &init_event);
#endif

  // Queue the decompression...
  std::vector<cl_event> dxt_events =
    DecompressDXTImage(gpu_ctx, { hdr }, queue, cmp_buf, init_event, dxt_output);

  // Block on read
  std::vector<uint8_t> decmp_data(dxt_size, 0xFF);
  CHECK_CL(clEnqueueReadBuffer, queue, dxt_output, CL_TRUE, 0, dxt_size, decmp_data.data(),
                                static_cast<cl_uint>(dxt_events.size()), dxt_events.data(), NULL);

  CHECK_CL(clReleaseMemObject, cmp_buf);
  for (auto e : dxt_events) { CHECK_CL(clReleaseEvent, e); }
  CHECK_CL(clReleaseMemObject, dxt_output);
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
                                        cl_mem cmp_data, cl_mem output, cl_event init) {
  return std::move(DecompressDXTImage(gpu_ctx, { hdr }, queue, cmp_data, init, output));
}

std::vector<cl_event> LoadCompressedDXTs(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                                         const std::vector<GenTCHeader> &hdrs, cl_command_queue queue,
                                         cl_mem cmp_data, cl_mem output, cl_event init) {
  return std::move(DecompressDXTImage(gpu_ctx, hdrs, queue, cmp_data, init, output));
}


bool InitializeDecoder(const std::unique_ptr<gpu::GPUContext> &gpu_ctx) {
  bool ok = true;

  // At least make sure that the work group size needed for each kernel is met...
  ok = ok && 256 <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_BuildTable], "build_table",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && ans::ocl::kThreadsPerEncodingGroup <= gpu_ctx->GetKernelWGInfo<size_t>(
    ans::kANSOpenCLKernels[ans::eANSOpenCLKernel_ANSDecode], "ans_decode_multiple",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && (kWaveletBlockDim * kWaveletBlockDim / 4) <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_InverseWavelet], "inv_wavelet",
    CL_KERNEL_WORK_GROUP_SIZE);

  ok = ok && 1 <= gpu_ctx->GetKernelWGInfo<size_t>(
    GenTC::kOpenCLKernels[GenTC::eOpenCLKernel_Assemble], "assemble_dxt",
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
  std::cout << "Num Palette Entries: " << (palette_bytes / 4) << std::endl;
  std::cout << "Y compressed size: " << y_cmp_sz << std::endl;
  std::cout << "Chroma compressed size: " << chroma_cmp_sz << std::endl;
  std::cout << "Palette size compressed: " << palette_sz << std::endl;
  std::cout << "Palette index deltas compressed: " << indices_sz << std::endl;
}

void GenTCHeader::LoadFrom(const uint8_t *buf) {
  // Read the header
  memcpy(this, buf, sizeof(*this));
#ifndef NDEBUG
  Print();
#endif
}

}
