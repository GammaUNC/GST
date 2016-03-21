#include "ans_ocl.h"

#include <numeric>
#include <iostream>

#include "ans_config.h"
#include "kernel_cache.h"
#include "histogram.h"

struct AnsTableEntry {
  cl_ushort freq;
  cl_ushort cum_freq;
  cl_uchar  symbol;
};

template<typename T>
static std::vector<T> ReadBuffer(cl_command_queue queue, cl_mem buffer, size_t num_elements) {
  std::vector<T> host_mem(num_elements);
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueBarrierWithWaitList, queue, 0, NULL, NULL);
#else
  CHECK_CL(clEnqueueBarrier, queue);
#endif
  CHECK_CL(clEnqueueReadBuffer, queue, buffer, true, 0, num_elements * sizeof(T), host_mem.data(), 0, NULL, NULL);
  return std::move(host_mem);
}

namespace ans {
namespace ocl{

std::vector<uint32_t> NormalizeFrequencies(const std::vector<uint32_t> &F) {
  return std::move(ans::GenerateHistogram(F, kANSTableSize));
}

ans::Options GetOpenCLOptions(const std::vector<uint32_t> &F) {
  Options opts;
  opts.b = 1 << 16;
  opts.k = 1 << 4;
  opts.M = kANSTableSize;
  opts.Fs = F;
  opts.type = eType_rANS;
  return opts;
}

std::unique_ptr<Encoder> CreateCPUEncoder(const std::vector<uint32_t> &F) {
  return Encoder::Create(GetOpenCLOptions(F));
}

// A CPU decoder that matches the OpenCLDecoder below.
std::unique_ptr<Decoder> CreateCPUDecoder(uint32_t state, const std::vector<uint32_t> &F) {
  return Decoder::Create(state, GetOpenCLOptions(F));
}

OpenCLDecoder::OpenCLDecoder(
  const std::unique_ptr<gpu::GPUContext> &ctx, const std::vector<uint32_t> &F, const int num_interleaved)
  : _num_interleaved(num_interleaved)
  , _M(kANSTableSize)
  , _gpu_ctx(ctx)
{
  cl_int errCreateBuffer;
  _table = clCreateBuffer(_gpu_ctx->GetOpenCLContext(), CL_MEM_READ_WRITE, _M * sizeof(AnsTableEntry), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  RebuildTable(F);
}

OpenCLDecoder::~OpenCLDecoder() {
  CHECK_CL(clReleaseMemObject, _table);
}

std::vector<cl_uchar> OpenCLDecoder::GetSymbols() const {
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(_gpu_ctx->GetCommandQueue(), _table, _M));

  std::vector<cl_uchar> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.symbol);
  }

  return std::move(result);
}

std::vector<cl_ushort> OpenCLDecoder::GetFrequencies() const {
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(_gpu_ctx->GetCommandQueue(), _table, _M));

  std::vector<cl_ushort> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.freq);
  }

  return std::move(result);
}

std::vector<cl_ushort> OpenCLDecoder::GetCumulativeFrequencies() const {
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(_gpu_ctx->GetCommandQueue(), _table, _M));

  std::vector<cl_ushort> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.cum_freq);
  }

  return std::move(result);
}

void OpenCLDecoder::RebuildTable(const std::vector<uint32_t> &F) const {
  std::vector<cl_uint> freqs = std::move(NormalizeFrequencies(F));
  assert(_M == std::accumulate(freqs.begin(), freqs.end(), 0U));

  cl_kernel build_table_kernel = _gpu_ctx->GetOpenCLKernel(
    kANSOpenCLKernels[eANSOpenCLKernel_BuildTable], "build_table");

#ifndef NDEBUG
  size_t work_group_size;
  CHECK_CL(clGetKernelWorkGroupInfo, build_table_kernel, _gpu_ctx->GetDeviceID(), CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t), &work_group_size, NULL);
  assert(work_group_size >= 32);
#endif

  cl_mem_flags flags = GetHostReadOnlyFlags();
  cl_uint num_freqs = static_cast<cl_uint>(freqs.size());

  // Note: we could do this on the GPU as well, but the array size here is almost never more than
  // about 256, so the CPU is actually much better at doing it. We can also stick it in constant
  // memory, which makes the upload not that bad...
  std::vector<cl_uint> cum_freqs(num_freqs, 0);
  std::partial_sum(freqs.begin(), freqs.end() - 1, cum_freqs.begin() + 1);

  cl_uint *freqs_ptr = const_cast<cl_uint *>(freqs.data());
  cl_uint *cum_freqs_ptr = cum_freqs.data();

  cl_int errCreateBuffer;
  cl_mem freqs_buffer = clCreateBuffer(_gpu_ctx->GetOpenCLContext(), flags, freqs.size() * sizeof(freqs_ptr[0]), freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  const size_t cum_freqs_buf_size = cum_freqs.size() * sizeof(cum_freqs_ptr[0]);
  cl_mem cum_freqs_buffer = clCreateBuffer(_gpu_ctx->GetOpenCLContext(), flags, cum_freqs_buf_size, cum_freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel, 0, sizeof(freqs_buffer), &freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 1, sizeof(cum_freqs_buffer), &cum_freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel, 2, sizeof(cl_uint), &num_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel, 3, sizeof(_table), &_table);

  CHECK_CL(clEnqueueNDRangeKernel, _gpu_ctx->GetCommandQueue(), build_table_kernel, 1, NULL, &_M, NULL, 0, NULL, NULL);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);

  // !SPEED! Enqueue barrier with no wait list... Actually we can do better
  // here, since we know that the kernel invocation will need to finish, we
  // can pass an event to the barrier when the CL version is greater than 1.2...
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueBarrierWithWaitList, _gpu_ctx->GetCommandQueue(), 0, NULL, NULL);
#else
  CHECK_CL(clEnqueueBarrier, queue);
#endif
}

std::vector<cl_uchar> OpenCLDecoder::Decode(
  cl_uint state,
  const std::vector<cl_uchar> &data) const {

  cl_int errCreateBuffer;
  cl_kernel decode_kernel = _gpu_ctx->GetOpenCLKernel(
    kANSOpenCLKernels[eANSOpenCLKernel_ANSDecode], "ans_decode");
  cl_context ctx = _gpu_ctx->GetOpenCLContext();

  // First, just set our table buffers...
  CHECK_CL(clSetKernelArg, decode_kernel, 0, sizeof(_table), &_table);

  // Create a data pointer
  uint32_t offset = data.size() + 8;
  std::vector<cl_uchar> ocl_data(offset, 0);
  memcpy(ocl_data.data(), &offset, sizeof(offset));
  memcpy(ocl_data.data() + 4, data.data(), data.size());
  memcpy(ocl_data.data() + ocl_data.size() - 4, &state, sizeof(state));

  // Data: just send the data pointer...
  cl_mem data_buffer = clCreateBuffer(ctx, GetHostReadOnlyFlags(), ocl_data.size(),
                                      ocl_data.data(), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(data_buffer), &data_buffer);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif

  // Allocate 256 slots for result
  cl_mem out_buffer = clCreateBuffer(ctx, out_flags, kNumEncodedSymbols, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(out_buffer), &out_buffer);

  // Run the kernel...
  const size_t num_streams = 1;
  CHECK_CL(clEnqueueNDRangeKernel, _gpu_ctx->GetCommandQueue(), decode_kernel, 1, NULL,
                                   &num_streams, NULL, 0, NULL, NULL);

  std::vector<cl_uchar> out = std::move(
    ReadBuffer<cl_uchar>(_gpu_ctx->GetCommandQueue(), out_buffer, kNumEncodedSymbols));

  // Release buffer objects...
  CHECK_CL(clReleaseMemObject, data_buffer);
  CHECK_CL(clReleaseMemObject, out_buffer);

  return std::move(out);
}

std::vector<std::vector<cl_uchar> > OpenCLDecoder::Decode(
  const std::vector<cl_uint> &states,
  const std::vector<cl_uchar> &data) const {

  cl_int errCreateBuffer;
  cl_kernel decode_kernel = _gpu_ctx->GetOpenCLKernel(
    kANSOpenCLKernels[eANSOpenCLKernel_ANSDecode], "ans_decode");
  cl_context ctx = _gpu_ctx->GetOpenCLContext();

  // First, just set our table buffers...
  CHECK_CL(clSetKernelArg, decode_kernel, 0, sizeof(_table), &_table);

  // Create a data pointer
  uint32_t offset = data.size() + 4 + states.size() * 4;
  std::vector<cl_uchar> ocl_data(offset, 0);
  memcpy(ocl_data.data(), &offset, sizeof(offset));
  memcpy(ocl_data.data() + 4, data.data(), data.size());
  memcpy(ocl_data.data() + 4 + data.size(), states.data(), 4 * states.size());

  // Data: just send the data pointer...
  cl_mem data_buf = clCreateBuffer(ctx, GetHostReadOnlyFlags(),
                                   ocl_data.size(), ocl_data.data(), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(data_buf), &data_buf);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif

  // Allocate 256 * num interleaved slots for result
  size_t total_encoded = states.size() * kNumEncodedSymbols;
  cl_mem out_buf = clCreateBuffer(ctx, out_flags, total_encoded, NULL,
                                  &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(out_buf), &out_buf);

  // Run the kernel...
  const size_t num_streams = states.size();
  assert(num_streams <= static_cast<size_t>(_num_interleaved));
  const size_t streams_per_work_group = num_streams;

#ifndef NDEBUG
  // Make sure that we can launch enough kernels and that we have sufficient
  // space on the device.
  assert(streams_per_work_group <
    _gpu_ctx->GetDeviceInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE));

  // I don't know of a GPU implementation that uses more than 3 dims..
  assert(3 ==
    _gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));

  struct WorkGroupSizes {
    size_t sizes[3];
  };
  WorkGroupSizes wgsz =
    _gpu_ctx->GetDeviceInfo<WorkGroupSizes>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  assert(streams_per_work_group < wgsz.sizes[0] );

  size_t total_constant_memory = 0;
  total_constant_memory += ocl_data.size();
  total_constant_memory += _M * sizeof(AnsTableEntry);

  assert(total_constant_memory <
    _gpu_ctx->GetDeviceInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE));
  assert(4 < _gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS));
#endif

  CHECK_CL(clEnqueueNDRangeKernel, _gpu_ctx->GetCommandQueue(), decode_kernel,
                                   1, NULL, &num_streams,
                                   &streams_per_work_group, 0, NULL, NULL);

  // Read back the buffer...
  std::vector<cl_uchar> all_symbols = std::move(
    ReadBuffer<cl_uchar>(_gpu_ctx->GetCommandQueue(), out_buf, total_encoded));

  std::vector<std::vector<cl_uchar> > out;
  out.reserve(num_streams);
  for (size_t i = 0; i < num_streams; ++i) {
    std::vector<cl_uchar> stream;
    size_t stream_start = i * kNumEncodedSymbols;
    size_t stream_end = stream_start + kNumEncodedSymbols;
    stream.insert(stream.begin(), all_symbols.begin() + stream_start, all_symbols.begin() + stream_end);
    out.push_back(std::move(stream));
  }

  // Release buffer objects...
  CHECK_CL(clReleaseMemObject, data_buf);
  CHECK_CL(clReleaseMemObject, out_buf);

  return std::move(out);
}

std::vector<std::vector<cl_uchar> > OpenCLDecoder::Decode(
  const std::vector<cl_uint> &states,
  const std::vector<std::vector<cl_uchar> > &data) const {

  const size_t total_streams = states.size();
  const size_t streams_per_work_group = _num_interleaved;
  const size_t total_work_groups = total_streams / streams_per_work_group;
  assert(total_work_groups * streams_per_work_group == total_streams); // Evenly distribute...
  assert(total_work_groups == data.size());

  cl_int errCreateBuffer;
  cl_kernel decode_kernel = _gpu_ctx->GetOpenCLKernel(
    kANSOpenCLKernels[eANSOpenCLKernel_ANSDecode], "ans_decode");
  cl_context ctx = _gpu_ctx->GetOpenCLContext();

  // First, just set our table buffers...
  CHECK_CL(clSetKernelArg, decode_kernel, 0, sizeof(_table), &_table);

  // Data: Coalesce all the data into one large ptr...
  std::vector<cl_uchar> all_the_data(4 * total_work_groups, 0);

  // One offset per work group.
  uint32_t offset = total_work_groups * 4;
  for (size_t i = 0; i < total_work_groups; ++i) {
    offset += data[i].size() + 4 * streams_per_work_group;
    memcpy(all_the_data.data() + i * 4, &offset, sizeof(offset));
  }

  // Put all the data together...
  size_t states_offset = 0;
  for (const auto &strm : data) {
    size_t last = all_the_data.size();
    all_the_data.resize(last + strm.size() + 4 * streams_per_work_group);

    cl_uchar *ptr = all_the_data.data() + last;
    memcpy(ptr, strm.data(), strm.size());
    ptr += strm.size();

    memcpy(ptr, &states[states_offset], 4 * streams_per_work_group);
    states_offset += streams_per_work_group;
  }

  cl_mem data_buf =
    clCreateBuffer(ctx, GetHostReadOnlyFlags(),
      all_the_data.size() * sizeof(cl_uchar), all_the_data.data(),
      &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(data_buf), &data_buf);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif

  // Allocate 256 * num interleaved slots for result
  size_t total_encoded = total_streams * kNumEncodedSymbols;
  cl_mem out_buf = clCreateBuffer(ctx, out_flags, total_encoded, NULL,
    &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(out_buf), &out_buf);

  // Run the kernel...
#ifndef NDEBUG
  // Make sure that we can launch enough kernels and that we have sufficient
  // space on the device.
  assert(streams_per_work_group <
    _gpu_ctx->GetDeviceInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE));

  // I don't know of a GPU implementation that uses more than 3 dims..
  assert(3 ==
    _gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));

  struct WorkGroupSizes {
    size_t sizes[3];
  };
  WorkGroupSizes wgsz =
    _gpu_ctx->GetDeviceInfo<WorkGroupSizes>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  assert(streams_per_work_group < wgsz.sizes[0]);

  size_t total_constant_memory = 0;
  total_constant_memory += all_the_data.size() * sizeof(cl_uchar);
  total_constant_memory += _M * sizeof(AnsTableEntry);

  assert(total_constant_memory <
    _gpu_ctx->GetDeviceInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE));
  assert(4 < _gpu_ctx->GetDeviceInfo<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS));
#endif

  CHECK_CL(clEnqueueNDRangeKernel, _gpu_ctx->GetCommandQueue(), decode_kernel,
    1, NULL, &total_streams, &streams_per_work_group, 0, NULL, NULL);

  // Read back the buffer...
  std::vector<cl_uchar> all_symbols = std::move(
    ReadBuffer<cl_uchar>(_gpu_ctx->GetCommandQueue(), out_buf, total_encoded));

  std::vector<std::vector<cl_uchar> > out;
  out.reserve(total_streams);
  for (size_t i = 0; i < total_streams; ++i) {
    std::vector<cl_uchar> stream;
    size_t stream_start = i * kNumEncodedSymbols;
    size_t stream_end = stream_start + kNumEncodedSymbols;
    stream.insert(stream.begin(), all_symbols.begin() + stream_start, all_symbols.begin() + stream_end);
    out.push_back(std::move(stream));
  }

  // Release buffer objects...
  CHECK_CL(clReleaseMemObject, data_buf);
  CHECK_CL(clReleaseMemObject, out_buf);

  return std::move(out);
}

}  // namespace ocl
}  // namespace ans
