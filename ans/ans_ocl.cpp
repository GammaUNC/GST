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

static std::vector<cl_uint> NormalizeFrequencies(const std::vector<int> &F) {
  std::vector<int> freqs = std::move(ans::GenerateHistogram(F, kANSTableSize));
  assert(freqs.size() == F.size());

  std::vector<cl_uint> result;
  result.reserve(freqs.size());
  for (const auto freq : freqs) {
    result.push_back(static_cast<cl_uint>(freq));
  }
  return std::move(result);
}

OpenCLEncoder::OpenCLEncoder(const std::vector<int> &F)
  : OpenCLEncoderBase(std::move(NormalizeFrequencies(F))) { }

OpenCLCPUDecoder::OpenCLCPUDecoder(cl_uint state, const std::vector<int> &F)
  : OpenCLDecoderBase(state, std::move(NormalizeFrequencies(F))) { }

OpenCLDecoder::OpenCLDecoder(
  const std::unique_ptr<gpu::GPUContext> &ctx, const std::vector<int> &F, const int num_interleaved)
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

void OpenCLDecoder::RebuildTable(const std::vector<int> &F) const {
  std::vector<cl_uint> freqs = std::move(NormalizeFrequencies(F));
  assert(_M == std::accumulate(freqs.begin(), freqs.end(), 0));

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

  // Offsets: sizeof the singular data stream
  cl_uint offset = static_cast<cl_uint>(data.size() / 2);
  cl_mem offset_buffer = clCreateBuffer(ctx, GetHostReadOnlyFlags(), sizeof(cl_uint), &offset, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 1, sizeof(offset_buffer), &offset_buffer);

  // Data: just send the data pointer...
  cl_uchar *data_ptr = const_cast<cl_uchar *>(data.data());
  cl_mem data_buffer = clCreateBuffer(ctx, GetHostReadOnlyFlags(), data.size() * sizeof(data_ptr[0]), data_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 2, sizeof(data_buffer), &data_buffer);

  // State: single state...
  cl_mem state_buffer = clCreateBuffer(ctx, GetHostReadOnlyFlags(), sizeof(cl_uint), &state, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 3, sizeof(state_buffer), &state_buffer);

#ifdef CL_VERSION_1_2
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
#else
  cl_mem_flags out_flags = CL_MEM_WRITE_ONLY;
#endif

  // Allocate 256 slots for result
  cl_mem out_buffer = clCreateBuffer(ctx, out_flags, kNumEncodedSymbols, NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);
  CHECK_CL(clSetKernelArg, decode_kernel, 4, sizeof(out_buffer), &out_buffer);

  // Run the kernel...
  const size_t num_streams = 1;
  CHECK_CL(clEnqueueNDRangeKernel, _gpu_ctx->GetCommandQueue(), decode_kernel, 1, NULL, &num_streams, NULL, 0, NULL, NULL);

  std::vector<cl_uchar> out = std::move(
    ReadBuffer<cl_uchar>(_gpu_ctx->GetCommandQueue(), out_buffer, kNumEncodedSymbols));

  // Release buffer objects...
  CHECK_CL(clReleaseMemObject, offset_buffer);
  CHECK_CL(clReleaseMemObject, data_buffer);
  CHECK_CL(clReleaseMemObject, state_buffer);
  CHECK_CL(clReleaseMemObject, out_buffer);

  return std::move(out);
}

std::vector<std::vector<cl_uchar> > OpenCLDecoder::Decode(
  const std::vector<cl_uint> &states,
  const std::vector<cl_uchar> &data) const {
  return std::vector<std::vector<cl_uchar> >();
}

std::vector<std::vector<cl_uchar> > OpenCLDecoder::Decode(
  const std::vector<cl_uint> &states,
  const std::vector<std::vector<cl_uchar> > &data) const {
  return std::vector<std::vector<cl_uchar> >();
}

}  // namespace ans
