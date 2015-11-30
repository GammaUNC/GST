#include "ans_ocl.h"

#include <numeric>
#include <iostream>

#include "ans_config.h"
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

static std::vector<uint32_t> NormalizeFrequencies(const std::vector<int> &F) {
  std::vector<int> freqs = std::move(ans::GenerateHistogram(F, kANSTableSize));
  assert(freqs.size() == F.size());

  std::vector<uint32_t> result;
  result.reserve(freqs.size());
  for (const auto freq : freqs) {
    result.push_back(static_cast<uint32_t>(freq));
  }
  return std::move(result);
}

// !FIXME! We should really put in some sort of logic to unload these kernels...
gpu::LoadedCLKernel _ourTableBuildingKernel;
gpu::LoadedCLKernel *gTableBuildingKernel = NULL;
const gpu::LoadedCLKernel *OpenCLDecoder::GetTableBuildingKernel(cl_context ctx, cl_device_id device) {
  if (NULL != gTableBuildingKernel) {
    return gTableBuildingKernel;
  }

  _ourTableBuildingKernel = gpu::InitializeOpenCLKernel(
    kANSOpenCLKernels[eANSOpenCLKernel_BuildTable], "build_table", ctx, device);
  gTableBuildingKernel = &_ourTableBuildingKernel;
  return gTableBuildingKernel;
}

OpenCLEncoder::OpenCLEncoder(const std::vector<int> &F)
  : OpenCLEncoderBase(std::move(NormalizeFrequencies(F))) { }

OpenCLCPUDecoder::OpenCLCPUDecoder(uint32_t state, const std::vector<int> &F)
  : OpenCLDecoderBase(state, std::move(NormalizeFrequencies(F))) { }

OpenCLDecoder::OpenCLDecoder(
  cl_context ctx, cl_device_id device, const std::vector<int> &F, const int num_interleaved)
  : _num_interleaved(num_interleaved)
  , _M(kANSTableSize)
  , _ctx(ctx)
  , _device(device)
{
  cl_int errCreateBuffer;
  _table = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _M * sizeof(AnsTableEntry), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  RebuildTable(F);
}

OpenCLDecoder::~OpenCLDecoder() {
  CHECK_CL(clReleaseMemObject, _table);
}

std::vector<cl_uchar> OpenCLDecoder::GetSymbols() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(build_table_kernel->_command_queue, _table, _M));

  std::vector<cl_uchar> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.symbol);
  }

  return std::move(result);
}

std::vector<cl_ushort> OpenCLDecoder::GetFrequencies() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(build_table_kernel->_command_queue, _table, _M));

  std::vector<cl_ushort> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.freq);
  }

  return std::move(result);
}

std::vector<cl_ushort> OpenCLDecoder::GetCumulativeFrequencies() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  std::vector<AnsTableEntry> table =
    std::move(ReadBuffer<AnsTableEntry>(build_table_kernel->_command_queue, _table, _M));

  std::vector<cl_ushort> result;
  result.reserve(table.size());

  for (auto entry : table) {
    result.push_back(entry.cum_freq);
  }

  return std::move(result);
}

void OpenCLDecoder::RebuildTable(const std::vector<int> &F) const {
  std::vector<uint32_t> freqs = std::move(NormalizeFrequencies(F));
  assert(_M == std::accumulate(freqs.begin(), freqs.end(), 0));
#ifdef CL_VERSION_1_2
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
#else
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
#endif

  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);

#ifndef NDEBUG
  size_t work_group_size;
  CHECK_CL(clGetKernelWorkGroupInfo, build_table_kernel->_kernel, _device, CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t), &work_group_size, NULL);
  assert(work_group_size >= 32);
#endif

  cl_uint num_freqs = static_cast<cl_uint>(freqs.size());

  // Note: we could do this on the GPU as well, but the array size here is almost never more than
  // about 256, so the CPU is actually much better at doing it. We can also stick it in constant
  // memory, which makes the upload not that bad...
  std::vector<uint32_t> cum_freqs(num_freqs, 0);
  std::partial_sum(freqs.begin(), freqs.end() - 1, cum_freqs.begin() + 1);

  uint32_t *freqs_ptr = const_cast<uint32_t *>(freqs.data());
  uint32_t *cum_freqs_ptr = cum_freqs.data();

  cl_int errCreateBuffer;
  cl_mem freqs_buffer = clCreateBuffer(_ctx, flags, freqs.size() * sizeof(freqs_ptr[0]), freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem cum_freqs_buffer = clCreateBuffer(_ctx, flags, cum_freqs.size() * sizeof(cum_freqs_ptr[0]), cum_freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 0, sizeof(freqs_buffer), &freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 1, sizeof(cum_freqs_buffer), &cum_freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 2, sizeof(cl_uint), &num_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 3, sizeof(_table), &_table);

  CHECK_CL(clEnqueueNDRangeKernel, build_table_kernel->_command_queue,
    build_table_kernel->_kernel, 1, NULL, &_M, NULL, 0, NULL, NULL);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);
}

bool OpenCLDecoder::Decode(
  std::vector<uint8_t> *out,
  const uint32_t state,
  const std::vector<uint8_t> &data) {

  return false;
}

bool OpenCLDecoder::Decode(
  std::vector<std::vector<uint8_t> > *out,
  const std::vector<uint32_t> &states,
  const std::vector<uint8_t> &data) {
  return false;
}

bool OpenCLDecoder::Decode(
  std::vector<std::vector<uint8_t> > *out,
  const std::vector<uint32_t> &states,
  const std::vector<std::vector<uint8_t> > &data) {
  return false;
}

}  // namespace ans
