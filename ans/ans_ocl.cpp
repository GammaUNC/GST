#include "ans_ocl.h"

#include <numeric>
#include <iostream>

#include "ans_config.h"

template<typename T>
static std::vector<T> ReadBuffer(cl_command_queue queue, cl_mem buffer, size_t num_elements) {
  std::vector<T> host_mem(num_elements);
#ifdef CL_VERSION_1_2
  CHECK_CL(clEnqueueBarrierWithWaitList, queue, 0, NULL, NULL);
#else
  CHECK_CL(clEnqueueBarrier, queue);
#endif
  CHECK_CL(clEnqueueReadBuffer, queue, buffer, true, 0, num_elements * sizeof(T), host_mem.data(), 0, NULL, NULL);
#ifndef NDEBUG
  std::cout << "Reading buffer..." << std::endl;
  int cnt = 0;
  for (auto e : host_mem) {
    std::cout << static_cast<uint32_t>(e) << " ";
    if (++cnt == 10) {
      cnt = 0;
      std::cout << std::endl;
    }
  }
  if (10 != cnt) std::cout << std::endl;
#endif
  return std::move(host_mem);
}

namespace ans {

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

OpenCLDecoder::OpenCLDecoder(
  cl_context ctx, cl_device_id device, const std::vector<uint32_t> &F, const int num_interleaved)
  : _num_interleaved(num_interleaved)
  , _M(std::accumulate(F.begin(), F.end(), 0))
  , _ctx(ctx)
  , _device(device)
{
  cl_int errCreateBuffer;
  _table_frequencies = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _M * sizeof(cl_ushort), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  _table_cumulative_frequencies = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _M * sizeof(cl_ushort), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  _table_symbols = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, _M * sizeof(cl_uchar), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  RebuildTable(F);
}

OpenCLDecoder::~OpenCLDecoder() {
  CHECK_CL(clReleaseMemObject, _table_frequencies);
  CHECK_CL(clReleaseMemObject, _table_cumulative_frequencies);
  CHECK_CL(clReleaseMemObject, _table_symbols);
}

std::vector<cl_uchar> OpenCLDecoder::GetSymbols() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  return std::move(ReadBuffer<cl_uchar>(build_table_kernel->_command_queue, _table_symbols, _M));
}

std::vector<cl_ushort> OpenCLDecoder::GetFrequencies() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  return std::move(ReadBuffer<cl_ushort>(build_table_kernel->_command_queue, _table_frequencies, _M));
}

std::vector<cl_ushort> OpenCLDecoder::GetCumulativeFrequencies() const {
  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(_ctx, _device);
  return std::move(ReadBuffer<cl_ushort>(build_table_kernel->_command_queue, _table_cumulative_frequencies, _M));
}

void OpenCLDecoder::RebuildTable(const std::vector<uint32_t> &F) const {
  assert(_M == std::accumulate(F.begin(), F.end(), 0));
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

  cl_uint num_freqs = static_cast<cl_uint>(F.size());

  // Note: we could do this on the GPU as well, but the array size here is almost never more than
  // about 256, so the CPU is actually much better at doing it. We can also stick it in constant
  // memory, which makes the upload not that bad...
  std::vector<uint32_t> cum_freqs(num_freqs, 0);
  std::partial_sum(F.begin(), F.end() - 1, cum_freqs.begin() + 1);

  uint32_t *freqs_ptr = const_cast<uint32_t *>(F.data());
  uint32_t *cum_freqs_ptr = cum_freqs.data();

  cl_int errCreateBuffer;
  cl_mem freqs_buffer = clCreateBuffer(_ctx, flags, F.size() * sizeof(freqs_ptr[0]), freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem cum_freqs_buffer = clCreateBuffer(_ctx, flags, cum_freqs.size() * sizeof(cum_freqs_ptr[0]), cum_freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 0, sizeof(freqs_buffer), &freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 1, sizeof(cum_freqs_buffer), &cum_freqs_buffer);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 2, sizeof(cl_uint), &num_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 3, sizeof(_table_frequencies), &_table_frequencies);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 4, sizeof(_table_cumulative_frequencies), &_table_cumulative_frequencies);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 5, sizeof(_table_symbols), &_table_symbols);

  size_t local_work_size = 256;
  CHECK_CL(clEnqueueNDRangeKernel, build_table_kernel->_command_queue,
    build_table_kernel->_kernel, 1, NULL, &_M, &local_work_size, 0, NULL, NULL);

  CHECK_CL(clReleaseMemObject, freqs_buffer);
  CHECK_CL(clReleaseMemObject, cum_freqs_buffer);
}

}  // namespace ans