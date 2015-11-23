#include "ans_ocl.h"

#include <numeric>
#include <iostream>

#include "ans_config.h"

template<typename T>
static void InspectBuffer(cl_command_queue queue, cl_mem buffer, size_t num_elements) {
  std::vector<T> host_mem(num_elements);
  clEnqueueReadBuffer(queue, buffer, true, 0, num_elements * sizeof(T), host_mem.data(), 0, NULL, NULL);
  std::cout << "Inspecting buffer: ";
  size_t idx = 0;
  while (idx < host_mem.size()) {
    for (int i = 0; i < 10; ++i) {
      std::cout << host_mem[idx] << " ";
    }
    std::cout << std::endl;
    idx++;
  }
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
{
#ifdef CL_VERSION_1_2
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
#else
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
#endif

  const gpu::LoadedCLKernel *build_table_kernel = GetTableBuildingKernel(ctx, device);

  uint32_t *freqs_ptr = const_cast<uint32_t *>(F.data());

  cl_int errCreateBuffer;
  cl_mem freqs_buffer = clCreateBuffer(ctx, flags, F.size() * sizeof(freqs_ptr[0]), freqs_ptr, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 0, sizeof(freqs_buffer), &freqs_buffer);

  size_t num_freqs = F.size();
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 1, sizeof(size_t *), &num_freqs);

  cl_mem table_freqs = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, _M * sizeof(cl_ushort), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem table_cumulative_freqs = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, _M * sizeof(cl_ushort), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_mem table_symbols = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, _M * sizeof(cl_uchar), NULL, &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 2, sizeof(table_freqs), &table_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 3, sizeof(table_cumulative_freqs), &table_cumulative_freqs);
  CHECK_CL(clSetKernelArg, build_table_kernel->_kernel, 4, sizeof(table_symbols), &table_symbols);

  size_t global_work_size = _M;
  CHECK_CL(clEnqueueNDRangeKernel, build_table_kernel->_command_queue, build_table_kernel->_kernel, 1, NULL,
    &global_work_size, 0, 0, NULL, NULL);

  InspectBuffer<cl_ushort>(build_table_kernel->_command_queue, table_freqs, _M);
  InspectBuffer<cl_ushort>(build_table_kernel->_command_queue, table_cumulative_freqs, _M);
  InspectBuffer<cl_uchar>(build_table_kernel->_command_queue, table_symbols, _M);
}

}  // namespace ans