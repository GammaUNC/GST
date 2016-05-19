#ifndef __ANS_OPENCL_H__
#define __ANS_OPENCL_H__

#include <cassert>
#include <cstdint>
#include <vector>

#include "ans.h"
#include "histogram.h"
#include "gpu.h"

namespace ans {
namespace ocl {

  // OpenCL decoder that can decode multiple interleaved rANS streams
  // provided that the following conditions are met:
  // 1. OpenCL context has been established
  // 2. All streams are encoded with the same settings: k = 2^15, b = 2^16
  // 3. The sum of the symbol frequencies (F) is 2^12
  // 4. Each stream has exactly 256 symbols
  // 5. The alphabet has at most 256 symbols.

  std::unique_ptr<Encoder> CreateCPUEncoder(const std::vector<uint32_t> &F);
  std::unique_ptr<Decoder> CreateCPUDecoder(uint32_t state, const std::vector<uint32_t> &F);

  class OpenCLDecoder {
  public:
    OpenCLDecoder(
      const std::unique_ptr<gpu::GPUContext> &ctx,
      const std::vector<uint32_t> &F,
      const int num_interleaved);
    ~OpenCLDecoder();

    std::vector<cl_uchar> Decode(
      cl_uint state,
      const std::vector<cl_uchar> &data) const;

    std::vector<std::vector<cl_uchar>> Decode(
      const std::vector<cl_uint> &states,
      const std::vector<cl_uchar> &data) const;

    std::vector<std::vector<cl_uchar>> Decode(
      const std::vector<cl_uint> &states,
      const std::vector<std::vector<cl_uchar> > &data) const;

    void RebuildTable(const std::vector<uint32_t> &F);

    std::vector<cl_uchar> GetSymbols() const;
    std::vector<cl_ushort> GetFrequencies() const;
    std::vector<cl_ushort> GetCumulativeFrequencies() const;

  private:
    // Disallow copy construction...
    OpenCLDecoder(const OpenCLDecoder &);

    const int _num_interleaved;
    const size_t _M;

    const std::unique_ptr<gpu::GPUContext> &_gpu_ctx;

    cl_mem _table;
    cl_event _build_table_event;
    bool _built_table;

    cl_mem_flags GetHostReadOnlyFlags() const {
      #ifdef CL_VERSION_1_2
            return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
      #else
            return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
      #endif
    }
  };

}  // namespace ocl
}  // namespace ans

#endif  // __ANS_OPENCL_H__
