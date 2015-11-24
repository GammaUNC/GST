#ifndef __ANS_OPENCL_H__
#define __ANS_OPENCL_H__

#include <cassert>
#include <cstdint>
#include <vector>

#include "gpu.h"

namespace ans {
  // OpenCL decoder that can decode multiple interleaved rANS streams
  // provided that the following conditions are met:
  // 1. OpenCL context has been established
  // 2. All streams are encoded with the same settings: k = 2^15, b = 2^16
  // 3. The sum of the symbol frequencies (F) is < 2^16
  // 4. Each stream has exactly 256 symbols
  // 5. The alphabet has at most 256 symbols.
  class OpenCLDecoder {
  public:
    OpenCLDecoder(
      cl_context ctx, cl_device_id device,
      const std::vector<uint32_t> &F,
      const int num_interleaved);
    ~OpenCLDecoder();

    bool Decode(
      std::vector<std::vector<uint32_t> > *out,
      const std::vector<uint32_t> &states,
      const std::vector<uint16_t> &data);

    void RebuildTable(const std::vector<uint32_t> &F) const;

    std::vector<cl_uchar> GetSymbols() const;
    std::vector<cl_ushort> GetFrequencies() const;
    std::vector<cl_ushort> GetCumulativeFrequencies() const;

  private:
    // Disallow copy construction...
    OpenCLDecoder(const OpenCLDecoder &);

    const int _num_interleaved;
    const size_t _M;

    static const gpu::LoadedCLKernel *GetTableBuildingKernel(cl_context ctx, cl_device_id device);

    cl_context _ctx;
    cl_device_id _device;

    cl_mem _table_symbols;
    cl_mem _table_frequencies;
    cl_mem _table_cumulative_frequencies;
  };

}  // namespace ans

#endif  // __ANS_OPENCL_H__