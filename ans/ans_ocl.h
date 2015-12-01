#ifndef __ANS_OPENCL_H__
#define __ANS_OPENCL_H__

#include <cassert>
#include <cstdint>
#include <vector>

#include "ans_encode.h"
#include "ans_decode.h"
#include "gpu.h"

namespace ans {
  // OpenCL decoder that can decode multiple interleaved rANS streams
  // provided that the following conditions are met:
  // 1. OpenCL context has been established
  // 2. All streams are encoded with the same settings: k = 2^15, b = 2^16
  // 3. The sum of the symbol frequencies (F) is 2^12
  // 4. Each stream has exactly 256 symbols
  // 5. The alphabet has at most 256 symbols.

  // If we expect our symbol frequency to have 1 << 11 precision, we only have 1 << 4
  // available for state-precision
  typedef Encoder< (1 << 16), (1 << 4) > OpenCLEncoderBase;
  typedef Decoder< (1 << 16), (1 << 4) > OpenCLDecoderBase;
  static const int kANSTableSize = (1 << 11);
  static const int kNumEncodedSymbols = 256;

  class OpenCLEncoder : public OpenCLEncoderBase {
  public:
    OpenCLEncoder(const std::vector<int> &F);
  };

  class OpenCLCPUDecoder : public OpenCLDecoderBase {
  public:
    OpenCLCPUDecoder(uint32_t state, const std::vector<int> &F);
  };

  class OpenCLDecoder {
  public:
    OpenCLDecoder(
      cl_context ctx, cl_device_id device,
      const std::vector<int> &F,
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

    void RebuildTable(const std::vector<int> &F) const;

    std::vector<cl_uchar> GetSymbols() const;
    std::vector<cl_ushort> GetFrequencies() const;
    std::vector<cl_ushort> GetCumulativeFrequencies() const;

  private:
    // Disallow copy construction...
    OpenCLDecoder(const OpenCLDecoder &);

    const int _num_interleaved;
    const size_t _M;

    cl_context _ctx;
    cl_device_id _device;

    cl_mem _table;

    cl_mem_flags GetHostReadOnlyFlags() const {
      #ifdef CL_VERSION_1_2
            return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
      #else
            return CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
      #endif
    }
  };

}  // namespace ans

#endif  // __ANS_OPENCL_H__
