#ifndef __GENTC_GPU_H__
#define __GENTC_GPU_H__

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#elif defined (_WIN32)
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#  include <CL/cl_ext.h>
#else
#  include <CL/cl_ext.h>
#  include <CL/cl_gl.h>
#endif

#include "cl_guards.h"

#include <cstdio>
#include <cassert>
#include <memory>
#include <vector>

namespace gpu {

  enum EContextType {
    eContextType_GenericGPU,
    eContextType_IntelCPU
  };

  class GPUContext {
  public:
    ~GPUContext();
    static std::unique_ptr<GPUContext> InitializeOpenCL(bool share_opengl);

    cl_command_queue GetCommandQueue() const { return _command_queue; }
    cl_device_id GetDeviceID() const { return _device;  }
    cl_context GetOpenCLContext() const { return _ctx; }

    cl_kernel GetOpenCLKernel(const std::string &filename, const std::string &kernel) const;
    void PrintDeviceInfo() const;

    EContextType Type() const { return _type; }

    template<typename T>
    T GetDeviceInfo(cl_device_info param) const {
      cl_uchar ret_buffer[256];
      size_t bytes_read;
      CHECK_CL(clGetDeviceInfo, GetDeviceID(), param, sizeof(ret_buffer),
                                ret_buffer, &bytes_read);
      assert(bytes_read < sizeof(ret_buffer));
      assert(bytes_read == sizeof(T));
      return reinterpret_cast<const T *>(ret_buffer)[0];
    }

    template<typename T>
    T GetKernelWGInfo(const std::string &filename, const std::string &kernel,
                      cl_kernel_work_group_info param) const {
      cl_kernel k = GetOpenCLKernel(filename, kernel);
      cl_uchar ret_buffer[256];
      size_t bytes_read;
      CHECK_CL(clGetKernelWorkGroupInfo, k, _device, param, sizeof(ret_buffer),
                                         ret_buffer, &bytes_read);
      assert(bytes_read < sizeof(ret_buffer));
      assert(bytes_read == sizeof(T));
      return reinterpret_cast<const T *>(ret_buffer)[0];
    }

  private:
    GPUContext() { }
    GPUContext(const GPUContext &);

    cl_command_queue _command_queue;
    cl_device_id _device;
    cl_context _ctx;

    EContextType _type;
  };

}  // namespace gpu

#endif //  __GENTC_GPU_H__
