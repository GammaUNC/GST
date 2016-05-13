#ifndef __GENTC_GPU_H__
#define __GENTC_GPU_H__

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#elif defined (_WIN32)
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#  include <CL/cl_ext.h>
#  include <CL/cl_gl_ext.h>
#else
#  include <CL/cl_ext.h>
#  include <CL/cl_gl.h>
#  include <CL/cl_gl_ext.h>
#endif

#ifdef _MSC_VER
#define STDCALL __stdcall
#else
#define STDCALL
#endif

#include "cl_guards.h"

#include <atomic>
#include <cstdio>
#include <cassert>
#include <memory>
#include <mutex>
#include <vector>
#ifndef NDEBUG
#include <iostream>
#endif

namespace gpu {

  enum EContextType {
    eContextType_GenericGPU,
    eContextType_IntelCPU
  };

  enum EOpenCLVersion {
    eOpenCLVersion_10,
    eOpenCLVersion_11,
    eOpenCLVersion_12,
    eOpenCLVersion_20
  };

  static const int kMaxNumWorkQueues = 4;
  class GPUContext {
  public:
    ~GPUContext();
    static std::unique_ptr<GPUContext> InitializeOpenCL(bool share_opengl);

    cl_command_queue GetDefaultCommandQueue() const { return _default_command_queue; }
    cl_command_queue GetNextQueue() const {
      int next = _next_work_queue++;
      return _work_queues[next % _num_work_queues];
    }

    void FlushAllQueues() const {
      CHECK_CL(clFlush, _default_command_queue);
      for (size_t i = 0; i < _num_work_queues; ++i) {
        CHECK_CL(clFlush, _work_queues[i]);
      }
    }

    cl_device_id GetDeviceID() const { return _device;  }
    cl_context GetOpenCLContext() const { return _ctx; }

    cl_kernel GetOpenCLKernel(const std::string &filename, const std::string &kernel) const;
    void PrintDeviceInfo() const;

    EContextType Type() const { return _type; }
    EOpenCLVersion Version() const { return _version; }

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

    struct LocalMemoryKernelArg {
      size_t _local_mem_sz;
    };

    template<cl_uint WorkDim, typename... Args>
    void EnqueueOpenCLKernel(cl_command_queue queue,
                             const std::string &filename, const std::string &kernel,
                             const size_t *global_sz, const size_t *local_sz,
                             cl_uint num_events, const cl_event *events, cl_event *ret_event,
                             Args... kernel_args) {
      std::unique_lock<std::mutex> lock(_enqueue_mutex);
      cl_kernel k = GetOpenCLKernel(filename, kernel);
      SetArgument(k, 0, kernel_args...);
#ifndef NDEBUG
      CHECK_CL(clFinish, queue);
      std::cout << "enqueuing: " << kernel.c_str();
      std::cout.flush();
#endif
      CHECK_CL(clEnqueueNDRangeKernel, queue, k,
                                       WorkDim, NULL, global_sz, local_sz,
                                       num_events, events, ret_event);
#ifndef NDEBUG
      std::cout << " ...";
      std::cout.flush();
      CHECK_CL(clFinish, queue);
      std::cout << "Done" << std::endl;
#endif
    }

  private:
    GPUContext() : _num_work_queues(0), _next_work_queue(0) { }
    GPUContext(const GPUContext &) { }

    void SetArgument(cl_kernel kernel, unsigned idx, LocalMemoryKernelArg mem) {
      CHECK_CL(clSetKernelArg, kernel, idx, mem._local_mem_sz, NULL);
    }

    template<typename T>
    void SetArgument(cl_kernel kernel, unsigned idx, T arg) {
      CHECK_CL(clSetKernelArg, kernel, idx, sizeof(T), &arg);
    }

    template<typename T, typename... Args>
    void SetArgument(cl_kernel kernel, unsigned idx, T arg, Args... rest) {
      CHECK_CL(clSetKernelArg, kernel, idx, sizeof(T), &arg);
      SetArgument(kernel, idx + 1, rest...);
    }

    template<typename... Args>
    void SetArgument(cl_kernel kernel, unsigned idx, LocalMemoryKernelArg arg, Args... rest) {
      CHECK_CL(clSetKernelArg, kernel, idx, arg._local_mem_sz, NULL);
      SetArgument(kernel, idx + 1, rest...);
    }

    cl_command_queue _default_command_queue;
    cl_device_id _device;
    cl_context _ctx;

    size_t _num_work_queues;
    mutable std::atomic_int _next_work_queue;
    cl_command_queue _work_queues[kMaxNumWorkQueues];

    std::mutex _enqueue_mutex;

    EContextType _type;
    EOpenCLVersion _version;
  };

}  // namespace gpu

#endif //  __GENTC_GPU_H__
