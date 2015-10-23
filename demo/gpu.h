#ifndef __GENTC_GPU_H__
#define __GENTC_GPU_H__

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#endif

#define GL_GLEXT_PROTOTYPES 1
#define GLFW_INCLUDE_GLEXT 1
#include <GLFW/glfw3.h>

#ifndef NDEBUG
#  define CHECK_CL(fn, ...)                                               \
  do {                                                                    \
    cl_int err = fn(__VA_ARGS__);                                         \
    if(CL_SUCCESS != err) {                                               \
      const char *errMsg = clErrMsg(err);                                 \
      fprintf(stderr, "OpenCL error at line %d: %s\n", __LINE__, errMsg); \
      assert (false);                                                     \
    }                                                                     \
  } while(0)
#else
#  define CHECK_CL(fn, ...) do { (void)(fn(__VA_ARGS__)); } while(0)
#endif

#ifdef CL_VERSION_1_2
  // If we're using OpenCL 1.2 or later, we should use the non-deprecated version of
  // the clCreateImage2D function to avoid those pesky compiler warnings...
  static cl_mem clCreateImage2D12(cl_context ctx, cl_mem_flags flags,
                                  const cl_image_format *fmt, size_t w, size_t h,
                                  size_t rowBytes, void *host_ptr, cl_int *errCode) {
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = w;
    desc.image_height = h;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = rowBytes;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    return clCreateImage(ctx, flags, fmt, &desc, host_ptr, errCode);
  }
#else
  static cl_int clUnloadCompiler11(cl_platform_id) {
    return clUnloadCompiler();
  }
#endif

  typedef cl_int (*clUnloadCompilerFunc)(cl_platform_id);
  typedef cl_mem (*clCreateImage2DFunc)(cl_context, cl_mem_flags, const cl_image_format *,
                                        size_t, size_t, size_t, void *, cl_int *);
#ifdef CL_VERSION_1_2
  static const clCreateImage2DFunc gCreateImage2DFunc = clCreateImage2D12;
  static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadPlatformCompiler;
#else
  static const clCreateImage2DFunc gCreateImage2DFunc = clCreateImage2D;
  static const clUnloadCompilerFunc gUnloadCompilerFunc = clUnloadCompiler11;
#endif

extern void InitializeOpenCLKernel();
extern void DestroyOpenCLKernel();
extern void RunKernel(unsigned char *data, GLuint texID, int x, int y, int channels);

#endif //  __GENTC_GPU_H__
