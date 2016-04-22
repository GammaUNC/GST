#ifndef __TCAR_CODEC_H__
#define __TCAR_CODEC_H__

#include <cstdint>
#include <functional>
#include <vector>

#include "dxt_image.h"
#include "gpu.h"

#ifdef __APPLE__
#  define GLFW_INCLUDE_GLCOREARB 1
#  define GL_GLEXT_PROTOTYPES 1
#  define GLFW_INCLUDE_GLEXT 1
#  include <GLFW/glfw3.h>
#  include <OpenGL/opengl.h>
#elif defined (_MSC_VER)
#  include <GL/glew.h>
#  include <GLFW/glfw3.h>
#else
#  define GL_GLEXT_PROTOTYPES 1
#  define GLFW_INCLUDE_GLEXT 1
#  include <GLFW/glfw3.h>
#  include <GL/glx.h>
#endif

#include "gl_guards.h"

namespace GenTC {
  // Compresses the DXT texture with the given width and height into a
  // GPU decompressible stream.
  std::vector<uint8_t> CompressDXT(const char *filename, const char *cmp_fn);
  std::vector<uint8_t> CompressDXT(int width, int height,
                                   const std::vector<uint8_t> &rgb_data,
                                   const std::vector<uint8_t> &dxt_data);

  DXTImage DecompressDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                         const std::vector<uint8_t> &cmp_data);

  void LoadCompressedDXTInto(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                             const std::vector<uint8_t> &cmp_data, GLuint pbo, GLuint texID);

  GLuint LoadCompressedDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                           const std::vector<uint8_t> &cmp_data, GLuint pbo);

  struct GenTCHeader {
    uint32_t width;
    uint32_t height;
    uint32_t ep1_y_cmp_sz;
    uint32_t ep1_co_cmp_sz;
    uint32_t ep1_cg_cmp_sz;
    uint32_t ep2_y_cmp_sz;
    uint32_t ep2_co_cmp_sz;
    uint32_t ep2_cg_cmp_sz;
    uint32_t palette_data_sz;
    uint32_t palette_cmp_sz;
    uint32_t indices_cmp_sz;

    void Print() const;
  };

  struct AsyncCallbackData;
  class CompressedDXTAsyncRequest {
   public:
    CompressedDXTAsyncRequest();
    void SetData(GLuint pbo, GLuint texID, GLsizei w, GLsizei h, std::function<void()> callback);
    bool IsReady() const;
    GLuint TextureHandle() const;
    void LoadTexture();
   private:
    std::shared_ptr<AsyncCallbackData> _data;
    friend void LoadCompressedDXTAsync(
      const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
      const std::vector<uint8_t> &cmp_data,
      const CompressedDXTAsyncRequest *req);
  };

  void LoadCompressedDXTAsync(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                              const std::vector<uint8_t> &cmp_data,
                              const CompressedDXTAsyncRequest *req);

  bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
               const char *filename, const char *cmp_fn);

}  // namespace GenTC

#endif  // __TCAR_CODEC_H__
