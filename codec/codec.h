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

  struct AsyncCallbackData;
  class CompressedDXTAsyncRequest {
   public:
    CompressedDXTAsyncRequest(const AsyncCallbackData &data);
    bool IsReady() const;
    GLuint TextureHandle() const;
    void LoadTexture();
   private:
    std::shared_ptr<AsyncCallbackData> _data;
    friend CompressedDXTAsyncRequest LoadCompressedDXTAsync(
      const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
      const std::vector<uint8_t> &cmp_data,
      std::function<void()> callback);
  };

  CompressedDXTAsyncRequest LoadCompressedDXTAsync(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
                                                   const std::vector<uint8_t> &cmp_data,
                                                   std::function<void()> callback);

  bool TestDXT(const std::unique_ptr<gpu::GPUContext> &gpu_ctx,
               const char *filename, const char *cmp_fn);

}  // namespace GenTC

#endif  // __TCAR_CODEC_H__
