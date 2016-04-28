#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4312 )
#endif  // _MSC_VER
#include "stb_image.h"
#ifdef _MSC_VER
#pragma warning( pop )
#endif  // _MSC_VER

#include "codec.h"

#ifdef __APPLE__
#  define GLFW_INCLUDE_GLCOREARB 1
#  define GL_GLEXT_PROTOTYPES 1
#  define GLFW_INCLUDE_GLEXT 1
#  include <GLFW/glfw3.h>
#  include <OpenGL/opengl.h>
#elif defined (_WIN32)
#  include <GL/glew.h>
#  include <GLFW/glfw3.h>
#else
#  define GL_GLEXT_PROTOTYPES 1
#  define GLFW_INCLUDE_GLEXT 1
#  include <GLFW/glfw3.h>
#  include <GL/glx.h>
#endif

#include "gl_guards.h"

#define GLIML_NO_PVR
#include "gliml\gliml.h"

#define CRND_HEADER_FILE_ONLY
#include "crn_decomp.h"

#define TEST_GTC

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static bool gPaused = false;
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  if ((key == GLFW_KEY_P) && action == GLFW_PRESS)
    gPaused = !gPaused;
}

const char *kVertexProg =
  "#version 110\n"
  ""
  "attribute vec3 position;\n"
  "attribute vec2 texCoord;\n"
  ""
  "varying vec2 uv;\n"
  ""
  "void main() {\n"
  "  gl_Position = vec4(position, 1.0);\n"
  "  uv = texCoord;\n"
  "}\n";

const char *kFragProg =
  "#version 110\n"
  ""
  "varying vec2 uv;\n"
  ""
  "uniform sampler2D tex;\n"
  ""
  "void main() {\n"
  "  gl_FragColor = vec4(texture2D(tex, uv).rgb, 1);\n"
  "}\n";

GLuint LoadShaders() {
  GLuint vertShdrID = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragShdrID = glCreateShader(GL_FRAGMENT_SHADER);

  glShaderSource(vertShdrID, 1, &kVertexProg , NULL);
  glCompileShader(vertShdrID);

  int result, logLength;
  
  glGetShaderiv(vertShdrID, GL_COMPILE_STATUS, &result);
  if (result != GL_TRUE) {
    glGetShaderiv(vertShdrID, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> VertexShaderErrorMessage(logLength);
    glGetShaderInfoLog(vertShdrID, logLength, NULL, &VertexShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
    fprintf(stdout, "Vertex shader compilation failed!\n");
    exit(1);
  }

  // Compile Fragment Shader
  glShaderSource(fragShdrID, 1, &kFragProg, NULL);
  glCompileShader(fragShdrID);

  // Check Fragment Shader
  glGetShaderiv(fragShdrID, GL_COMPILE_STATUS, &result);
  if (result != GL_TRUE) {
    glGetShaderiv(fragShdrID, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> FragmentShaderErrorMessage(logLength);
    glGetShaderInfoLog(fragShdrID, logLength, NULL, &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "Fragment shader compilation failed!\n");
    exit(1);
  }

  // Link the program
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vertShdrID);
  glAttachShader(prog, fragShdrID);
  glLinkProgram(prog);

  // Check the program
  glGetProgramiv(prog, GL_LINK_STATUS, &result);
  if (result != GL_TRUE) {
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> ProgramErrorMessage( std::max(logLength, int(1)) );
    glGetProgramInfoLog(prog, logLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
  }

  glDeleteShader(vertShdrID);
  glDeleteShader(fragShdrID);

  return prog;
}

void LoadGTC(const std::unique_ptr<gpu::GPUContext> &ctx,
             GLuint pbo, GLuint texID, const std::string &filePath) {
  // Load in compressed data.
  std::ifstream is (filePath.c_str(), std::ifstream::binary);
  if (!is) {
    assert(!"Error opening GenTC texture!");
    return;
  }

  is.seekg(0, is.end);
  size_t length = static_cast<size_t>(is.tellg());
  is.seekg(0, is.beg);

  std::vector<uint8_t> cmp_data(length);
  is.read(reinterpret_cast<char *>(cmp_data.data()), length);
  assert(is);
  is.close();

//  glFlush();
//  glFinish();

  // Grab the header from the data
  GenTC::GenTCHeader hdr;
  hdr.LoadFrom(cmp_data.data());

  // Create the data for OpenCL
  cl_int errCreateBuffer;
  static const size_t kHeaderSz = sizeof(hdr);
  cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  cl_mem cmp_buf = clCreateBuffer(ctx->GetOpenCLContext(), flags, cmp_data.size() - kHeaderSz,
                                  const_cast<uint8_t *>(cmp_data.data() + kHeaderSz),
                                  &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Create an OpenGL handle to our pbo
  // !SPEED! We don't need to recreate this every time....
  cl_mem output = clCreateFromGLBuffer(ctx->GetOpenCLContext(), CL_MEM_READ_WRITE, pbo,
                                       &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Acquire the PBO
  cl_event acquire_event;
  CHECK_CL(clEnqueueAcquireGLObjects, ctx->GetDefaultCommandQueue(),
                                      1, &output, 0, NULL, &acquire_event);

  // Load it
  std::vector<cl_event> cmp_events =
    std::move(GenTC::LoadCompressedDXT(ctx, hdr, cmp_buf, output, &acquire_event));

  // Release the PBO
  cl_event release_event;
  CHECK_CL(clEnqueueReleaseGLObjects, ctx->GetDefaultCommandQueue(),
                                      1, &output,
                                      static_cast<cl_uint>(cmp_events.size()),
                                      cmp_events.data(), &release_event);

  CHECK_CL(clFlush, ctx->GetCommandQueue());

  // Wait on the release
  CHECK_CL(clWaitForEvents, 1, &release_event);

  // Cleanup CL
  CHECK_CL(clReleaseMemObject, cmp_buf);
  CHECK_CL(clReleaseMemObject, output);
  CHECK_CL(clReleaseEvent, acquire_event);
  CHECK_CL(clReleaseEvent, release_event);
  for (auto event : cmp_events) {
    CHECK_CL(clReleaseEvent, event);
  }

  // Copy the texture over
  GLsizei width = static_cast<GLsizei>(hdr.width);
  GLsizei height = static_cast<GLsizei>(hdr.height);
  GLsizei dxt_size = (width * height) / 2;
  CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, pbo);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, texID);
  CHECK_GL(glCompressedTexSubImage2D, GL_TEXTURE_2D, 0, 0, 0, hdr.width, hdr.height,
                                      GL_COMPRESSED_RGB_S3TC_DXT1_EXT, dxt_size, 0);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
}


void LoadDDS(const std::unique_ptr<gpu::GPUContext> &ctx,
             GLuint pbo, GLuint texID, const std::string &filePath) {
  // Load in compressed data.
  std::ifstream is(filePath.c_str(), std::ifstream::binary);
  if (!is) {
    assert(!"Error opening DDS texture!");
    return;
  }

  is.seekg(0, is.end);
  size_t length = static_cast<size_t>(is.tellg());
  is.seekg(0, is.beg);

  std::vector<uint8_t> cmp_data(length);
  is.read(reinterpret_cast<char *>(cmp_data.data()), length);
  assert(is);
  is.close();

  gliml::context gliml_ctx;
  gliml_ctx.enable_dxt(true);

  if (!gliml_ctx.load(cmp_data.data(), static_cast<int>(cmp_data.size()))) {
    std::cerr << "Error reading GLIML file!" << std::endl;
    exit(EXIT_FAILURE);
  }

  assert(gliml_ctx.num_faces() == 1);
  assert(gliml_ctx.num_mipmaps(0) == 1);
  assert(gliml_ctx.is_2d());

  // Initialize the texture...
  CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, 0);
  CHECK_GL(glBindTexture, gliml_ctx.texture_target(), texID);
  if (gliml_ctx.is_compressed()) {
    CHECK_GL(glCompressedTexSubImage2D, gliml_ctx.texture_target(), 0, 0, 0,
                                        gliml_ctx.image_width(0, 0),
                                        gliml_ctx.image_height(0, 0),
                                        GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                                        gliml_ctx.image_size(0, 0),
                                        gliml_ctx.image_data(0, 0));
  } else {
    CHECK_GL(glTexSubImage2D, gliml_ctx.texture_target(), 0, 0, 0,
                              gliml_ctx.image_width(0, 0),
                              gliml_ctx.image_height(0, 0),
                              gliml_ctx.image_format(), gliml_ctx.image_type(),
                              gliml_ctx.image_data(0, 0));
  }
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
}

void LoadJPG(const std::unique_ptr<gpu::GPUContext> &ctx,
             GLuint pbo, GLuint texID, const std::string &filePath) {
  int width, height, comp;
  unsigned char *img = stbi_load(filePath.c_str(), &width, &height, &comp, 3);
  CHECK_GL(glTexSubImage2D, GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
  stbi_image_free(img);
}

void LoadCRN(const std::unique_ptr<gpu::GPUContext> &ctx,
             GLuint pbo, GLuint texID, const std::string &filePath) {
  // Load in compressed data.
  std::ifstream is(filePath.c_str(), std::ifstream::binary);
  if (!is) {
    assert(!"Error opening DDS texture!");
    return;
  }

  is.seekg(0, is.end);
  size_t length = static_cast<size_t>(is.tellg());
  is.seekg(0, is.beg);

  std::vector<uint8_t> cmp_data(length);
  is.read(reinterpret_cast<char *>(cmp_data.data()), length);
  assert(is);
  is.close();

  crnd::uint32 crn_data_sz = static_cast<crnd::uint32>(cmp_data.size());

  crnd::crn_texture_info tinfo;
  if (!crnd::crnd_get_texture_info(cmp_data.data(), crn_data_sz, &tinfo)) {
    assert(!"Invalid texture?");
    return;
  }

  crnd::crnd_unpack_context crn_ctx = crnd::crnd_unpack_begin(cmp_data.data(), crn_data_sz);
  if (!crn_ctx) {
    assert(!"Error beginning crn decoding!");
    return;
  }

  const int num_blocks_x = (tinfo.m_width + 3) / 4;
  const int num_blocks_y = (tinfo.m_height + 3) / 4;
  const int num_blocks = num_blocks_x * num_blocks_y;

  std::vector<uint8_t> dxt_vec(num_blocks * 8);
  void *dxt_data = reinterpret_cast<void *>(dxt_vec.data());
  if (!crnd::crnd_unpack_level(crn_ctx, &dxt_data, num_blocks * 8, num_blocks_x * 8, 0)) {
    assert(!"Error decoding crunch texture!");
    return;
  }

  crnd::crnd_unpack_end(crn_ctx);

  // Initialize the texture...
  GLsizei dxt_sz = static_cast<GLsizei>(dxt_vec.size());
  CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, 0);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, texID);
  CHECK_GL(glCompressedTexSubImage2D, GL_TEXTURE_2D, 0, 0, 0, tinfo.m_width, tinfo.m_height,
                                      GL_COMPRESSED_RGB_S3TC_DXT1_EXT, dxt_sz, dxt_data);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
}

int main(int argc, char* argv[])
{
    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    window = glfwCreateWindow(896, 512, "Video", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    fprintf(stdout, "GL Vendor: %s\n", glGetString(GL_VENDOR));
    fprintf(stdout, "GL Renderer: %s\n", glGetString(GL_RENDERER));
    fprintf(stdout, "GL Version: %s\n", glGetString(GL_VERSION));
    fprintf(stdout, "GL Shading Language Version: %s\n",
            glGetString(GL_SHADING_LANGUAGE_VERSION));

#ifndef NDEBUG
    std::string extensionsString(reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS)));
    std::vector<char> extensionVector(extensionsString.begin(), extensionsString.end());
    for (size_t i = 0; i < extensionVector.size(); ++i) {
      if (extensionVector[i] == ',' || extensionVector[i] == ' ') {
        extensionVector[i] = '\0';
      }
    }

    fprintf(stdout, "GL extensions:\n");
    fprintf(stdout, "  %s\n", &(extensionVector[0]));
    for (size_t i = 1; i < extensionVector.size(); i++) {
      if (extensionVector[i] == '\0' && i < extensionVector.size() - 1) {
        fprintf(stdout, "  %s\n", &(extensionVector[i + 1]));
      }
    }
#endif

#ifdef _WIN32
    if (GLEW_OK != glewInit()) {
      std::cerr << "Failed to initialize glew!" << std::endl;
      exit(1);
    }
#endif

    std::unique_ptr<gpu::GPUContext> ctx = gpu::GPUContext::InitializeOpenCL(true);

    glfwSetKeyCallback(window, key_callback);

    GLuint prog = LoadShaders();

    GLint posLoc = glGetAttribLocation(prog, "position");
    assert ( posLoc >= 0 );

    GLint uvLoc = glGetAttribLocation(prog, "texCoord");
    assert ( uvLoc >= 0 );

    GLint texLoc = glGetUniformLocation(prog, "tex");
    assert ( texLoc >= 0 );

    GLuint texID, pbo;
    glGenTextures(1, &texID);

    // Initialize the texture...
    glBindTexture(GL_TEXTURE_2D, texID);
#ifdef TEST_JPG
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, 1792, 1024);
#else
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 1792, 1024);
#endif
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1792 * 1024 / 2, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    static const GLfloat g_FullScreenQuad[] = {
      -1.0f, -1.0f, 0.0f,
      1.0f, -1.0f, 0.0f,
      -1.0f, 1.0f, 0.0f,
      1.0f, 1.0f, 0.0f
    };

    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_FullScreenQuad), g_FullScreenQuad, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    static const GLfloat g_FullScreenUVs[] = {
      0.0f, 1.0f,
      1.0f, 1.0f,
      0.0f, 0.0f,
      1.0f, 0.0f
    };

    GLuint uvBuffer;
    glGenBuffers(1, &uvBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_FullScreenUVs), g_FullScreenUVs, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    static int gFrameNumber = 0;
    static const int kNumFrames = 2000;

    double frame_times[8] = { 0 };
    int frame_time_idx = 0;

    while (!glfwWindowShouldClose(window)) {
      double start_time = glfwGetTime();

      glfwPollEvents();

      if (gPaused) {
        continue;
      }

      assert (glGetError() == GL_NO_ERROR);
      
      int width, height;

      glfwGetFramebufferSize(window, &width, &height);

      glUseProgram(prog);

      glViewport(0, 0, width, height);
      glClear(GL_COLOR_BUFFER_BIT);

      std::ostringstream stream;
#ifdef TEST_GTC
      stream << "../test/dump_gtc/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".gtc";
      LoadGTC(ctx, pbo, texID, stream.str());
#elif defined TEST_CRN
      stream << "../test/dump_crn/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".crn";
      LoadCRN(ctx, pbo, texID, stream.str());
#elif defined TEST_DDS
      stream << "../test/dump_dds/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".dds";
      LoadDDS(ctx, pbo, texID, stream.str());
#elif defined TEST_JPG
      stream << "../test/dump_jpg/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".jpg";
      LoadJPG(ctx, pbo, texID, stream.str());
#endif

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texID);
      glUniform1i(texLoc, 0);

      glEnableVertexAttribArray(posLoc);
      glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
      glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      glEnableVertexAttribArray(uvLoc);
      glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
      glVertexAttribPointer(uvLoc, 2, GL_FLOAT, GL_FALSE, 0, NULL);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      glDisableVertexAttribArray(posLoc);
      glDisableVertexAttribArray(uvLoc);

      glfwSwapBuffers(window);

      double end_time = glfwGetTime();
      frame_times[frame_time_idx] = (end_time - start_time) * 1000.0;
      frame_time_idx = (frame_time_idx + 1) % 8;
      gFrameNumber = (gFrameNumber + 1) % kNumFrames;

      static double last_end_time = 0.0;
      if (end_time - last_end_time > 1) {
        double time = std::accumulate(frame_times, frame_times + 8, 0.0);
        double frame_time = time / 8.0;
        double fps = 1000.0 / frame_time;
        std::cout << "\r";
        std::cout << "FPS: " << fps;
        std::cout.flush();
        last_end_time = end_time;
      }
    }
    std::cout << std::endl;

    // Finish GPU things
    clFlush(ctx->GetDefaultCommandQueue());
    clFinish(ctx->GetDefaultCommandQueue());
    glFlush();
    glFinish();

    glDeleteTextures(1, &texID);
    glDeleteBuffers(1, &pbo);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
    glDeleteProgram(prog);

    // Delete OpenCL crap before we destroy everything else...
    ctx = nullptr;

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

//! [code]
