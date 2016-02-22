#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>

#pragma warning( push )
#pragma warning( disable : 4312 )
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( pop )

#include "gpu.h"
#include "config.h"

#ifdef __APPLE__
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

#ifdef _WIN32
#  include <GL/glew.h>
#endif

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
#endif

typedef cl_mem (*clCreateImage2DFunc)(cl_context, cl_mem_flags, const cl_image_format *,
                                      size_t, size_t, size_t, void *, cl_int *);

#ifdef CL_VERSION_1_2
static const clCreateImage2DFunc gCreateImage2DFunc = clCreateImage2D12;
#else
static const clCreateImage2DFunc gCreateImage2DFunc = clCreateImage2D;
#endif

static void RunKernel(const std::unique_ptr<gpu::GPUContext> &ctx, cl_kernel kernel,
                      unsigned char *data, GLuint pbo, int x, int y, int channels) {
  unsigned char *src_data = data;
  if (3 == channels) {
    unsigned char *new_data = (unsigned char *)malloc(x * y * 4);
    int nPixels = x * y;
    for (int i = 0; i < nPixels; ++i) {
      new_data[4*i] = src_data[3*i];
      new_data[4*i + 1] = src_data[3*i + 1];
      new_data[4*i + 2] = src_data[3*i + 2];
      new_data[4*i + 3] = 0xFF;
    }

    src_data = new_data;
  }

  cl_image_format fmt;
  fmt.image_channel_data_type = CL_UNSIGNED_INT8;
  fmt.image_channel_order = CL_RGBA;

  const size_t nBlocksX = ((x + 3) / 4);
  const size_t nBlocksY = ((y + 3) / 4);

  // Upload the data to the GPU...
  cl_int errCreateImage;
  cl_mem input = gCreateImage2DFunc(ctx->GetOpenCLContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, &fmt,
                                    x, y, 4*x, src_data, &errCreateImage);
  CHECK_CL((cl_int), errCreateImage);

  // Finished all OpenGL calls, now we can start making OpenCL calls...
  cl_int errCreateFromGL;
  cl_mem dxt_mem = clCreateFromGLBuffer(ctx->GetOpenCLContext(), CL_MEM_WRITE_ONLY, pbo, &errCreateFromGL);
  CHECK_CL((cl_int), errCreateFromGL);

  // Acquire lock on GL objects...
  cl_command_queue queue = ctx->GetCommandQueue();
  CHECK_CL(clEnqueueAcquireGLObjects, queue, 1, &dxt_mem, 0, NULL, NULL);

  // Set the arguments
  CHECK_CL(clSetKernelArg, kernel, 0, sizeof(input), &input);
  CHECK_CL(clSetKernelArg, kernel, 1, sizeof(dxt_mem), &dxt_mem);

  size_t global_work_size[2] = { nBlocksX, nBlocksY };
  CHECK_CL(clEnqueueNDRangeKernel, queue, kernel, 2, NULL, global_work_size, 0, 0, NULL, NULL);

  // Release the GL objects
  CHECK_CL(clEnqueueReleaseGLObjects, queue, 1, &dxt_mem, 0, NULL, NULL);
  CHECK_CL(clFinish, queue);

  // Release the buffers
  CHECK_CL(clReleaseMemObject, input);
  CHECK_CL(clReleaseMemObject, dxt_mem);

  if (data != src_data) {
    free(src_data);
  }
}

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

void LoadTexture(const std::unique_ptr<gpu::GPUContext> &ctx, cl_kernel kernel, GLuint pbo,
                 GLuint texID, const std::string &filePath) {

  // Load the image data...
  int x = 0, y = 0, channels = 0;
  unsigned char *data = stbi_load(filePath.c_str(), &x, &y, &channels, 0);
  if (!data) {
    fprintf(stderr, "Error loading image: %s\n", filePath.c_str());
    exit(1);
  }

  assert ( x == 960 );
  assert ( y == 540 );
  
  glFinish();

  RunKernel(ctx, kernel, data, pbo, x, y, channels);
  
  // "Bind" the newly created texture : all future texture functions will modify this texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

  glBindTexture(GL_TEXTURE_2D, texID);
  glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, x, y,
    GL_COMPRESSED_RGB_S3TC_DXT1_EXT, x * y / 2, 0);

  GLint query;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &query);
  assert ( query == GL_TRUE );

  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &query);
  assert ( query == x * y / 2 );

  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &query);
  assert ( query == GL_COMPRESSED_RGB_S3TC_DXT1_EXT );

  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  stbi_image_free(data);
}

int main(int argc, char* argv[])
{
    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    window = glfwCreateWindow(960, 540, "Video", NULL, NULL);
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
    cl_kernel kernel = ctx->GetOpenCLKernel(OPENCL_KERNEL_PATH, "compressDXT");

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
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 960, 540);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 960 * 540 / 2, NULL, GL_DYNAMIC_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

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
    static const int kNumFrames = 720;

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
      stream << "../test/dump_jpg/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".jpg";
      LoadTexture(ctx, kernel, pbo, texID, stream.str());

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

      if (frame_time_idx % 8 == 0) {
        double time = std::accumulate(frame_times, frame_times + 8, 0.0);
        double frame_time = time / 8.0;
        double fps = 1000.0 / frame_time;
        std::cout << "\033[20D";
        std::cout << "\033[K";
        std::cout << "FPS: " << fps;
        std::cout.flush();
      }
    }
    std::cout << std::endl;

    // Finish GPU things
    clFlush(ctx->GetCommandQueue());
    clFinish(ctx->GetCommandQueue());
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
