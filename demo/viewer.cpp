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
#include "gliml/gliml.h"

#define CRND_HEADER_FILE_ONLY
#include "crn_decomp.h"

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

  glShaderSource(vertShdrID, 1, &kVertexProg, NULL);
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
    std::vector<char> ProgramErrorMessage(std::max(logLength, int(1)));
    glGetProgramInfoLog(prog, logLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
  }

  glDeleteShader(vertShdrID);
  glDeleteShader(fragShdrID);

  return prog;
}

void LoadGTC(const std::unique_ptr<gpu::GPUContext> &ctx, bool has_dxt,
             GLuint texID, const std::string &filePath) {
  GenTC::GenTCHeader hdr;
  // Load in compressed data.
  std::ifstream is(filePath.c_str(), std::ifstream::binary);
  if (!is) {
    assert(!"Error opening GenTC texture!");
    return;
  }

  is.seekg(0, is.end);
  size_t length = static_cast<size_t>(is.tellg());
  is.seekg(0, is.beg);

  static const size_t kHeaderSz = sizeof(hdr);
  const size_t mem_sz = length - kHeaderSz;

  is.read(reinterpret_cast<char *>(&hdr), kHeaderSz);

  std::vector<uint8_t> cmp_data(mem_sz + 512);
  is.read(reinterpret_cast<char *>(cmp_data.data()) + 512, mem_sz);
  assert(is);
  assert(is.tellg() == static_cast<std::streamoff>(length));
  is.close();

  const cl_uint num_blocks = hdr.height * hdr.width / 16;
  cl_uint *offsets = reinterpret_cast<cl_uint *>(cmp_data.data());
  cl_uint output_offset = 0;
  offsets[0] = output_offset; output_offset += 2 * num_blocks; // Y planes
  offsets[1] = output_offset; output_offset += 4 * num_blocks; // Chroma planes
  offsets[2] = output_offset; output_offset += static_cast<cl_uint>(hdr.palette_bytes); // Palette
  offsets[3] = output_offset; output_offset += num_blocks; // Indices

  cl_uint input_offset = 0;
  offsets[4] = input_offset; input_offset += hdr.y_cmp_sz;
  offsets[5] = input_offset; input_offset += hdr.chroma_cmp_sz;
  offsets[6] = input_offset; input_offset += hdr.palette_sz;
  offsets[7] = input_offset; input_offset += hdr.indices_sz;

  GLsizei width = static_cast<GLsizei>(hdr.width);
  GLsizei height = static_cast<GLsizei>(hdr.height);
  GLsizei dxt_size = (width * height) / 2;

  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  if (has_dxt) {
    glBufferData(GL_PIXEL_UNPACK_BUFFER, dxt_size, NULL, GL_STREAM_COPY);
  } else {
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, NULL, GL_STREAM_COPY);
  }

  // Create the data for OpenCL
  cl_int errCreateBuffer;
  cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  cl_mem cmp_buf = clCreateBuffer(ctx->GetOpenCLContext(), flags, cmp_data.size(),
                                  const_cast<uint8_t *>(cmp_data.data()), &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  // Create an OpenGL handle to our pbo
  // !SPEED! We don't need to recreate this every time....
  cl_mem output = clCreateFromGLBuffer(ctx->GetOpenCLContext(), CL_MEM_READ_WRITE, pbo,
                                       &errCreateBuffer);
  CHECK_CL((cl_int), errCreateBuffer);

  cl_command_queue queue = ctx->GetNextQueue();

  // Acquire the PBO
  cl_event acquire_event;
  CHECK_CL(clEnqueueAcquireGLObjects, queue, 1, &output, 0, NULL, &acquire_event);

  // Load it
  cl_event cmp_event;
  if (has_dxt) {
    cmp_event = GenTC::LoadCompressedDXT(ctx, hdr, queue, cmp_buf, output, 1, &acquire_event);
  }
  else {
    cmp_event = GenTC::LoadRGB(ctx, hdr, queue, cmp_buf, output, 1, &acquire_event);
  }

  // Release the PBO
  cl_event release_event;
  CHECK_CL(clEnqueueReleaseGLObjects, queue, 1, &output, 1, &cmp_event, &release_event);

  CHECK_CL(clFlush, ctx->GetDefaultCommandQueue());

  // Wait on the release
  CHECK_CL(clWaitForEvents, 1, &release_event);

  // Cleanup CL
  CHECK_CL(clReleaseMemObject, cmp_buf);
  CHECK_CL(clReleaseMemObject, output);
  CHECK_CL(clReleaseEvent, acquire_event);
  CHECK_CL(clReleaseEvent, release_event);
  CHECK_CL(clReleaseEvent, cmp_event);

  // Copy the texture over
  CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, pbo);
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, texID);
  if (has_dxt) {
    CHECK_GL(glCompressedTexImage2D, GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                                     width, height, 0, dxt_size, 0);
  } else {
    CHECK_GL(glTexImage2D, GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
  }
  CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
  CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, 0);
  CHECK_GL(glDeleteBuffers, 1, &pbo);
}

int main(int argc, char* argv[])
{
  GLFWwindow* window;

  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  window = glfwCreateWindow(896, 512, "Viewer", NULL, NULL);
  if (!window) {
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

  std::string extensionsString(reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS)));
  std::vector<char> extensionVector(extensionsString.begin(), extensionsString.end());
  for (size_t i = 0; i < extensionVector.size(); ++i) {
    if (extensionVector[i] == ',' || extensionVector[i] == ' ') {
      extensionVector[i] = '\0';
    }
  }

  bool has_dxt = false;
#ifndef NDEBUG
  fprintf(stdout, "GL extensions:\n");
  fprintf(stdout, "  %s\n", &(extensionVector[0]));
#endif
  for (size_t i = 1; i < extensionVector.size(); i++) {
    if (extensionVector[i] == '\0' && i < extensionVector.size() - 1) {
#ifndef NDEBUG
      fprintf(stdout, "  %s\n", &(extensionVector[i + 1]));
#endif
      if (strstr(&extensionVector[i + 1], "GL_EXT_texture_compression_s3tc") != NULL) {
        has_dxt = true;
      }
    }
  }

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
  assert(posLoc >= 0);

  GLint uvLoc = glGetAttribLocation(prog, "texCoord");
  assert(uvLoc >= 0);

  GLint texLoc = glGetUniformLocation(prog, "tex");
  assert(texLoc >= 0);

  GLuint texID;
  glGenTextures(1, &texID);
  LoadGTC(ctx, has_dxt, texID, std::string(argv[1]));
  std::cout << "Loaded " << argv[1] << std::endl;

  CHECK_GL(glBindTexture, GL_TEXTURE_2D, texID);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

  GLint width, height;
  CHECK_GL(glGetTexLevelParameteriv, GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
  CHECK_GL(glGetTexLevelParameteriv, GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
  glfwSetWindowSize(window, width, height);

  static const GLfloat g_FullScreenQuad[] = {
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f
  };

  GLuint vertexBuffer;
  CHECK_GL(glGenBuffers, 1, &vertexBuffer);

  CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, vertexBuffer);
  CHECK_GL(glBufferData, GL_ARRAY_BUFFER, sizeof(g_FullScreenQuad), g_FullScreenQuad, GL_STATIC_DRAW);
  CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

  static const GLfloat g_FullScreenUVs[] = {
    0.0f, 1.0f,
    1.0f, 1.0f,
    0.0f, 0.0f,
    1.0f, 0.0f
  };

  GLuint uvBuffer;
  CHECK_GL(glGenBuffers, 1, &uvBuffer);

  CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, uvBuffer);
  CHECK_GL(glBufferData, GL_ARRAY_BUFFER, sizeof(g_FullScreenUVs), g_FullScreenUVs, GL_STATIC_DRAW);
  CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

  CHECK_GL(glPixelStorei, GL_UNPACK_ALIGNMENT, 1);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    if (gPaused) {
      continue;
    }

    assert(glGetError() == GL_NO_ERROR);

    int width, height;

    glfwGetFramebufferSize(window, &width, &height);

    CHECK_GL(glUseProgram, prog);

    CHECK_GL(glViewport, 0, 0, width, height);
    CHECK_GL(glClear, GL_COLOR_BUFFER_BIT);

    CHECK_GL(glActiveTexture, GL_TEXTURE0);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, texID);
    CHECK_GL(glUniform1i, texLoc, 0);

    CHECK_GL(glEnableVertexAttribArray, posLoc);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, vertexBuffer);
    CHECK_GL(glVertexAttribPointer, posLoc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

    CHECK_GL(glEnableVertexAttribArray, uvLoc);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, uvBuffer);
    CHECK_GL(glVertexAttribPointer, uvLoc, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

    CHECK_GL(glDrawArrays, GL_TRIANGLE_STRIP, 0, 4);

    CHECK_GL(glDisableVertexAttribArray, posLoc);
    CHECK_GL(glDisableVertexAttribArray, uvLoc);

    glfwSwapBuffers(window);
  }

  // Finish GPU things
  clFlush(ctx->GetDefaultCommandQueue());
  clFinish(ctx->GetDefaultCommandQueue());
  CHECK_GL(glFlush);
  CHECK_GL(glFinish);

  CHECK_GL(glDeleteTextures, 1, &texID);
  CHECK_GL(glDeleteBuffers, 1, &vertexBuffer);
  CHECK_GL(glDeleteBuffers, 1, &uvBuffer);
  CHECK_GL(glDeleteProgram, prog);

  // Delete OpenCL crap before we destroy everything else...
  ctx = nullptr;

  glfwDestroyWindow(window);

  glfwTerminate();
  exit(EXIT_SUCCESS);
}

//! [code]
