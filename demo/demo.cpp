#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "gpu.h"

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

void LoadTexture(GLuint texID, const std::string &filePath) {
  // "Bind" the newly created texture : all future texture functions will modify this texture
  glBindTexture(GL_TEXTURE_2D, texID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

  int x = 0, y = 0, channels = 0;
  unsigned char *data = stbi_load(filePath.c_str(), &x, &y, &channels, 0);
  if (!data) {
    fprintf(stderr, "Error loading image: %s\n", filePath.c_str());
    exit(1);
  }

  switch (channels) {
  case 3:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    break;
  case 4:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    break;
  default:
    fprintf(stderr, "Unable to load image with %d channels!\n", channels);
    exit(1);
    break;
  }

  // Nice trilinear filtering.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

  stbi_image_free(data);
}

void LoadTexture2(GLuint pbo, GLuint texID, const std::string &filePath) {

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

  RunKernel(data, pbo, x, y, channels);
  
  // "Bind" the newly created texture : all future texture functions will modify this texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

  glBindTexture(GL_TEXTURE_2D, texID);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

  glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                         x, y, 0, x * y / 2, 0);

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

int main(void)
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

    InitializeOpenCLKernel();

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
    
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      if (gPaused) {
        continue;
      }

      assert (glGetError() == GL_NO_ERROR);
      
      float ratio;
      int width, height;

      glfwGetFramebufferSize(window, &width, &height);
      ratio = width / (float) height;

      glUseProgram(prog);

      glViewport(0, 0, width, height);
      glClear(GL_COLOR_BUFFER_BIT);

      std::ostringstream stream;
      stream << "../test/dump_jpg/frame";
      for (int i = 1000; i > 0; i /= 10) {
        stream << (((gFrameNumber + 1) / i) % 10);
      }
      stream << ".jpg";
      LoadTexture(texID, stream.str());
      // LoadTexture2(pbo, texID, stream.str());

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

      gFrameNumber = (gFrameNumber + 1) % kNumFrames;
    }

    DestroyOpenCLKernel();

    glDeleteTextures(1, &texID);
    glDeleteBuffers(1, &pbo);

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

//! [code]
