#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include "win/dirent.h"
#else // _MSC_VER
#include <dirent.h>
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4312 )
#endif  // _MSC_VER
#include "stb_image.h"
#ifdef _MSC_VER
#pragma warning( pop )
#endif  // _MSC_VER

#include "gpu.h"
#include "codec.h"

#include "ctpl/ctpl_stl.h"

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

static const int kWindowWidth = 512;
static const int kWindowHeight = 512;
static const float kAspect = static_cast<float>(kWindowWidth) / static_cast<float>(kWindowHeight);

static void GetNumSubWindows2D(size_t count, size_t aspect, size_t *nx, size_t *ny) {
  *nx = 1;
  *ny = count;
  while ((*ny % 2) == 0 && 2 * aspect * *nx < *ny) {
    *ny /= 2;
    *nx *= 2;
  }
}

static void GetCropWindow(size_t num, size_t count, float aspect,
                          float *x1, float *x2, float *y1, float *y2) {
  size_t nx = 0;
  size_t ny = 0;

  if (aspect < 1.f) {
    size_t inva = static_cast<size_t>(1.f / aspect);
    GetNumSubWindows2D(count, inva, &nx, &ny);
  } else {
    GetNumSubWindows2D(count, static_cast<size_t>(aspect), &ny, &nx);
  }

  // Compute x and y pixel sample range for sub window
  size_t xo = num % nx;
  size_t yo = num / nx;

  *x1 = static_cast<float>(xo) / static_cast<float>(nx);
  *x2 = static_cast<float>(xo + 1) / static_cast<float>(nx);

  *y1 = static_cast<float>(yo) / static_cast<float>(ny);
  *y2 = static_cast<float>(yo + 1) / static_cast<float>(ny);
}

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
  "uniform sampler2D tex;\n"
  ""
  "void main() {\n"
  "  gl_FragColor = vec4(texture2D(tex, uv).rgb, 1);\n"
  "}\n";

GLuint LoadShaders() {
  CHECK_GL_AND_RETURN(GLuint, vertShdrID, glCreateShader, GL_VERTEX_SHADER);
  CHECK_GL_AND_RETURN(GLuint, fragShdrID, glCreateShader, GL_FRAGMENT_SHADER);

  CHECK_GL(glShaderSource, vertShdrID, 1, &kVertexProg , NULL);
  CHECK_GL(glCompileShader, vertShdrID);

  int result, logLength;
  
  CHECK_GL(glGetShaderiv, vertShdrID, GL_COMPILE_STATUS, &result);
  if (result != GL_TRUE) {
    CHECK_GL(glGetShaderiv, vertShdrID, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> VertexShaderErrorMessage(logLength);
    CHECK_GL(glGetShaderInfoLog, vertShdrID, logLength, NULL, &VertexShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
    fprintf(stdout, "Vertex shader compilation failed!\n");
    exit(1);
  }

  // Compile Fragment Shader
  CHECK_GL(glShaderSource, fragShdrID, 1, &kFragProg, NULL);
  CHECK_GL(glCompileShader, fragShdrID);

  // Check Fragment Shader
  CHECK_GL(glGetShaderiv, fragShdrID, GL_COMPILE_STATUS, &result);
  if (result != GL_TRUE) {
    CHECK_GL(glGetShaderiv, fragShdrID, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> FragmentShaderErrorMessage(logLength);
    CHECK_GL(glGetShaderInfoLog, fragShdrID, logLength, NULL, &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "Fragment shader compilation failed!\n");
    exit(1);
  }

  // Link the program
  CHECK_GL_AND_RETURN(GLuint, prog, glCreateProgram);
  CHECK_GL(glAttachShader, prog, vertShdrID);
  CHECK_GL(glAttachShader, prog, fragShdrID);
  CHECK_GL(glLinkProgram, prog);

  // Check the program
  CHECK_GL(glGetProgramiv, prog, GL_LINK_STATUS, &result);
  if (result != GL_TRUE) {
    CHECK_GL(glGetProgramiv, prog, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> ProgramErrorMessage( std::max(logLength, int(1)) );
    CHECK_GL(glGetProgramInfoLog, prog, logLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
  }

  CHECK_GL(glDeleteShader, vertShdrID);
  CHECK_GL(glDeleteShader, fragShdrID);

  return prog;
}

static std::vector<uint8_t> LoadFile(const std::string &filePath) {
  // Load in compressed data.
  std::ifstream is (filePath.c_str(), std::ifstream::binary);
  if (!is) {
    std::cerr << "Error opening GenTC texture: " << filePath << std::endl;
    exit(EXIT_FAILURE);
  }

  is.seekg(0, is.end);
  size_t length = static_cast<size_t>(is.tellg());
  is.seekg(0, is.beg);

  std::vector<uint8_t> cmp_data(length);
  is.read(reinterpret_cast<char *>(cmp_data.data()), length);
  assert(is);
  is.close();
  return std::move(cmp_data);
}

class Texture {
private:
  GLuint _id;
  GLuint _vtx_buffer;
  GLuint _uv_buffer;

  GLint _tex_loc;
  GLint _pos_loc;
  GLint _uv_loc;

public:
  Texture(GLint texLoc, GLint posLoc, GLint uvLoc,
          GLuint texID, size_t num, size_t count)
    : _id(texID), _tex_loc(texLoc), _pos_loc(posLoc), _uv_loc(uvLoc) {
    float x1, x2, y1, y2;
    GetCropWindow(num, count, kAspect, &x1, &x2, &y1, &y2);

    const GLfloat g_FullScreenQuad[] = {
      x1 * 2.0f - 1.0f, y1 * 2.0f - 1.0f, 0.0f,
      x2 * 2.0f - 1.0f, y1 * 2.0f - 1.0f, 0.0f,
      x1 * 2.0f - 1.0f, y2 * 2.0f - 1.0f, 0.0f,
      x2 * 2.0f - 1.0f, y2 * 2.0f - 1.0f, 0.0f
    };

    CHECK_GL(glGenBuffers, 1, &_vtx_buffer);

    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, _vtx_buffer);
    CHECK_GL(glBufferData, GL_ARRAY_BUFFER, sizeof(g_FullScreenQuad), g_FullScreenQuad, GL_STATIC_DRAW);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

    static const GLfloat g_FullScreenUVs[] = {
      0.0f, 1.0f, 1.0f, 1.0f,
      0.0f, 0.0f, 1.0f, 0.0f
    };

    CHECK_GL(glGenBuffers, 1, &_uv_buffer);

    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, _uv_buffer);
    CHECK_GL(glBufferData, GL_ARRAY_BUFFER, sizeof(g_FullScreenUVs), g_FullScreenUVs, GL_STATIC_DRAW);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);
  }

  ~Texture() {
    CHECK_GL(glDeleteTextures, 1, &_id);
    CHECK_GL(glDeleteBuffers, 1, &_vtx_buffer);
    CHECK_GL(glDeleteBuffers, 1, &_uv_buffer);
  }

  void Draw() const {
    CHECK_GL(glActiveTexture, GL_TEXTURE0);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, _id);
    CHECK_GL(glUniform1i, _tex_loc, 0);

    CHECK_GL(glEnableVertexAttribArray, _pos_loc);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, _vtx_buffer);
    CHECK_GL(glVertexAttribPointer, _pos_loc, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

    CHECK_GL(glEnableVertexAttribArray, _uv_loc);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, _uv_buffer);
    CHECK_GL(glVertexAttribPointer, _uv_loc, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    CHECK_GL(glBindBuffer, GL_ARRAY_BUFFER, 0);

    CHECK_GL(glDrawArrays, GL_TRIANGLE_STRIP, 0, 4);

    CHECK_GL(glDisableVertexAttribArray, _pos_loc);
    CHECK_GL(glDisableVertexAttribArray, _uv_loc);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
  }
};

// Requesting thread sets sz to a size, dst to a valid pointer, and waits on the cv
// Producing thread sets sz to zero, places a requested pbo in dst, and then notifies the cv.
struct PBORequest {
  size_t sz;
  GLuint pbo;
  cl_mem dst_buf;
  cl_event acquire_event;

  void SetupPBO() {
    assert(sz > 0);
    CHECK_GL(glGenBuffers, 1, &pbo);
    CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, pbo);
    CHECK_GL(glBufferData, GL_PIXEL_UNPACK_BUFFER, sz, NULL, GL_DYNAMIC_DRAW);
    CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, 0);    
  }

  void CreateCLBuffer(const std::unique_ptr<gpu::GPUContext> &ctx) {
    // Create an OpenGL handle to our pbo
    // !SPEED! We don't need to recreate this every time....
    cl_int errCreateBuffer;
    dst_buf = clCreateFromGLBuffer(ctx->GetOpenCLContext(), CL_MEM_READ_WRITE, pbo,
                                   &errCreateBuffer);
    CHECK_CL((cl_int), errCreateBuffer);
  }
};

class AsyncTexRequest {
 public:
  virtual ~AsyncTexRequest() { }
  bool Run(std::function<void()> *ret) {
    if (_fns.empty()) {
      return false;
    }

    *ret = _fns.front();
    _fns.pop();
    return true;
  }

  void QueueWork(std::function<void()> fn) {
    _fns.push(fn);
  }

  virtual GLuint TextureHandle() const = 0;
  virtual bool NeedsPBO(PBORequest **ret) = 0;
  virtual void LoadTexture() const = 0;
 private:
  std::queue<std::function<void()> > _fns;
};

class AsyncGenTCReq : public AsyncTexRequest {
 public:
  AsyncGenTCReq(const std::unique_ptr<gpu::GPUContext> &ctx, GLuint id)
    : AsyncTexRequest()
    , _ctx(ctx)
    , _texID(id)
  { }
  virtual ~AsyncGenTCReq() { }

  virtual GLuint TextureHandle() const override { return _texID; }

  virtual bool NeedsPBO(PBORequest **ret) override {
    *ret = &_pbo;
    return true;
  }

  virtual void LoadTexture() const override {
    // Initialize the texture...
    CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, _pbo.pbo);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, _texID);
    CHECK_GL(glCompressedTexImage2D, GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
             static_cast<GLsizei>(_hdr.width), static_cast<GLsizei>(_hdr.height), 0,
             static_cast<GLsizei>(_pbo.sz), 0);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    CHECK_GL(glBindBuffer, GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);

    // We don't need the PBO anymore
    CHECK_GL(glDeleteBuffers, 1, &_pbo.pbo);
    CHECK_CL(clReleaseMemObject, _pbo.dst_buf);
    CHECK_CL(clReleaseMemObject, _cmp_buf);
  }

  virtual void Preload(const std::vector<uint8_t> &cmp_data) {
    _hdr.LoadFrom(cmp_data.data());
    _pbo.sz = (_hdr.width * _hdr.height) / 2;

    // Create the data for OpenCL
    cl_int errCreateBuffer;
    static const size_t kHeaderSz = sizeof(_hdr);
    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
    _cmp_buf = clCreateBuffer(_ctx->GetOpenCLContext(), flags, cmp_data.size() - kHeaderSz,
                              const_cast<uint8_t *>(cmp_data.data() + kHeaderSz),
                              &errCreateBuffer);
    CHECK_CL((cl_int), errCreateBuffer);
  }

  virtual std::vector<cl_event> QueueDXT() {
    // Load it
    return std::move(GenTC::LoadCompressedDXT(_ctx, _hdr, _cmp_buf, _pbo.dst_buf, &_pbo.acquire_event));
  }

 private:
  const std::unique_ptr<gpu::GPUContext> &_ctx;
  GLuint _texID;

  GenTC::GenTCHeader _hdr;
  cl_mem _cmp_buf;
  PBORequest _pbo;
};

class AsyncGenericReq : public AsyncTexRequest {
 public:
  AsyncGenericReq(GLuint id)
    : AsyncTexRequest()
    , _texID(id)
  { }

  virtual GLuint TextureHandle() const override { return _texID; }
  virtual void LoadTexture() const override {
    assert(this->_n == 3);

    // Initialize the texture...
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, _texID);
    CHECK_GL(glTexImage2D, GL_TEXTURE_2D, 0, GL_RGB8, _x, _y, 0, GL_RGB, GL_UNSIGNED_BYTE, this->_data);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    CHECK_GL(glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    CHECK_GL(glBindTexture, GL_TEXTURE_2D, 0);
    stbi_image_free(this->_data);
  }

  virtual bool NeedsPBO(PBORequest **ret) override { return false; }
  virtual void LoadFile(const std::string &filename) {
    this->_data = stbi_load(filename.c_str(), &_x, &_y, &_n, 3);
    assert(_n == 3);  // Only load RGB textures...
    _n = 3;
  }

 private:
  GLuint _texID;
  int _x, _y, _n;
  unsigned char *_data;
};

std::vector<std::unique_ptr<Texture> > LoadTextures(const std::unique_ptr<gpu::GPUContext> &ctx,
                                                    GLint texLoc, GLint posLoc, GLint uvLoc,
                                                    bool async, const char *dirname) {
  // Load textures!
  DIR *dir = opendir(dirname);
  if (!dir) {
    std::cerr << "Error opening directory " << dirname << std::endl;
    exit(EXIT_FAILURE);
  }

  // Collect the actual filenames
  std::vector<std::string> filenames;
  struct dirent *entry = NULL;
  while ((entry = readdir(dir)) != NULL) {
    // A few exceptions...
    if (strlen(entry->d_name) == 1 && strncmp(entry->d_name, ".", 1) == 0) continue;
    if (strlen(entry->d_name) == 2 && strncmp(entry->d_name, "..", 2) == 0) continue;
    filenames.push_back(std::string(dirname) + std::string("/") + std::string(entry->d_name));
  }
  closedir(dir);

  // We'll have as many textures as we have filenames
  std::vector<std::unique_ptr<Texture> > textures;
  textures.reserve(filenames.size());

  // Load up a bunch of requests
  std::vector<std::unique_ptr<AsyncTexRequest> > reqs;
  reqs.reserve(textures.size());

  // Events that we need to wait on before we release GL objects...
  std::mutex dxt_events_mutex;
  std::vector<cl_event> dxt_events;

  for (size_t i = 0; i < filenames.size(); ++i) {

#ifndef NDEBUG
    std::cout << "Loading texture: " << filenames[i] << std::endl;
#endif

    GLuint texID;
    CHECK_GL(glGenTextures, 1, &texID);

    size_t len = filenames[i].length();
    assert(len >= 4);
    if (strncmp(filenames[i].c_str() + len - 4, ".gtc", 4) == 0) {
      reqs.push_back(std::unique_ptr<AsyncTexRequest>(new AsyncGenTCReq(ctx, texID)));
      AsyncTexRequest *req = reqs.back().get();
      const std::string &fname = filenames[i];

      // Pre-pbo
      req->QueueWork([&fname, &ctx, req]() {
        std::vector<uint8_t> cmp_data = std::move(LoadFile(fname));
        reinterpret_cast<AsyncGenTCReq *>(req)->Preload(cmp_data);
      });

      // Post-pbo
      req->QueueWork([&fname, &ctx, &dxt_events, &dxt_events_mutex, req]() {
        std::vector<cl_event> es = reinterpret_cast<AsyncGenTCReq *>(req)->QueueDXT();
        std::unique_lock<std::mutex> lock(dxt_events_mutex);
        dxt_events.insert(dxt_events.end(), es.begin(), es.end());
      });
    } else {
      reqs.push_back(std::unique_ptr<AsyncTexRequest>(new AsyncGenericReq(texID)));
      AsyncTexRequest *req = reqs.back().get();
      const std::string &fname = filenames[i];
      req->QueueWork([req, &fname]() {
        reinterpret_cast<AsyncGenericReq *>(req)->LoadFile(fname);
      });
    }

    textures.push_back(std::unique_ptr<Texture>(
                       new Texture(texLoc, posLoc, uvLoc, texID, i, filenames.size())));
  }

  // Loop until all requests are dun:
  //   - Run all of the requests that need it.
  //   - Collect GL/CL interop resources for each request
  std::vector<PBORequest *> pbo_reqs;
  std::vector<cl_mem> pbos;
  pbo_reqs.reserve(textures.size());
  pbos.reserve(textures.size());

  std::atomic_int num_loaded; num_loaded = 0;
  std::condition_variable loading_cv;

  const unsigned kTotalNumThreads = async ? std::thread::hardware_concurrency() : 1;
  ctpl::thread_pool pool(kTotalNumThreads);

  bool acquire = true;
  cl_event acquire_event;
  cl_event release_event;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double idle_time = 0.0;
  double interop_time = 0.0;

  for(int pass = 0; pass < 2; ++pass) {
    // Queue up work
    std::mutex m;
    std::condition_variable done;
    std::atomic_int num_finished(0);
    int num_running = 0;
    for (auto &req : reqs) {
      std::function<void()> fn;
      if (req->Run(&fn)) {
        pool.push([fn, &num_finished, &done, &m](int) {
          fn();
          std::unique_lock<std::mutex> lock(m);
          num_finished++;
          done.notify_one();
        });
        num_running++;
      }
    }

    // If the work is more or less done, then we don't need to
    // keep going...
    if (0 == num_running) {
      break;
    }

    // Wait for the work to finish
    start = std::chrono::high_resolution_clock::now();
    {
      std::unique_lock<std::mutex> lock(m);
      done.wait(lock, [&]() { return num_running == num_finished; });
    }
    end = std::chrono::high_resolution_clock::now();
    idle_time += std::chrono::duration<double>(end-start).count();

    if (acquire) {
      // Collect all of our pbo requests if we need to acquire them
      for (auto &req : reqs) {
        PBORequest *pbo_req;
        if (req->NeedsPBO(&pbo_req)) {
          assert(pbo_req->sz == 131072);
          pbo_reqs.push_back(pbo_req);
          pbo_req->SetupPBO();
        }
      }

      if (pbo_reqs.size() == 0) {
        continue;
      }

      // Wait for the GPU to finish
      CHECK_GL(glFlush);
      CHECK_GL(glFinish);

      // Now, create CL buffers for all of our pbos
      for (auto req : pbo_reqs) {
        req->CreateCLBuffer(ctx);
        pbos.push_back(req->dst_buf);
      }

      // Acquire all of the CL buffers at once...
      start = std::chrono::high_resolution_clock::now();
      CHECK_CL(clEnqueueAcquireGLObjects, ctx->GetCommandQueue(), pbos.size(),
                                          pbos.data(), 0, NULL, &acquire_event);
      end = std::chrono::high_resolution_clock::now();
      interop_time += std::chrono::duration<double>(end-start).count();
      std::cout << "Loading textures acquire GL time: " << idle_time << "s" << std::endl;

      // Set the event for all the requests so that they know
      // that it's ok to use it...
      for (auto req : pbo_reqs) {
        req->acquire_event = acquire_event;
      }

      // We're done, let's do the rest of the work...
      acquire = false;
    } else {
      start = std::chrono::high_resolution_clock::now();
      CHECK_CL(clEnqueueReleaseGLObjects, ctx->GetCommandQueue(),
                                          pbos.size(), pbos.data(),
                                          dxt_events.size(), dxt_events.data(), &release_event);
      end = std::chrono::high_resolution_clock::now();
      interop_time += std::chrono::duration<double>(end-start).count();
    }
  }

  // Wait for the OpenCL event to finish...
  if (dxt_events.size() > 0) {
    start = std::chrono::high_resolution_clock::now();
    CHECK_CL(clWaitForEvents, 1, &release_event);
    end = std::chrono::high_resolution_clock::now();
    idle_time += std::chrono::duration<double>(end-start).count();
  }

  CHECK_CL(clReleaseEvent, acquire_event);
  CHECK_CL(clReleaseEvent, release_event);

  // I think we're done now...
  for (auto &req : reqs) {
    req->LoadTexture();
  }

  std::cout << "Loading textures idle time: " << idle_time << "s" << std::endl;
  std::cout << "Loading textures interop time: " << interop_time << "s" << std::endl;

  return std::move(textures);
}

int main(int argc, char* argv[] ) {
    if (argc <= 1) {
      std::cerr << "Usage: " << argv[0] << " [-p] <directory>" << std::endl;
      exit(EXIT_FAILURE);
    }

    const char *dirname = argv[1];
    bool profiling = false;
    if (argc == 3 && strncmp(argv[1], "-p", 2) == 0) {
      profiling = true;
      dirname = argv[2];
    }

    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
      exit(EXIT_FAILURE);

    GLFWwindow* window = glfwCreateWindow(512, 512, "Photos", NULL, NULL);
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
    CHECK_GL_AND_RETURN(GLint, posLoc, glGetAttribLocation, prog, "position");
    CHECK_GL_AND_RETURN(GLint, uvLoc, glGetAttribLocation, prog, "texCoord");
    CHECK_GL_AND_RETURN(GLint, texLoc, glGetUniformLocation, prog, "tex");

    assert ( posLoc >= 0 );
    assert ( uvLoc >= 0 );
    assert ( texLoc >= 0 );

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::unique_ptr<Texture> > texs =
      LoadTextures(ctx, texLoc, posLoc, uvLoc, true, dirname);
      // LoadTextures(ctx, texLoc, posLoc, uvLoc, false, dirname);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Loaded " << texs.size() << " texture"
              << ((texs.size() == 1) ? "" : "s") << " in "
              << std::chrono::duration<double>(end-start).count() << "s"
              << std::endl;
    
    CHECK_GL(glPixelStorei, GL_UNPACK_ALIGNMENT, 1);

    static const int kFrameTimeHistorySz = 8;
    double frame_times[kFrameTimeHistorySz] = { 0 };
    int frame_time_idx = 0;
    double elapsed_since_refresh = 0.0;

    while (!glfwWindowShouldClose(window)) {
      double start_time = glfwGetTime();
      glfwPollEvents();

      if (gPaused) {
        continue;
      }

      int width, height;
      glfwGetFramebufferSize(window, &width, &height);
      CHECK_GL(glViewport, 0, 0, width, height);

      CHECK_GL(glClear, GL_COLOR_BUFFER_BIT);
      CHECK_GL(glUseProgram, prog);
      for (const auto &tex : texs) {
        tex->Draw();
      }

      glfwSwapBuffers(window);
      if (profiling) {
        glfwSetWindowShouldClose(window, GL_TRUE);
      }

      double end_time = glfwGetTime();
      frame_times[frame_time_idx] = (end_time - start_time) * 1000.0;
      frame_time_idx = (frame_time_idx + 1) % kFrameTimeHistorySz;
      elapsed_since_refresh += (end_time - start_time);

      if (elapsed_since_refresh > 1.0) {
        double time = std::accumulate(frame_times, frame_times + kFrameTimeHistorySz, 0.0);
        double frame_time = time / static_cast<double>(kFrameTimeHistorySz);
        double fps = 1000.0 / frame_time;

        std::cout << '\r';
        std::cout << "FPS: " << fps;
        std::cout.flush();
        elapsed_since_refresh = 0.0;
      }
    }
    std::cout << std::endl;

    // Finish GPU things
    clFlush(ctx->GetCommandQueue());
    clFinish(ctx->GetCommandQueue());
    CHECK_GL(glFlush);
    CHECK_GL(glFinish);

    CHECK_GL(glDeleteProgram, prog);

    // Delete OpenCL crap before we destroy everything else...
    ctx = nullptr;

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

//! [code]
