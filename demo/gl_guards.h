#ifndef __GENTC_GL_GUARDS_H__
#define __GENTC_GL_GUARDS_H__

#ifndef NDEBUG
static const char *GetGLErrorString(GLenum e) {
  switch(e) {
    case GL_INVALID_ENUM:                  return "Invalid enum";
    case GL_INVALID_VALUE:                 return "Invalid value";
    case GL_INVALID_OPERATION:             return "Invalid operation";
    case GL_OUT_OF_MEMORY:                 return "Out of memory";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "Invalid Framebuffer Operation";
    default: return "Unknown GL error";
  }
}

#  define PRINT_ERROR(e) \
  const char *errMsg = GetGLErrorString(e);                               \
  if (NULL != errMsg) {                                                 \
    fprintf(stderr, "OpenGL error (%s : %d): %s\n",                     \
            __FILE__, __LINE__, errMsg);                                \
  } else {                                                              \
    fprintf(stderr, "Unknown OpenGL error (%s : %d): 0x%x\n",           \
            __FILE__, __LINE__, e);                                     \
  }                                                                     \
  assert (false);                                                       \

#  define CHECK_GL(fn, ...)                                             \
  do {                                                                  \
    fn(__VA_ARGS__);                                                    \
    GLenum e = glGetError();                                            \
    if (e != GL_NO_ERROR) {                                             \
      PRINT_ERROR(e);                                                   \
    }                                                                   \
  } while(0)

#  define CHECK_GL_AND_RETURN(ty, var, fn, ...)                         \
  ty var = fn(__VA_ARGS__);                                             \
  do {                                                                  \
    GLenum e = glGetError();                                            \
    if(e != GL_NO_ERROR) {                                              \
      PRINT_ERROR(e);                                                   \
    }                                                                   \
  } while(0)
#else
#  define CHECK_GL(fn, ...) do { (void)(fn(__VA_ARGS__)); } while(0)
#  define CHECK_GL_AND_RETURN(ty, var, fn, ...) ty var = fn(__VA_ARGS__)
#endif

#endif  // __GENTC_GL_GUARDS_H__
