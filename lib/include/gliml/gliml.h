#pragma once
//------------------------------------------------------------------------------
/**
    gliml - GL image loader library
    
    gliml main header file. Include this file after the GL header, and 
    optionally define the following macros before the include:
    
    GLIML_ASSERT(x)     - your custom assert implementation, default is assert
    GLIML_NO_DDS        - don't include DDS support
    GLIML_NO_PVR        - don't include PVR support
    GLIML_NO_KTX        - don't include KTX support
*/

#ifndef GLIML_ASSERT
#define GLIML_ASSERT(x) assert(x)
#endif

// see GL headers
#define GLIML_GL_TEXTURE_2D 0x0DE1
#define GLIML_GL_TEXTURE_3D 0x806F
#define GLIML_GL_TEXTURE_CUBE_MAP 0x8513
#define GLIML_GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GLIML_GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GLIML_GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#define GLIML_GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG 0x8C00
#define GLIML_GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG 0x8C01
#define GLIML_GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG 0x8C02
#define GLIML_GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG 0x8C03
#define GLIML_GL_COMPRESSED_RGB8_ETC2 0x9274
#define GLIML_GL_COMPRESSED_SRGB8_ETC2 0x9275
#define GLIML_GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 0x9276
#define GLIML_GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 0x9277
#define GLIML_GL_COMPRESSED_RGBA8_ETC2_EAC 0x9278
#define GLIML_GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC 0x9279
#define GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0x8516
#define GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0x8517
#define GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0x8518
#define GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0x8519
#define GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0x851A

#define GLIML_GL_ALPHA 0x1906
#define GLIML_GL_RGB 0x1907
#define GLIML_GL_RGBA 0x1908
#define GLIML_GL_LUMINANCE 0x1909
#define GLIML_GL_LUMINANCE_ALPHA 0x190A
#define GLIML_GL_BGRA 0x80E1
#define GLIML_GL_BGR 0x80E0

#define GLIML_GL_UNSIGNED_BYTE 0x1401
#define GLIML_GL_UNSIGNED_SHORT_4_4_4_4 0x8033
#define GLIML_GL_UNSIGNED_SHORT_5_5_5_1 0x8034
#define GLIML_GL_UNSIGNED_SHORT_5_6_5   0x8363

//------------------------------------------------------------------------------
namespace gliml {

/// these typedefs are inside the gliml namespace, but GL compatible
typedef unsigned int    GLenum;
typedef unsigned char   GLboolean;
typedef unsigned int    GLbitfield;
typedef void            GLvoid;
typedef signed char     GLbyte;         /* 1-byte signed */
typedef short           GLshort;        /* 2-byte signed */
typedef int             GLint;          /* 4-byte signed */
typedef unsigned char   GLubyte;        /* 1-byte unsigned */
typedef unsigned short  GLushort;       /* 2-byte unsigned */
typedef unsigned int    GLuint;         /* 4-byte unsigned */
typedef int             GLsizei;        /* 4-byte signed */
typedef float           GLfloat;        /* single precision float */
typedef float           GLclampf;       /* single precision float in [0,1] */
typedef double          GLdouble;       /* double precision float */
typedef double          GLclampd;       /* double precision float in [0,1] */
    
// test if image data is in DDS format
#ifndef GLIML_NO_DDS
bool is_dds(const void* data, unsigned int size);
#endif

// test if image data is in PVR format
#ifndef GLIML_NO_PVR
bool is_pvr(const void* data, unsigned int size);
#endif

// test if image data is in KTX format
#ifndef GLIML_NO_KTX
bool is_ktx(const void* data, unsigned int size);
#endif

class context {
public:
    /// default constructor
    context();
    /// destructor
    ~context();
    
    /// enable or disable DXT support (set depending on DXT GL extension)
    void enable_dxt(bool b);
    /// enable or disable PVRTC support (set depending on PVRTC GL extension)
    void enable_pvrtc(bool b);
    /// enable or disable ETC2 support 
    void enable_etc2(bool b);
    /// enable BGRA support
    void enable_bgra(bool b);

    #ifndef GLIML_NO_DDS
    /// load DDS image data into context
    bool load_dds(const void* data, unsigned int size);
    #endif
    #ifndef GLIML_NO_PVR
    /// load PVRTC image data into context
    bool load_pvr(const void* data, unsigned int size);
    #endif
    #ifndef GLIML_NO_KTX
    /// load KTX image data into context
    bool load_ktx(const void* data, unsigned int size);
    #endif
    /// auto-detect format and load
    bool load(const void* data, unsigned int size);
    /// get detailed error code if load returns false
    int error() const;
    /// get the texture target of context
    GLenum texture_target() const;
    /// return true if context contains a compressed texture
    bool is_compressed() const;
    /// return true if context contains a 2D texture
    bool is_2d() const;
    /// return true if context contains a 3D texture
    bool is_3d() const;
    /// get number of faces
    int num_faces() const;
    /// get number of mipmaps in a face
    int num_mipmaps(int face_index) const;
    /// get texture target for texture image function
    GLenum image_target(int face_index) const;
    /// get internal format for texture image function
    GLint image_internal_format() const;
    /// get width for texture image function
    GLsizei image_width(int face_index, int mip_index) const;
    /// get height for texture image function
    GLsizei image_height(int face_index, int mip_index) const;
    /// get depth for texture image function (3D textures)
    GLsizei image_depth(int face_index, int mip_index) const;
    /// get image format for texture image function
    GLenum image_format() const;
    /// get type for texture image function
    GLenum image_type() const;
    /// get image size for compressed texture image function
    GLsizei image_size(int face_index, int mip_index) const;
    /// get pointer to image data
    const GLvoid* image_data(int face_index, int mip_index) const;
    
private:
    /// clear the object
    void clear();

    static const int MaxNumFaces = 6;
    static const int MaxNumMipmaps = 16;
    
    bool dxtEnabled;
    bool pvrtcEnabled;
    bool etc2Enabled;
    bool bgraEnabled;
    int errorCode;
    GLenum target;
    bool isCompressed;
    bool is2D;
    bool is3D;
    GLint internalFormat;
    GLenum format;
    GLenum type;
    int numFaces;
    struct face {
        GLenum target;
        int numMipmaps;
        struct mipmap {
            GLsizei width;
            GLsizei height;
            GLsizei depth;
            GLsizei size;
            const GLvoid* data;
        } mipmaps[MaxNumMipmaps];
    } faces[MaxNumFaces];
};

#define GLIML_SUCCESS (0)
#define GLIML_ERROR_INVALID_COMPRESSED_FORMAT (1)
#define GLIML_ERROR_TEXTURE_ARRAYS_NOT_SUPPORTED (2)
#define GLIML_ERROR_INVALID_NUMBER_OF_CUBEMAP_FACES (3)
#define GLIML_ERROR_UNKNOWN_FILE_FORMAT (4)
#define GLIML_ERROR_DXT_NOT_ENABLED (5)
#define GLIML_ERROR_PVRTC_NOT_ENABLED (6)
#define GLIML_ERROR_ETC2_NOT_ENABLED (7)
#define GLIML_ERROR_ENDIAN_MISMATCH (8)
#define GLIML_ERROR_BGRA_NOT_ENABLED (9)

#include "gliml.inl"
#ifndef GLIML_NO_DDS
#include "gliml_dds.h"
#endif
#ifndef GLIML_NO_PVR
#include "gliml_pvr.h"
#endif
#ifndef GLIML_NO_KTX
#include "gliml_ktx.h"
#endif

} // namespace gliml
//------------------------------------------------------------------------------
