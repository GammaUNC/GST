gliml
=====

(work in progress!)

Minimalistic image loader library for GL projects:

- header only, just include a header and optionally set a few defines to enable or disable features
- focus on compressed formats that can be dumped right into GL texture objects
- doesn't use C++ exceptions, RTTI, STL, strings or file functions
- doesn't call into GL, doesn't include GL headers
- no dynamic memory allocation
- overridable assert macro

File formats: DDS, PVR, KTX

Texture formats: DXT1, DXT3, DXT5, PVR2BPP, PVR4BPP, ETC2

Basic usage:

1. load file data into memory
2. create **gliml::context** object 
3. enable DXT, PVR, ETC2 support depending on GL extensions/version
3. call **gliml::context::load()** to parse the file data into the gliml::context object
4. setup a GL texture using the data in the gliml::context object

See Oryol for real-world example:

https://github.com/floooh/oryol/blob/master/code/Modules/Assets/Gfx/TextureLoader.cc

Sample code (WIP):

```cpp
#define GLIML_ASSERT(x) my_assert(x)
#include "gliml/gliml.h"

void glimlSample() {

    // load file into memory (gliml doesn't have any file I/O functions)
    std::ifstream file("my_texture.dds", std::ios::in | std::ios::binary);
    fil.seekg(0, std::ios::end);
    int size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    // check GL for DXT / PVR support
    bool hasDXTExtension = ...;
    bool hasPVRTCExtension = ...;
    bool hasETC2 = ...;

    // now extract the file data into GL data using gliml
    gliml::context ctx;
    ctx.enable_dxt(hasDXTExtension);   
    ctx.enable_pvrtc(hasPVRTCExtension);
    ctx.enable_etc2(hasETC2)
    if (ctx.load(&buffer.front(), size)) {
        
        // create a GL texture
        GLuint gltx;
        glGenTextures(1, &gltx);
        glBindTexture(ctx.texture_target(), gltx);

        // set desired texture params
        glTexParameteri(ctx.texture_target(), GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(ctx.texture_target(), GL_TEXTURE_WRAP_T, GL_REPEAT);
        if (ctx.is_3d()) {
            glTexParameteri(ctx.texture_target(), GL_TEXTURE_WRAP_R, GL_REPEAT);
        }
        glTexParameteri(ctx.texture_target(), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (ctx.num_mipmaps(0) > 1) {
            glTexParameteri(ctx.texture_target(), GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        else {
            glTexParameteri(ctx.texture_target(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
  
        // for each (cube-map) face...
        for (int face_index = 0; face_index < ctx.num_faces(); face_index++) {
            // for each mip-map level
            for (int mip_index = 0; mip_index < ctx.num_mipmaps(face_index); mip_index++) {
                if (ctx.is_compressed()) {
                    // compressed
                    if (ctx.is_2d()) {
                        // compressed 2D or cube texture
                        glCompressedTexImage2D(ctx.image_target(face_index),
                                               mip_index,
                                               ctx.image_internal_format(),
                                               ctx.image_width(face_index, mip_index),
                                               ctx.image_height(face_index, mip_index),
                                               0,
                                               ctx.image_size(face_index, mip_index),
                                               ctx.image_data(face_index, mip_index));
                    }
                    else {
                        // compressed 3D texture
                        assert(ctx.is_3d());
                        glCompressedTexImage3D(ctx.image_target(face_index),
                                               mip_index,
                                               ctx.image_internal_format(),
                                               ctx.image_width(face_index, mip_index),
                                               ctx.image_height(face_index, mip_index),
                                               ctx.image_depth(face_index, mip_index),
                                               0,
                                               ctx.image_size(face_index, mip_index),
                                               ctx.image_data(face_index, mip_index));
                    }
                }
                else {
                    // uncompressed
                    if (ctx.is_2d()) {
                        // 2D or CUBE texture
                        glTexImage2D(ctx.image_target(face_index),
                                     mip_index,
                                     ctx.image_internal_format(),
                                     ctx.image_width(face_index, mip_index),
                                     ctx.image_height(face_index, mip_index),
                                     0,
                                     ctx.image_format(),
                                     ctx.image_type(),
                                     ctx.image_data(face_index, mip_index));
                    }
                    else {
                        // 3D texture
                        assert(ctx.is_3d());
                        glTexImage3D(ctx.image_target(face_index),
                                     mip,
                                     ctx.image_internal_format(),
                                     ctx.image_width(face_index, mip_index),
                                     ctx.image_height(face_index, mip_index),
                                     ctx.image_depth(face_index, mip_index),
                                     0,
                                     ctx.image_format(),
                                     ctx.image_type(),
                                     ctx.image_data(face_index, mip_index));
                    }
                }
            } // for mip...
        } // for face...
    }
}
```
