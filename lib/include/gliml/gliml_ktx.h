#pragma once
//------------------------------------------------------------------------------
/**
    gliml_ktx.h
    KTX file format support.
    see http://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/
*/

#pragma pack(push,1)
struct ktx_header {
    unsigned char identifier[12];
    unsigned int endianness;
    unsigned int glType;
    unsigned int glTypeSize;
    unsigned int glFormat;
    unsigned int glInternalFormat;
    unsigned int glBaseInternalFormat;
    unsigned int pixelWidth;
    unsigned int pixelHeight;
    unsigned int pixelDepth;
    unsigned int numberOfArrayElements;
    unsigned int numberOfFaces;
    unsigned int numberOfMipmapLevels;
    unsigned int bytesOfKeyValueData;
};
#pragma pack(pop)

#include "gliml_ktx.inl"
