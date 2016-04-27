#pragma once
//------------------------------------------------------------------------------
/**
    gliml_pvr.h
    PVRTC specific code of gliml.
    See the PowerVR Tools package at http://community.imgtec.com/developers/powervr/tools/
    for details
*/

#pragma pack(push, 1)
struct pvr_header {
    unsigned int version;
    unsigned int flags;
    unsigned int pixelFormat0;
    unsigned int pixelFormat1;
    unsigned int colorSpace;
    unsigned int channelType;
    unsigned int height;
    unsigned int width;
    unsigned int depth;
    unsigned int numSurfaces;
    unsigned int numFaces;
    unsigned int mipMapCount;
    unsigned int metaDataSize;
};
#pragma pack(pop)

#include "gliml_pvr.inl"
