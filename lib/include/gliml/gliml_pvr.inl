//------------------------------------------------------------------------------
//  gliml_pvr.inl
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
inline bool
is_pvr(const void* data, unsigned int byteSize) {
    if (byteSize > sizeof(pvr_header)) {
        const pvr_header* hdr = (const pvr_header*) data;
        return hdr->version == 0x03525650;
    }
    return false;
}

//------------------------------------------------------------------------------
inline bool
context::load_pvr(const void* data, unsigned int byteSize) {
    GLIML_ASSERT(gliml::is_pvr(data, byteSize));
    this->clear();
    
    const pvr_header* hdr = (const pvr_header*) data;
    const unsigned char* dataBytePtr = (const unsigned char*) hdr;
    dataBytePtr += sizeof(pvr_header) + hdr->metaDataSize;
    
    // texture arrays not yet supported
    if (hdr->numSurfaces > 1) {
        this->errorCode = GLIML_ERROR_TEXTURE_ARRAYS_NOT_SUPPORTED;
        return false;
    }
    
    // cube map?
    bool isCubeMap = false;
    if ((hdr->numFaces != 1) && (hdr->numFaces != 6)) {
        this->errorCode = GLIML_ERROR_INVALID_NUMBER_OF_CUBEMAP_FACES;
        return false;
    }
    if (hdr->numFaces == 6) {
        this->target = GLIML_GL_TEXTURE_CUBE_MAP;
        this->is2D = true;
        this->is3D = false;
        this->numFaces = 6;
        isCubeMap = true;
    }
    // 3D texture?
    else if (hdr->depth > 1) {
        this->target = GLIML_GL_TEXTURE_3D;
        this->is2D = false;
        this->is3D = true;
        this->numFaces = 1;
    }
    // 2D texture?
    else {
        this->target = GLIML_GL_TEXTURE_2D;
        this->is2D = true;
        this->is3D = false;
        this->numFaces = 1;
    }
    
    // image format
    if (!this->pvrtcEnabled) {
        this->errorCode = GLIML_ERROR_PVRTC_NOT_ENABLED;
        return false;
    }
    this->isCompressed = true;
    if (hdr->pixelFormat0 == 1) {
        this->format = this->internalFormat = GLIML_GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
    }
    else if (hdr->pixelFormat0 == 3) {
        this->format = this->internalFormat = GLIML_GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
    }
    else {
        this->errorCode = GLIML_ERROR_INVALID_COMPRESSED_FORMAT;
        return false;
    }
    
    // for each mipmap...
    unsigned int mipIndex;
    for (mipIndex = 0; mipIndex < hdr->mipMapCount; mipIndex++) {
        unsigned int faceIndex;
        for (faceIndex = 0; faceIndex < hdr->numFaces; faceIndex++) {
            face& curFace = this->faces[faceIndex];
            face::mipmap& curMip = curFace.mipmaps[mipIndex];
            
            if (isCubeMap) {
                switch (faceIndex) {
                    case 0:     curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_X; break;
                    case 1:     curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_X; break;
                    case 2:     curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_Y; break;
                    case 3:     curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y; break;
                    case 4:     curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_POSITIVE_Z; break;
                    default:    curFace.target = GLIML_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z; break;
                }
            }
            else {
                curFace.target = this->target;
            }
            curFace.numMipmaps = hdr->mipMapCount;
            
            // mipmap dimensions
            int w = hdr->width >> mipIndex;
            if (w <= 0) w = 1;
            int h = hdr->height >> mipIndex;
            if (h <= 0) h = 1;
            int d = hdr->depth >> mipIndex;
            if (d <= 0) d = 1;
            curMip.width = w;
            curMip.height = h;
            curMip.depth = d;
            
            int blockSize, widthBlocks, heightBlocks, bpp;
            if (GLIML_GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG == this->format) {
                blockSize = 4 * 4;
                widthBlocks = w / 4;
                heightBlocks = h / 4;
                bpp = 4;
            }
            else {
                blockSize = 8 * 4;
                widthBlocks = w / 8;
                heightBlocks = h / 4;
                bpp = 2;
            }
        
            // clamp to minimal block size
            widthBlocks = widthBlocks < 2 ? 2 : widthBlocks;
            heightBlocks = heightBlocks < 2 ? 2 : heightBlocks;
        
            curMip.size = widthBlocks * heightBlocks * ((blockSize * bpp) / 8);
            curMip.data = dataBytePtr;
            dataBytePtr += curMip.size;
        }
    }
    GLIML_ASSERT(dataBytePtr == ((const unsigned char*)data) + byteSize);
    
    // and done
    return true;
}
