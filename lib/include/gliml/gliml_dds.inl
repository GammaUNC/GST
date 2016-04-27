//------------------------------------------------------------------------------
//  gliml_dds.inl
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
inline bool
is_dds(const void* data, unsigned int byteSize) {
    if (byteSize > sizeof(dds_header)) {
        const dds_header* hdr = (const dds_header*) data;
        return hdr->dwMagicFourCC == ' SDD';
    }
    return false;
}

//------------------------------------------------------------------------------
inline bool
context::load_dds(const void* data, unsigned int byteSize) {
    GLIML_ASSERT(gliml::is_dds(data, byteSize));
    this->clear();
    
    const dds_header* hdr = (const dds_header*) data;
    const unsigned char* dataBytePtr = (const unsigned char*) hdr;
    dataBytePtr += sizeof(dds_header);

    // cubemap?
    bool isCubeMap = false;
    if (GLIML_DDSF_CUBEMAP & hdr->dwCaps2) {
        this->target = GLIML_GL_TEXTURE_CUBE_MAP;
        this->is2D = true;
        this->is3D = false;
        this->numFaces = 6;
        isCubeMap = true;
    }
    // 3D texture?
    else if ((GLIML_DDSF_VOLUME & hdr->dwCaps2) && (hdr->dwDepth > 0)) {
        this->target = GLIML_GL_TEXTURE_3D;
        this->is2D = false;
        this->is3D = true;
        this->numFaces = 1;
    }
    // 2D texture
    else {
        this->target = GLIML_GL_TEXTURE_2D;
        this->is2D = true;
        this->is3D = false;
        this->numFaces = 1;
    }
    
    // image format
    int bytesPerElement = 0;
    if (hdr->ddspf.dwFlags & GLIML_DDSF_FOURCC) {
        // test if DXT compressed formats are supported by GL implementation
        if (!this->dxtEnabled) {
            this->errorCode = GLIML_ERROR_DXT_NOT_ENABLED;
            return false;
        }    
        this->isCompressed = true;
        switch (hdr->ddspf.dwFourCC) {
            case GLIML_FOURCC_DXT1:
                this->format = this->internalFormat = GLIML_GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
                bytesPerElement = 8;
                break;
            case GLIML_FOURCC_DXT3:
                this->format = this->internalFormat = GLIML_GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
                bytesPerElement = 16;
                break;
            case GLIML_FOURCC_DXT5:
                this->format = this->internalFormat = GLIML_GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
                bytesPerElement = 16;
                break;
            default:
                this->errorCode = GLIML_ERROR_INVALID_COMPRESSED_FORMAT;
                return false;
        }
    }
    else if ((hdr->ddspf.dwFlags & (GLIML_DDSF_RGBA|GLIML_DDSF_RGB)) && (hdr->ddspf.dwRGBBitCount == 32)) {
        // 32-bit RGBA, check byte order
        bytesPerElement = 4;
        this->type = GLIML_GL_UNSIGNED_BYTE;
        this->internalFormat = GLIML_GL_RGBA;
        if (hdr->ddspf.dwRBitMask == 0x00FF0000) {
            // Direct3D style BGRA
            if (this->bgraEnabled) {
                this->format = GLIML_GL_BGRA;
            }
            else {
                this->errorCode = GLIML_ERROR_BGRA_NOT_ENABLED;
                return false;
            }
        }
        else {
            // OpenGLES style RGBA
            this->format = GLIML_GL_RGBA;
        }
    }    
    else if ((hdr->ddspf.dwFlags & GLIML_DDSF_RGB) && (hdr->ddspf.dwRGBBitCount == 24)) {
        // 24-bit RGB, check byte order
        bytesPerElement = 3;
        this->type = GLIML_GL_UNSIGNED_BYTE;
        this->internalFormat = GLIML_GL_RGB;
        if (hdr->ddspf.dwRBitMask == 0x00FF0000) {
            // Direct3D style BGR
            if (this->bgraEnabled) {
                this->format = GLIML_GL_BGR;
            }
            else {
                this->errorCode = GLIML_ERROR_BGRA_NOT_ENABLED;
                return false;
            }
        }
        else {
            // OpenGLES style RGB
            this->format = GLIML_GL_RGB;
        }
    }
    else if (hdr->ddspf.dwRGBBitCount == 16) {
        // 16-bit RGB(A)
        bytesPerElement = 2;
        if (hdr->ddspf.dwABitMask == 0) {
            // RGB565 or BGR565
            this->type = GLIML_GL_UNSIGNED_SHORT_5_6_5;
            this->internalFormat = GLIML_GL_RGB;
            if (hdr->ddspf.dwRBitMask == (0x1F << 11)) {
                // Direct3D style 
                if (this->bgraEnabled) {
                    this->format = GLIML_GL_BGR;
                }
                else {
                    this->errorCode = GLIML_ERROR_BGRA_NOT_ENABLED;
                    return false;
                }
            }
            else {
                // OpenGLES style
                this->format = GLIML_GL_RGB;
            }
        }
        else {
            // RGBA4 or BGRA4 or RGBA5551 or BGRA5551
             this->internalFormat = GLIML_GL_RGBA;
            if (hdr->ddspf.dwRBitMask == 0x00F0) {
                // Direct3D style bgra4
                if (this->bgraEnabled) {
                    this->type = GLIML_GL_UNSIGNED_SHORT_4_4_4_4;
                    this->format = GLIML_GL_BGRA;
                }
                else {
                    this->errorCode = GLIML_ERROR_BGRA_NOT_ENABLED;
                    return false;
                }
            }
            else if (hdr->ddspf.dwRBitMask == 0xF000) {
                // OpenGLES style rgba4
                this->type = GLIML_GL_UNSIGNED_SHORT_4_4_4_4;
                this->format = GLIML_GL_RGBA;
            }
            else if (hdr->ddspf.dwRBitMask == (0x1F << 1)) {
                // Direc3D style bgra5551
                if (this->bgraEnabled) {
                    this->type = GLIML_GL_UNSIGNED_SHORT_5_5_5_1;
                    this->format = GLIML_GL_BGRA;
                }
                else {
                    this->errorCode = GLIML_ERROR_BGRA_NOT_ENABLED;
                    return false;
                }
            }
            else {
                // OpenGLES style rgba5551
                this->type = GLIML_GL_UNSIGNED_SHORT_5_5_5_1;
                this->format = GLIML_GL_RGBA;
            }
        }
    }
    else if (hdr->ddspf.dwRGBBitCount == 8) {
        this->format = GLIML_GL_LUMINANCE;
        this->internalFormat = GLIML_GL_LUMINANCE;
        this->type = GLIML_GL_UNSIGNED_BYTE;
        bytesPerElement = 1;
    }
    
    // for each face...
    int faceIndex;
    for (faceIndex = 0; faceIndex < this->numFaces; faceIndex++) {
        face& curFace = this->faces[faceIndex];
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
        curFace.numMipmaps = (hdr->dwMipMapCount == 0) ? 1 : hdr->dwMipMapCount;
        
        // for each mipmap
        int mipIndex;
        for (mipIndex = 0; mipIndex < curFace.numMipmaps; mipIndex++) {
            face::mipmap& curMip = curFace.mipmaps[mipIndex];
            
            // mipmap dimensions
            int w = hdr->dwWidth >> mipIndex;
            if (w <= 0) w = 1;
            int h = hdr->dwHeight >> mipIndex;
            if (h <= 0) h = 1;
            int d = hdr->dwDepth >> mipIndex;
            if (d <= 0) d = 1;
            curMip.width = w;
            curMip.height = h;
            curMip.depth = d;
            
            // mipmap byte size
            if (this->isCompressed) {
                curMip.size = ((w+3)/4) * ((h+3)/4) * d * bytesPerElement;
            }
            else {
                curMip.size = w * h * d * bytesPerElement;
            }
            
            // set and advance surface data pointer
            curMip.data = dataBytePtr;
            dataBytePtr += curMip.size;
        }
    }
    GLIML_ASSERT(dataBytePtr == ((const unsigned char*)data) + byteSize);
    
    // ...and we're done
    return true;
}
