//------------------------------------------------------------------------------
//  gliml_ktx.inl
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
inline bool
is_ktx(const void* data, unsigned int byteSize) {
    static const int numMagic = 12;
    static const unsigned char magic[numMagic] = {
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
    };
    if (byteSize > sizeof(ktx_header)) {
        const ktx_header* hdr = (const ktx_header*) data;
        int i;
        for (i = 0; i < numMagic; i++) {
            if (hdr->identifier[i] != magic[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------
inline void*
mip_data_and_size(const ktx_header* hdr, int faceIndex, int mipIndex, int& outSize) {
    unsigned char* result = 0;
    unsigned char* ptr = (unsigned char*) hdr;
    ptr += sizeof(ktx_header) + hdr->bytesOfKeyValueData;
    for (int i = 0; i <= mipIndex; i++) {
        const unsigned int faceSize = *(unsigned int*)ptr;

        // set result to start of face data within miplevel
        result = ptr + 4 + (faceSize * faceIndex);
        outSize = faceSize;

        // advance to start of next miplevel
        int mipPadding = 3 - ((faceSize + 3) % 4);
        GLIML_ASSERT(mipPadding >= 0);
        ptr = ptr + 4 + (faceSize * hdr->numberOfFaces) + mipPadding;
    }
    return result;
}

//------------------------------------------------------------------------------
inline bool
context::load_ktx(const void* data, unsigned int byteSize) {
    GLIML_ASSERT(gliml::is_ktx(data, byteSize));
    this->clear();

    const ktx_header* hdr = (const ktx_header*) data;

    // check if file and host system endianess match
    if (hdr->endianness != 0x04030201) {
        this->errorCode = GLIML_ERROR_ENDIAN_MISMATCH;
        return false;
    }
    // array textures not currently supported
    if (hdr->numberOfArrayElements > 0) {
        this->errorCode = GLIML_ERROR_TEXTURE_ARRAYS_NOT_SUPPORTED;
        return false;
    }

    // parse header
    this->type = hdr->glType;
    this->format = hdr->glFormat;
    this->isCompressed = (0 == this->format);
    this->numFaces = hdr->numberOfFaces;
    if (!((1 == this->numFaces) || (6 == this->numFaces))) {
        this->errorCode = GLIML_ERROR_INVALID_NUMBER_OF_CUBEMAP_FACES;
        return false;
    }
    if (this->isCompressed) {
        // compressed format
        this->internalFormat = hdr->glInternalFormat;

        // check if format is ETC2, and ETC2 was enabled
        if ((GLIML_GL_COMPRESSED_RGB8_ETC2 == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_SRGB8_ETC2 == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_RGBA8_ETC2_EAC == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC == this->internalFormat))
        {
            if (!this->etc2Enabled) {
                this->errorCode = GLIML_ERROR_ETC2_NOT_ENABLED;
                return false;
            }
        }
        // check if format is PVRTC, and PVRTC was enabled
        else if ((GLIML_GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG == this->internalFormat) ||
            (GLIML_GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG == this->internalFormat))
        {
            if (!this->pvrtcEnabled) {
                this->errorCode = GLIML_ERROR_PVRTC_NOT_ENABLED;
                return false;
            }
        }

        // check if format is DXT, and DXT was enabled
        else if ((GLIML_GL_COMPRESSED_RGBA_S3TC_DXT1_EXT == this->internalFormat) ||
                 (GLIML_GL_COMPRESSED_RGBA_S3TC_DXT3_EXT == this->internalFormat) ||
                 (GLIML_GL_COMPRESSED_RGBA_S3TC_DXT5_EXT == this->internalFormat))
        {
            if (!this->dxtEnabled) {
                this->errorCode = GLIML_ERROR_DXT_NOT_ENABLED;
                return false;
            }
        }
    }
    else {
        // uncompressed format
        this->internalFormat = hdr->glBaseInternalFormat;
    }

    // texture type
    bool isCubeMap = false;
    if (0 != hdr->pixelDepth) {
        this->target = GLIML_GL_TEXTURE_3D;
        this->is3D = true;
    }
    else if (hdr->pixelHeight != 0) {
        this->is2D = true;
        if (1 == this->numFaces) {
            this->target = GLIML_GL_TEXTURE_2D;
        }
        else {
            this->target = GLIML_GL_TEXTURE_CUBE_MAP;
            isCubeMap = true;
        }
    }

    // setup face-data
    int faceIndex, mipIndex;
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
        curFace.numMipmaps = hdr->numberOfMipmapLevels;
        if (0 == curFace.numMipmaps) {
            // FIXME: this means that a mipmap should be generated, but
            // we'll just return only a single mipmap instead
            curFace.numMipmaps = 1;
        }

        // for each mipmap...
        for (mipIndex = 0; mipIndex < curFace.numMipmaps; mipIndex++) {
            face::mipmap& curMip = curFace.mipmaps[mipIndex];

            // mipmap dimensions
            int w = hdr->pixelWidth >> mipIndex;
            if (w <= 0) w = 1;
            int h = hdr->pixelHeight >> mipIndex;
            if (h <= 0) h = 1;
            int d = hdr->pixelDepth >> mipIndex;
            if (d <= 0) d = 1;
            curMip.width = w;
            curMip.height = h;
            curMip.depth = d;
            curMip.data = mip_data_and_size(hdr, faceIndex, mipIndex, curMip.size);
            GLIML_ASSERT(curMip.data < ((const char*)data)+byteSize);
        }
    }
    return true;
}