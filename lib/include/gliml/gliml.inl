//------------------------------------------------------------------------------
//  gliml.inl
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
inline context::context() :
dxtEnabled(false),
pvrtcEnabled(false),
etc2Enabled(false),
bgraEnabled(false) {
    this->clear();
}

//------------------------------------------------------------------------------
inline context::~context() {
    // empty
}

//------------------------------------------------------------------------------
inline void
context::enable_dxt(bool b) {
    this->dxtEnabled = b;
}

//------------------------------------------------------------------------------
inline void
context::enable_pvrtc(bool b) {
    this->pvrtcEnabled = b;
}

//------------------------------------------------------------------------------
inline void
context::enable_etc2(bool b) {
    this->etc2Enabled = b;
}

//------------------------------------------------------------------------------
inline void
context::enable_bgra(bool b) {
    this->bgraEnabled = b;
}

//------------------------------------------------------------------------------
inline bool
context::load(const void* data, unsigned int size) {
    #ifndef GLIML_NO_DDS
    if (is_dds(data, size)) {
        return this->load_dds(data, size);
    }
    #endif
    #ifndef GLIML_NO_PVR
    if (is_pvr(data, size)) {
        return this->load_pvr(data, size);
    }
    #endif
    #ifndef GLIML_NO_KTX
    if (is_ktx(data, size)) {
        return this->load_ktx(data, size);
    }
    #endif
    this->errorCode = GLIML_ERROR_UNKNOWN_FILE_FORMAT;
    return false;
}

//------------------------------------------------------------------------------
inline void
context::clear() {
    this->errorCode = GLIML_SUCCESS;
    this->target = 0;
    this->isCompressed = false;
    this->is2D = false;
    this->is3D = false;
    this->internalFormat = 0;
    this->format = 0;
    this->type = 0;
    this->numFaces = 0;
    int faceIndex;
    for (faceIndex = 0; faceIndex < MaxNumFaces; faceIndex++) {
        face& curFace = this->faces[faceIndex];
        curFace.target = 0;
        curFace.numMipmaps = 0;
        int mipIndex;
        for (mipIndex = 0; mipIndex < MaxNumMipmaps; mipIndex++) {
            face::mipmap& curMipmap = curFace.mipmaps[mipIndex];
            curMipmap.width = 0;
            curMipmap.height = 0;
            curMipmap.depth = 0;
            curMipmap.size  = 0;
            curMipmap.data = 0;
        }
    }
}

//------------------------------------------------------------------------------
inline GLenum
context::texture_target() const {
    return this->target;
}

//------------------------------------------------------------------------------
inline int
context::error() const {
    return this->errorCode;
}

//------------------------------------------------------------------------------
inline bool
context::is_compressed() const {
    return this->isCompressed;
}

//------------------------------------------------------------------------------
inline bool
context::is_2d() const {
    return this->is2D;
}

//------------------------------------------------------------------------------
inline bool
context::is_3d() const {
    return this->is3D;
}

//------------------------------------------------------------------------------
inline int
context::num_faces() const {
    return this->numFaces;
}

//------------------------------------------------------------------------------
inline int
context::num_mipmaps(int face_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    return this->faces[face_index].numMipmaps;
}

//------------------------------------------------------------------------------
inline GLenum
context::image_target(int face_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    return this->faces[face_index].target;
}

//------------------------------------------------------------------------------
inline GLint
context::image_internal_format() const {
    return this->internalFormat;
}

//------------------------------------------------------------------------------
inline GLsizei
context::image_width(int face_index, int mip_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    GLIML_ASSERT((mip_index >= 0) && (mip_index < this->faces[face_index].numMipmaps));
    return this->faces[face_index].mipmaps[mip_index].width;
}

//------------------------------------------------------------------------------
inline GLsizei
context::image_height(int face_index, int mip_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    GLIML_ASSERT((mip_index >= 0) && (mip_index < this->faces[face_index].numMipmaps));
    return this->faces[face_index].mipmaps[mip_index].height;
}

//------------------------------------------------------------------------------
inline GLsizei
context::image_depth(int face_index, int mip_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    GLIML_ASSERT((mip_index >= 0) && (mip_index < this->faces[face_index].numMipmaps));
    return this->faces[face_index].mipmaps[mip_index].depth;
}

//------------------------------------------------------------------------------
inline GLenum
context::image_format() const {
    return this->format;
}

//------------------------------------------------------------------------------
inline GLenum
context::image_type() const {
    return this->type;
}

//------------------------------------------------------------------------------
inline GLsizei
context::image_size(int face_index, int mip_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    GLIML_ASSERT((mip_index >= 0) && (mip_index < this->faces[face_index].numMipmaps));
    return this->faces[face_index].mipmaps[mip_index].size;
}

//------------------------------------------------------------------------------
inline const GLvoid*
context::image_data(int face_index, int mip_index) const {
    GLIML_ASSERT((face_index >= 0) && (face_index < this->numFaces));
    GLIML_ASSERT((mip_index >= 0) && (mip_index < this->faces[face_index].numMipmaps));
    return this->faces[face_index].mipmaps[mip_index].data;
}
