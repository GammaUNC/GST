#include "codec_base.h"

#include <cstring>
#include <iostream>

namespace GenTC {

void GenTCHeader::Print() const {
  std::cout << "Width: " << width << std::endl;
  std::cout << "Height: " << height << std::endl;
  std::cout << "Num Palette Entries: " << (palette_bytes / 4) << std::endl;
  std::cout << "Y compressed size: " << y_cmp_sz << std::endl;
  std::cout << "Chroma compressed size: " << chroma_cmp_sz << std::endl;
  std::cout << "Palette size compressed: " << palette_sz << std::endl;
  std::cout << "Palette index deltas compressed: " << indices_sz << std::endl;
}

void GenTCHeader::LoadFrom(const uint8_t *buf) {
  // Read the header
  memcpy(this, buf, sizeof(*this));
#ifndef NDEBUG
  Print();
#endif
}

}  //  namespace GenTC
