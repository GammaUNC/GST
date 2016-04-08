#include <cassert>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "codec.h"

// Our encoder is quite simple...
int main(int argc, char **argv) {
  // Make sure that we have the proper number of arguments...
  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <original> [compressed] <output>" << std::endl;
    return 1;
  }

  const char *orig_fn = argv[1];
  const char *cmp_fn = (argc == 3) ? NULL : argv[2];
  const char *dst_fn = (argc == 4) ? argv[3] : argv[2];

  std::vector<uint8_t> cmp_img = std::move(GenTC::CompressDXT(orig_fn, cmp_fn));
  std::ofstream out (dst_fn, std::ofstream::binary);
  out.write(reinterpret_cast<const char *>(cmp_img.data()), cmp_img.size());
  out.close();

  return 0;
}
