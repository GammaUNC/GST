#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

const sampler_t gSampler =
  CLK_ADDRESS_REPEAT | CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

uchar2 PackPixel(uchar3 p) {
  const uchar r = p.x & 0xF8;
  const uchar g = p.y & 0xFC;
  const uchar b = p.z & 0xF8;
  return (uchar2)(r | g >> 5, g << 3 | (b >> 3));
}

__kernel void compressDXT(__read_only image2d_t src,
                          __global __write_only uchar8 *dst)
{
  const uint width = get_image_width(src);
  const uint height = get_image_height(src);
  const uint nBlocksX = width / 4;
  const uint nBlocksY = height / 4;

  const ushort blockX = get_global_id(0);
  const ushort blockY = get_global_id(1);

  const int x = blockX * 4;
  const int y = blockY * 4;

  // Determine min/max
  int3 minPixel = (int3)(255, 255, 255);
  int3 maxPixel = (int3)(0, 0, 0);
  for (uint j = 0; j < 4; j++) {
    for (uint i = 0; i < 4; i++) {
      int4 p = read_imagei(src, gSampler, (int2)(x + i, y + j));
      minPixel.x = min(minPixel.x, p.x);
      minPixel.y = min(minPixel.y, p.y);
      minPixel.z = min(minPixel.z, p.z);

      maxPixel.x = max(maxPixel.x, p.x);
      maxPixel.y = max(maxPixel.y, p.y);
      maxPixel.z = max(maxPixel.z, p.z);
    }
  }

  const int3 inset = (maxPixel - minPixel) >> 4;
  const int3 minPixelInset = minPixel + inset;
  const int3 maxPixelInset = maxPixel - inset;
  minPixel.x = min(minPixelInset.x, 255);
  minPixel.y = min(minPixelInset.y, 255);
  minPixel.z = min(minPixelInset.z, 255);
  maxPixel.x = max(maxPixelInset.x, 0);
  maxPixel.y = max(maxPixelInset.y, 0);
  maxPixel.z = max(maxPixelInset.z, 0);

  const int3 color0 = (int3)(
    ((uint)(maxPixel.x) & 0xF8) | ((uint)(maxPixel.x) >> 5),
    ((uint)(maxPixel.y) & 0xFC) | ((uint)(maxPixel.y) >> 6),
    ((uint)(maxPixel.z) & 0xF8) | ((uint)(maxPixel.z) >> 5));
  const int3 color1 = (int3)(
    ((uint)(minPixel.x) & 0xF8) | ((uint)(minPixel.x) >> 5),
    ((uint)(minPixel.y) & 0xFC) | ((uint)(minPixel.y) >> 6),
    ((uint)(minPixel.z) & 0xF8) | ((uint)(minPixel.z) >> 5));
  const int3 color2 = ((int3)(2) * color0 + color1) / 3;
  const int3 color3 = (color0 + (int3)(2) * color1) / 3;

  uint modulation = 0;
  for(int i = 15; i >= 0; i--) {
    const int xx = 3 - ((15 - i) % 4);
    const int yy = (15 - i) / 4;
    const uint4 c = read_imageui(src, gSampler, (int2)(x + xx, y + yy));
    
    const int d0 =
      abs(color0.x - (int)(c.x)) +
      abs(color0.y - (int)(c.y)) +
      abs(color0.z - (int)(c.z));

    const int d1 =
      abs(color1.x - (int)(c.x)) +
      abs(color1.y - (int)(c.y)) +
      abs(color1.z - (int)(c.z));

    const int d2 =
      abs(color2.x - (int)(c.x)) +
      abs(color2.y - (int)(c.y)) +
      abs(color2.z - (int)(c.z));

    const int d3 =
      abs(color3.x - (int)(c.x)) +
      abs(color3.y - (int)(c.y)) +
      abs(color3.z - (int)(c.z));

    const int b0 = d0 > d3;
    const int b1 = d1 > d2;
    const int b2 = d0 > d2;
    const int b3 = d1 > d3;
    const int b4 = d2 > d3;

    const int x0 = b1 & b2;
    const int x1 = b0 & b3;
    const int x2 = b0 & b4;

    modulation |= (x2 | ((x0 | x1) << 1)) << (i << 1);    
  }

  const uchar4 mods = (uchar4)
    (modulation & 0xFF,
     (modulation >> 8) & 0xFF,
     (modulation >> 16) & 0xFF,
     (modulation >> 24) & 0xFF);
  
  const uchar2 minPacked = PackPixel((uchar3)(minPixel.x, minPixel.y, minPixel.z));
  const uchar2 maxPacked = PackPixel((uchar3)(maxPixel.x, maxPixel.y, maxPixel.z));

  const int idx = blockY * nBlocksX + blockX;
  dst[idx].s10 = maxPacked;
  dst[idx].s32 = minPacked;
  dst[idx].s7654 = mods;
}
