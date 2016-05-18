#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifdef GENTC_APPLE
static uint NumBlocks();
static uint ThreadIdx();
static int Get(const __global char *planes, uint offset);
static int GetY(const __global char *planes, uint endpoint_idx);
static int GetCo(const __global char *planes, uint endpoint_idx);
static int GetCg(const __global char *planes, uint endpoint_idx);
static int4 YCoCgToRGB(int4 in);
static ushort GetPixel(const __global char *planes, uint endpoint_idx);
#endif

uint NumBlocks() {
  return get_global_size(0) * get_global_size(1);
}

uint ThreadIdx() {
  return get_global_id(1) * get_global_size(0) + get_global_id(0);
}

int Get(const __global char *planes, uint offset) {
  const uint idx = NumBlocks() * offset + ThreadIdx();
  return (int)(planes[idx]);
}

int GetY(const __global char *planes, uint endpoint_idx) {
  return Get(planes, endpoint_idx);
}

int GetCo(const __global char *planes, uint endpoint_idx) {
  return Get(planes, 2 + 2 * endpoint_idx);
}

int GetCg(const __global char *planes, uint endpoint_idx) {
  return Get(planes, 2 + 2 * endpoint_idx + 1);
}

int4 YCoCgToRGB(int4 in) {
  int4 out;
  int t = in.x - (in.z / 2);
  out.y = in.z + t;
  out.z = (t - in.y) / 2;
  out.x = out.z + in.y;
  return out;
}

ushort GetPixel(const __global char *planes, uint endpoint_idx) {
  int y = GetY(planes, endpoint_idx);
  int co = GetCo(planes, endpoint_idx);
  int cg = GetCg(planes, endpoint_idx);

  int4 rgb = YCoCgToRGB((int4)(y, co, cg, 0));

  // RGB should be 565 at this point...
  ushort pixel = 0;
  pixel |= (rgb.x << 11);
  pixel |= (rgb.y << 5);
  pixel |= rgb.z;

  return pixel;
}

__kernel void assemble_dxt(const __global    int  *global_palette,
                           const __constant uint  *global_offsets,
                           const __global   char  *endpoint_planes,
                           const __global    int  *indices,
                                 __global ushort  *global_out) {
  const uint global_offset = NumBlocks() * 6 * get_global_id(2);

  ushort ep1 = GetPixel(endpoint_planes + global_offset, 0);
  ushort ep2 = GetPixel(endpoint_planes + global_offset, 1);

  __global ushort *out = global_out + get_global_id(2) * NumBlocks() * 4;
  out[4 * ThreadIdx() + 0] = ep1;
  out[4 * ThreadIdx() + 1] = ep2;

  const __global int *palette = global_palette + global_offsets[4 * get_global_id(2) + 2] / 4;
  const uint plt_idx = indices[get_global_id(2) * NumBlocks() + ThreadIdx()];
  *((__global uint *)(out) + 2 * ThreadIdx() + 1) = palette[plt_idx];
}

__kernel void assemble_rgb(const __global   uint  *global_palette,
                           const __constant uint  *global_offsets,
                           const __global   char  *endpoint_planes,
						   const __global    int  *indices,
						         __global  uchar  *global_out) {
  const uint global_offset = NumBlocks() * 6 * get_global_id(2);

  int4 palette[4];

  palette[0].x = GetY(endpoint_planes + global_offset, 0);
  palette[0].y = GetCo(endpoint_planes + global_offset, 0);
  palette[0].z = GetCg(endpoint_planes + global_offset, 0);
  palette[0] = YCoCgToRGB(palette[0]);

  palette[1].x = GetY(endpoint_planes + global_offset, 1);
  palette[1].y = GetCo(endpoint_planes + global_offset, 1);
  palette[1].z = GetCg(endpoint_planes + global_offset, 1);
  palette[1] = YCoCgToRGB(palette[1]);

  palette[0].x = (palette[0].x << 3) | (palette[0].x >> 2);
  palette[0].y = (palette[0].y << 2) | (palette[0].y >> 4);
  palette[0].z = (palette[0].z << 3) | (palette[0].z >> 2);

  palette[1].x = (palette[1].x << 3) | (palette[1].x >> 2);
  palette[1].y = (palette[1].y << 2) | (palette[1].y >> 4);
  palette[1].z = (palette[1].z << 3) | (palette[1].z >> 2);

  palette[2] = (2 * palette[0] + palette[1]) / 3;
  palette[3] = (palette[0] + 2 * palette[1]) / 3;

  const uint plt_idx = indices[get_global_id(2) * NumBlocks() + ThreadIdx()];
  uint idx = (global_palette + global_offsets[4 * get_global_id(2) + 2] / 4)[plt_idx];

  __global uchar *out = global_out + NumBlocks() * 3 * 16 * get_global_id(2);
  for (int i = 0; i < 16; ++i) {
    int4 rgb = palette[idx & 3];

    uint x = 4 * get_global_id(0) + (i % 4);
    uint y = 4 * get_global_id(1) + (i / 4);

    uint out_offset  = 3 * (4 * get_global_size(0) * y + x);
    out[out_offset + 0] = rgb.x;
    out[out_offset + 1] = rgb.y;
    out[out_offset + 2] = rgb.z;

    idx >>= 2;
  }
}