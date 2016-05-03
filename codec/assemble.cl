#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

int Get(const __global char *planes, uint offset) {
  const uint idx = get_global_size(0) * offset + get_global_id(0);
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

void YCoCgToRGB(int *y, int *co, int *cg) {
  int t = *y - (*cg / 2);
  int g = *cg + t;
  *cg = (t - *co) / 2;
  *y = *cg + *co;
  *co = g;
}

ushort GetPixel(const __global char *planes, uint endpoint_idx) {
  int r = GetY(planes, endpoint_idx);
  int g = GetCo(planes, endpoint_idx);
  int b = GetCg(planes, endpoint_idx);

  YCoCgToRGB(&r, &g, &b);

  // RGB should be 565 at this point...
  ushort pixel = 0;
  pixel |= (r << 11);
  pixel |= (g << 5);
  pixel |= b;

  return pixel;
}

__kernel void assemble_dxt(const __global    int  *global_palette,
                           const __constant uint  *global_offsets,
                           const __global   char  *endpoint_planes,
						   const __global    int  *indices,
						         __global ushort  *global_out) {

  const uint global_offset = get_global_size(0) * 6 * get_global_id(1);

  ushort ep1 = GetPixel(endpoint_planes + global_offset, 0);
  ushort ep2 = GetPixel(endpoint_planes + global_offset, 1);

  __global ushort *out = global_out + get_global_id(1) * get_global_size(0) * 4;
  out[4 * get_global_id(0) + 0] = ep1;
  out[4 * get_global_id(0) + 1] = ep2;

  const __global int *palette = global_palette + global_offsets[4 * get_global_id(1) + 2] / 4;
  const uint plt_idx = indices[get_global_id(1) * get_global_size(0) + get_global_id(0)];
  __global uint *out_indices = (__global uint *)(out) + 2 * get_global_id(0) + 1;
  *out_indices = palette[plt_idx];
}