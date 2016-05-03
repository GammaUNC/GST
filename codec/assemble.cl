#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

ushort GetPixel(const __global char *planes, uint endpoint_idx) {
  const uint y_idx = get_global_size(0) * endpoint_idx + get_global_id(0);
  const uint co_idx = get_global_size(0) * (2 + 2 * endpoint_idx) + get_global_id(0);
  const uint cg_idx = get_global_size(0) * (2 + 2 * endpoint_idx + 1) + get_global_id(0);

  int y = (int)(planes[y_idx]);
  int co = (int)(planes[co_idx]);
  int cg = (int)(planes[cg_idx]);

  int t = y - (cg / 2);
  int g = cg + t;
  int b = (t - co) / 2;
  int r = b + co;

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
						         __global ushort  *global_out) {

  const uint global_offset = get_global_size(0) * 6 * get_global_id(1);

  ushort ep1 = GetPixel(endpoint_planes + global_offset, 0);
  ushort ep2 = GetPixel(endpoint_planes + global_offset, 1);

  __global ushort *out = global_out + get_global_id(1) * get_global_size(0) * 4;
  out[4*get_global_id(0) + 0] = ep1;
  out[4*get_global_id(0) + 1] = ep2;

  __global uint *indices = (__global uint *)(out + 4*get_global_id(0) + 2);
  const __global int *palette = global_palette + global_offsets[4 * get_global_id(1) + 2] / 4;
  *indices = palette[*indices];
}
