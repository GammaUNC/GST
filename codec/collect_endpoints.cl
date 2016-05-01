#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void collect_endpoints(const __global char *in,
                                __global ushort *out) {
  const uint num_values_in_plane = get_global_size(0);
  const uint global_offset = num_values_in_plane * 6 * get_global_id(2);
  const uint y_idx = num_values_in_plane * get_global_id(1) + get_global_id(0);
  const uint co_idx = num_values_in_plane * (2 + 2 * get_global_id(1)) + get_global_id(0);
  const uint cg_idx = num_values_in_plane * (2 + 2 * get_global_id(1) + 1) + get_global_id(0);

  int y = (int)(in[global_offset + y_idx]);
  int co = (int)(in[global_offset + co_idx]);
  int cg = (int)(in[global_offset + cg_idx]);

  int t = y - (cg / 2);
  int g = cg + t;
  int b = (t - co) / 2;
  int r = b + co;

  // RGB should be 565 at this point...
  ushort pixel = 0;
  pixel |= (r << 11);
  pixel |= (g << 5);
  pixel |= b;

  out[get_global_id(2) * 4 + 4*get_global_id(0) + get_global_id(1)] = pixel;
}
