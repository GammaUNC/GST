#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void collect_endpoints(const __global char *_y,
                                const __global char *_co,
                                const __global char *_cg,
                                __global ushort *out,
                                uint endpoint_index) {
  int y = (int)(_y[get_global_id(0)]);
  int co = (int)(_co[get_global_id(0)]);
  int cg = (int)(_cg[get_global_id(0)]);

  int t = y - (cg / 2);
  int g = cg + t;
  int b = (t - co) / 2;
  int r = b + co;

  // RGB should be 565 at this point...
  ushort pixel = 0;
  pixel |= (r << 11);
  pixel |= (g << 5);
  pixel |= b;

  out[4*get_global_id(0) + endpoint_index] = pixel;
}
