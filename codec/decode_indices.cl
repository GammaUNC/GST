#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void decode_indices(const __global int *palette,
                             const __global uchar *index_data,
                             __global int *out) {
  size_t num_vals = get_global_size(0);
  uint tid = get_global_id(0);

  // First read in data
  uint idx = 2 * tid + 1;
  out[idx] = (int)(index_data[get_global_id(0)]) - 128;

  uint offset = 2;
  for (int i = (num_vals >> 1); i > 0; i = i >> 1) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < i) {
      uint a = offset * (2 * tid + 1) - 1;
      uint b = offset * (2 * tid + 2) - 1;
      out[b] += out[a];
    }
    offset *= 2;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (tid == 0) {
    out[2 * (num_vals - 1) + 1] = 0;
  }

  for (int d = 1; d < num_vals; d *= 2) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    offset >>= 1;
    if (tid < d) {
      uint a = offset * (2 * tid + 1) - 1;
      uint b = offset * (2 * tid + 2) - 1;

      int t = out[a];
      out[a] = out[b];
      out[b] += t;
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  uint idxx = 2 * tid + 1;
  uint outt = out[idxx];
  printf("%d, %d:\t%d\n", get_group_id(0), idxx, outt);
  out[idx] = palette[out[idx]];
}
