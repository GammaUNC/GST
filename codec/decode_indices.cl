#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define LOCAL_SCAN_SIZE_LOG 7
#define LOCAL_SCAN_SIZE 128

__kernel void decode_indices(const __global uchar *index_data,
							 int stage, __global int *out) {
  const size_t num_vals = LOCAL_SCAN_SIZE;
  __local int scratch[LOCAL_SCAN_SIZE];

  // First read in data
  uint idx_offset = (1 << (stage * LOCAL_SCAN_SIZE_LOG));
  uint gidx = idx_offset * (get_global_id(0) + 1) - 1;
  if (0 == stage) {
	scratch[get_local_id(0)] = (int)(index_data[get_global_id(0)]) - 128;
  } else {
    scratch[get_local_id(0)] = out[2 * gidx + 1];
  }

  int tid = get_local_id(0);
  uint offset = 1;
  for (int i = (num_vals >> 1); i > 0; i = i >> 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < i) {
      uint a = offset * (2 * tid + 1) - 1;
      uint b = offset * (2 * tid + 2) - 1;
      scratch[b] += scratch[a];
    }

    offset *= 2;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    scratch[num_vals - 1] = 0;
  }

  for (int d = 1; d < num_vals; d *= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    offset >>= 1;
    if (tid < d) {
      uint a = offset * (2 * tid + 1) - 1;
      uint b = offset * (2 * tid + 2) - 1;

      int t = scratch[a];
      scratch[a] = scratch[b];
      scratch[b] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  out[2 * gidx + 1] = scratch[get_local_id(0)];
}

__kernel void collect_indices(const __global int *palette,
							  int stage, __global int *out) {
  if (0 == stage) {
    uint idx = 2 * get_global_id(0) + 1;
    out[idx] = palette[out[idx]];
  } else {
    uint offset = 1 << (stage * LOCAL_SCAN_SIZE_LOG);
    uint gidx = offset * get_group_id(0) - 1;

    uint next_offset = 1 << ((stage - 1) * LOCAL_SCAN_SIZE_LOG);
    uint tidx = next_offset * (get_global_id(0) + 1) - 1;
    if (get_group_id(0) > 0 && tidx != gidx) {
      out[2 * tidx + 1] += out[2 * gidx + 1];
    }
  }
}
