#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define LOCAL_SCAN_SIZE_LOG 7
#define LOCAL_SCAN_SIZE 128

__kernel void decode_indices(const __global   uchar *index_data,
                             const __constant uint  *global_offsets,
                             const            uint   stage,
							 const            uint   num_vals,
							       __global    int  *out) {
  __local int scratch[LOCAL_SCAN_SIZE];

  // First read in data
  const uint out_offset = get_global_id(1) * num_vals;
  const uint idx_offset = (1 << (stage * LOCAL_SCAN_SIZE_LOG));
  const uint gidx = idx_offset * (get_global_id(0) + 1) - 1;
  if (0 == stage) {
    const uint data_offset = global_offsets[4 * get_global_id(1) + 3];
    scratch[get_local_id(0)] = (int)(index_data[data_offset + get_global_id(0)]) - 128;
  } else {
    scratch[get_local_id(0)] = out[out_offset + 2 * gidx + 1];
  }

  const int tid = get_local_id(0);
  uint offset = 1;
  for (int i = (LOCAL_SCAN_SIZE >> 1); i > 0; i = i >> 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < i) {
      const uint a = offset * (2 * tid + 1) - 1;
      const uint b = offset * (2 * tid + 2) - 1;
      scratch[b] += scratch[a];
    }

    offset *= 2;
  }

  // Change it up for inclusive scan...
  offset >>= 1;
  for (int d = 2; d < LOCAL_SCAN_SIZE; d *= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    offset >>= 1;
    if (offset > 0 && 0 < tid && tid < d) {
      uint a = offset * (2 * tid) - 1;
      uint b = offset * (2 * tid + 1) - 1;
      scratch[b] += scratch[a];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  out[out_offset + 2 * gidx + 1] = scratch[get_local_id(0)];
}

__kernel void collect_indices(const __global    int  *palette,
                              const __constant uint  *global_offsets,
							  const             int   stage,
							  const            uint   num_vals,
							        __global    int  *out) {
  const uint out_offset = get_global_id(1) * num_vals;
  const uint palette_offset = global_offsets[4 * get_global_id(1) + 2] / 4;
  
  // !SPEED! This should really just be two separate kernels
  if (0 == stage) {
    uint idx = 2 * get_global_id(0) + 1;
    out[out_offset + idx] = palette[palette_offset + out[idx]];
  } else {
    uint offset = 1 << (stage * LOCAL_SCAN_SIZE_LOG);
    uint gidx = offset * get_group_id(0) - 1;

    uint next_offset = 1 << ((stage - 1) * LOCAL_SCAN_SIZE_LOG);
    uint tidx = next_offset * (get_global_id(0) + 1) - 1;

    if (get_group_id(0) > 0 && get_local_id(0) != (LOCAL_SCAN_SIZE - 1)) {
      out[out_offset + 2 * tidx + 1] += out[out_offset + 2 * gidx + 1];
    }
  }
}
