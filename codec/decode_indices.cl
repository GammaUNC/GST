#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define LOCAL_SCAN_SIZE_LOG 7
#define LOCAL_SCAN_SIZE 128

__kernel void decode_indices(const __global   uchar *global_index_data,
                             const __constant uint  *global_offsets,
                             const            uint   stage,
                             const            uint   num_vals,
                                   __global   int   *global_out) {
  const __global uchar *const index_data =
    global_index_data + global_offsets[4 * get_global_id(1) + 3];

  __global int *const out = global_out + num_vals * get_global_id(1);
  __local int scratch[LOCAL_SCAN_SIZE];

  // First read in data
  // !SPEED! This almost definitely causes bank conflicts for stage > 0.
  const uint idx_offset = (1 << (stage * LOCAL_SCAN_SIZE_LOG));
  const int gidx = idx_offset * (get_global_id(0) + 1) - 1;
  const int pgidx = idx_offset * get_global_id(0) - 1;

  if (0 == stage) {
    scratch[get_local_id(0)] = (int)(index_data[get_global_id(0)]) - 128;
  } else if (gidx < num_vals) {
    scratch[get_local_id(0)] = out[gidx];
  } else if (pgidx < num_vals) {
    scratch[get_local_id(0)] = out[num_vals - 1];
  } else {
    scratch[get_local_id(0)] = 0;
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
  if (0 == stage || gidx < num_vals) {
    out[gidx] = scratch[get_local_id(0)];
  } else if (pgidx < num_vals && gidx >= num_vals) {
    out[num_vals - 1] = scratch[get_local_id(0)];
  }
}

__kernel void collect_indices(const            int    stage,
                              const            uint   num_vals,
                                    __global   int   *global_out) {
  __global int *const out = global_out + num_vals * get_global_id(1);

  uint offset = 1 << (stage * LOCAL_SCAN_SIZE_LOG);
  uint gidx = offset * get_group_id(0) - 1;

  uint next_offset = 1 << ((stage - 1) * LOCAL_SCAN_SIZE_LOG);
  uint tidx = next_offset * (get_global_id(0) + 1) - 1;

  if (tidx < (num_vals - 1) && gidx < (num_vals - 1) && get_group_id(0) > 0 && get_local_id(0) != (LOCAL_SCAN_SIZE - 1)) {
    out[tidx] += out[gidx];
  }
}
