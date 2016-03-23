#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// We use one thread per pixel, and the group size (local work size)
// dictates how big the dimensions are of the 

__kernel void inv_wavelet(const __constant uchar *wavelet_data,
					      const int value_offset,
                          __local uchar *local_data,
                          __global uchar *out_data)
{
  uint local_idx = get_local_id(1) * get_local_size(0) + get_local_id(0);
  uint local_sz = get_local_size(0) * get_local_size(1);
  uint group_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);

  // Grab global value and place it in local data in preparation for inv
  // wavelet transform....
  local_data = wavelet_data[group_idx * local_sz + local_idx];

  // We need a barrier to make sure that all threads read the data they needed
  barrier(CLK_LOCAL_MEM_FENCE);
}
