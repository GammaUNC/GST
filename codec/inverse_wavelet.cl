#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

int NormalizeIndex(int idx, int range) {
  return abs(idx - (int)(idx >= range) * (idx - range + 2));
}

int GetAt(__local int *ptr, uint x, uint y) {
  const uint stride = 2 * get_local_size(1);
  return ptr[y * stride + x];
}

void PutAt(__local int *ptr, uint x, uint y, int val) {
  const uint stride = 2 * get_local_size(1);
  ptr[y * stride + x] = val;
}

void InverseWaveletEven(__local int *src, __local int *scratch,
                        uint x, uint y, uint len, uint mid) {
  // Original C++ code:
  // int prev = static_cast<int>(mid_pt) + NormalizeIndex(i - 1, len) / 2;
  // int next = static_cast<int>(mid_pt) + NormalizeIndex(i + 1, len) / 2;
  // dst[i] = src[i / 2] - (src[prev] + src[next] + 2) / 4;

  const uint idx = 2 * x;
  const int prev = mid + NormalizeIndex((int)(idx) - 1, (int)len) / 2;
  const int next = mid + NormalizeIndex((int)(idx) + 1, (int)len) / 2;

  const int src_prev = GetAt(src, (uint)prev, y);
  const int src_next = GetAt(src, (uint)next, y);

  // Transpose the result!
  PutAt(scratch, y, idx, GetAt(src, x, y) - (src_prev + src_next + 2) / 4);
}

void InverseWaveletOdd(__local int *src, __local int *scratch,
                       uint x, uint y, uint len, uint mid) {
  // Original C++ code:
  // int src_idx = static_cast<int>(mid_pt) + i / 2;
  // int prev = NormalizeIndex(i - 1, len);
  // int next = NormalizeIndex(i + 1, len);
  // dst[i] = src[src_idx] + (dst[prev] + dst[next]) / 2;

  const uint idx = mid + x;
  const int prev = NormalizeIndex((int)(2 * x), len);
  const int next = NormalizeIndex((int)(2 * x) + 2, len);

  // Remember: scratch is transposed from even!
  const int dst_prev = GetAt(scratch, y, prev);
  const int dst_next = GetAt(scratch, y, next);

  // Transpose this result, too!
  PutAt(scratch, y, 2 * x + 1, GetAt(src, idx, y) + (dst_prev + dst_next) / 2);
}

// We use one thread per pixel, and the group size (local work size)
// dictates how big the dimensions are of the 

__kernel void inv_wavelet(const __global   uchar *global_wavelet_data,
                          const __constant uint  *output_offsets,
                                __local    int   *local_data,
                                __global   char  *global_out_data)
{
  const uint local_x = get_local_id(0);
  const uint local_y = get_local_id(1);
  const int local_dim = 2 * get_local_size(1);
  const int wavelet_block_size = local_dim * local_dim;
  const int total_num_vals = 4 * get_global_size(0) * get_global_size(1);
  const __global uchar *wavelet_data = global_wavelet_data
    + output_offsets[4 * (get_global_id(2) / 6)]
	+ (get_global_id(2) % 6) * total_num_vals;

  __global char *out_data = global_out_data + get_global_id(2) * total_num_vals;

  // Grab global value and place it in local data in preparation for inv
  // wavelet transform. Data is expected to be linearized in the block.
  // Additionally, data is expected to come fresh out of a rANS compression
  // step, so we need to apply the appropriate value offset to return the data
  // to a signed value that is relevant to the wavelet transform.
  //
  // !FIXME! We're using four-byte integers here, but we can probably get away
  // with signed two-byte integers to reduce cache misses.
  {
    const uint group_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    const __global uchar *global_data = wavelet_data + group_idx * wavelet_block_size;

    const uint lidx = 4 * (local_y * get_local_size(0) + local_x);
	for (int i = 0; i < 4; ++i) {
      local_data[lidx + i] = ((int)(global_data[lidx + i])) - 128;
	}

    // We need a barrier to make sure that all threads read the data they needed
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Here we have half as many threads as we have values in our array. We arrange
  // the threads such that they are all assigned in the left half of the block that
  // we're operating on. Because the wavelet transform is separable, it means that
  // we need to do a horizontal operation followed by the same vertical operation.
  // Using this to our advantage, if we write into a transposed block, then we can
  // do the same horizontal operation twice.

  __local int *src = local_data;
  __local int *scratch = local_data + wavelet_block_size;

  const uint log_dim = 31 - clz(local_dim);
  for (uint i = 0; i < log_dim - 1; ++i) {
    const int len = 1 << (i + 1);
    const int mid = len >> 1;

    const bool use_thread = local_x < mid && local_y < len;

    // Do the even and odd values, transposed results will be in scratch.
    if (use_thread) {
      InverseWaveletEven(src, scratch, local_x, local_y, len, mid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (use_thread) {
      InverseWaveletOdd(src, scratch, local_x, local_y, len, mid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Now that the transposed results of one pass are in scratch,
    // we can do the same operation on them to get the final result back into src...

    if (use_thread) {
      InverseWaveletEven(scratch, src, local_x, local_y, len, mid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (use_thread) {
      InverseWaveletOdd(scratch, src, local_x, local_y, len, mid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const int len = 1 << log_dim;
  const int mid = len >> 1;
  InverseWaveletEven(src, scratch, local_x, local_y, len, mid);
  InverseWaveletEven(src, scratch, local_x, local_y + get_local_size(1), len, mid);

  barrier(CLK_LOCAL_MEM_FENCE);

  InverseWaveletOdd(src, scratch, local_x, local_y, len, mid);
  InverseWaveletOdd(src, scratch, local_x, local_y + get_local_size(1), len, mid);

  barrier(CLK_LOCAL_MEM_FENCE);

  InverseWaveletEven(scratch, src, local_x, local_y, len, mid);
  InverseWaveletEven(scratch, src, local_x, local_y + get_local_size(1), len, mid);

  barrier(CLK_LOCAL_MEM_FENCE);

  InverseWaveletOdd(scratch, src, local_x, local_y, len, mid);
  InverseWaveletOdd(scratch, src, local_x, local_y + get_local_size(1), len, mid);

  barrier(CLK_LOCAL_MEM_FENCE);

  // Write final value back into global memory
  {
    const uint local_stride = local_dim;
    const uint global_stride = local_stride * get_num_groups(0);

	const uint odd_column = local_x & 0x1;
	const uint ly = 2 * local_y + odd_column;
	const uint lx = 4 * (local_x >> 1);
    const uint lidx = ly * local_stride + lx;

	const uint gy = get_group_id(1) * local_dim + 2 * local_y + odd_column;
	const uint gx = 4 * (get_global_id(0) >> 1);
    const uint gidx = gy * global_stride + gx;

	for (int i = 0; i < 4; ++i) {
	    out_data[gidx + i] = (char)(local_data[lidx + i]);
	}
  }
}
