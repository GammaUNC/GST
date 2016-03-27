#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

int NormalizeIndex(int idx, int range) {
  return abs(idx - (int)(idx >= range) * (idx - range + 2));
}

int GetAt(__local int *ptr, uint x, uint y) {
  uint stride = get_local_size(1);
  return ptr[y * stride + x];
}

void PutAt(__local int *ptr, uint x, uint y, int val) {
  uint stride = get_local_size(1);
  ptr[y * stride + x] = val;
}

void InverseWaveletEven(__local int *src, __local int *scratch,
                        uint x, uint y, uint len, uint mid) {
  // Original C++ code:
  // int prev = static_cast<int>(mid_pt) + NormalizeIndex(i - 1, len) / 2;
  // int next = static_cast<int>(mid_pt) + NormalizeIndex(i + 1, len) / 2;
  // dst[i] = src[i / 2] - (src[prev] + src[next] + 2) / 4;

  uint idx = 2 * x;
  int prev = mid + NormalizeIndex((int)(idx) - 1, (int)len) / 2;
  int next = mid + NormalizeIndex((int)(idx) + 1, (int)len) / 2;

  int src_prev = GetAt(src, (uint)prev, y);
  int src_next = GetAt(src, (uint)next, y);

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

  uint idx = mid + x;
  int prev = NormalizeIndex((int)(2 * x), len);
  int next = NormalizeIndex((int)(2 * x) + 2, len);

  // Remember: scratch is transposed from even!
  int dst_prev = GetAt(scratch, y, prev);
  int dst_next = GetAt(scratch, y, next);

  // Transpose this result, too!
  PutAt(scratch, y, 2 * x + 1, GetAt(src, idx, y) + (dst_prev + dst_next) / 2);
}

// We use one thread per pixel, and the group size (local work size)
// dictates how big the dimensions are of the 

__kernel void inv_wavelet(const __constant uchar *wavelet_data,
					      const int value_offset,
                          __local int *local_data,
                          __global char *out_data)
{
  int local_dim = get_local_size(1);
  int wavelet_block_size = local_dim * local_dim;

  uint local_x = get_local_id(0);
  uint local_y = get_local_id(1);

  // Grab global value and place it in local data in preparation for inv
  // wavelet transform. Data is expected to be linearized in the block.
  // Additionally, data is expected to come fresh out of a rANS compression
  // step, so we need to apply the appropriate value offset to return the data
  // to a signed value that is relevant to the wavelet transform.
  //
  // !FIXME! We're using four-byte integers here, but we can probably get away
  // with signed two-byte integers to reduce cache misses.
  {
    uint group_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    const __constant uchar *global_data = wavelet_data + group_idx * wavelet_block_size;

    uint lidx = 2 * (local_y * get_local_size(0) + local_x);
    local_data[lidx] = ((int)(global_data[lidx])) + value_offset;
    local_data[lidx + 1] = ((int)(global_data[lidx + 1])) + value_offset;

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

  uint log_dim = 31 - clz(local_dim);
  for (uint i = 0; i < log_dim; ++i) {
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

  // Write final value back into global memory
  {
    uint local_stride = get_local_size(1);
    uint global_stride = local_stride * get_num_groups(0);

    uint lidx = local_y * local_stride + 2 * local_x;
    uint gidx = get_global_id(1) * global_stride + 2 * get_global_id(0);

    out_data[gidx] = (char)(local_data[lidx]);
    out_data[gidx + 1] = (char)(local_data[lidx + 1]);
  }
}
