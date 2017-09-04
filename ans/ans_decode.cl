#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#define ANS_TABLE_SIZE_LOG  11
#define ANS_TABLE_SIZE      (1 << ANS_TABLE_SIZE_LOG)
#define NUM_ENCODED_SYMBOLS 256
#define ANS_DECODER_K       (1 << 4)
#define ANS_DECODER_L       (ANS_DECODER_K * ANS_TABLE_SIZE)

typedef struct AnsTableEntry_Struct {
	ushort freq;
	ushort cum_freq;
	uchar  symbol;
} AnsTableEntry;

#ifdef GENTC_APPLE
static void ans_decode_single(
  const    __global   AnsTableEntry *table,
  volatile __local    uint          *normalization_mask,
  const               uint           stream_group_id,
  const    __global   uchar         *data,
           __global   uchar         *out_stream);
#endif

void ans_decode_single(const    __global   AnsTableEntry *table,
                       volatile __local    uint          *normalization_mask,
                       const               uint           stream_group_id,
                       const    __global   uchar         *data,
                       __global   uchar         *out_stream) {
  uint offset = ((const __global uint *)data)[stream_group_id];
  uint state = ((const __global uint *)(data + offset) - get_local_size(0))[get_local_id(0)];
  uint next_to_read = (offset - (get_local_size(0) * 4)) / 2;
  const __global ushort *stream_data = (const __global ushort *)data;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < NUM_ENCODED_SYMBOLS; ++i) {
    const uint symbol = state & (ANS_TABLE_SIZE - 1);
    const __global AnsTableEntry *entry = table + symbol;
    state = (state >> ANS_TABLE_SIZE_LOG) * entry->freq
      - entry->cum_freq + symbol;

    // Set the bit for this invocation...
    const uint normalization_bit =
      ((uint)(state < ANS_DECODER_L)) << get_local_id(0);
    atomic_or(normalization_mask, normalization_bit);

    barrier(CLK_LOCAL_MEM_FENCE);

    // If we need to renormalize, then do so...
    const uint total_to_read = popcount(*normalization_mask);
    if (normalization_bit != 0) {
      const uint up_to_me_mask = normalization_bit - 1;
      uint num_to_skip = total_to_read;
      num_to_skip -= popcount(*normalization_mask & up_to_me_mask) + 1;
      state = (state << 16) | stream_data[next_to_read - num_to_skip - 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Clear the bit in the normalization mask...
    atomic_and(normalization_mask, ~normalization_bit);

    // Advance the read pointer by the number of shorts read
    next_to_read -= total_to_read;

    // Write the result
    // !SPEED! We might be hitting bank conflicts here, but upon trying to work
    // around this issue, there wasn't any significantly observable speedup. This
    // may be a more significant concern where we read from the stream data...
    const int gidx = (get_local_id(0) + stream_group_id * get_local_size(0)) * NUM_ENCODED_SYMBOLS;
    out_stream[gidx + NUM_ENCODED_SYMBOLS - 1 - i] = entry->symbol;
  }
}

__kernel void ans_decode(const __global   AnsTableEntry *global_table,
                         const __global   uchar         *data,
                               __global   uchar         *out_stream) {
	// Load everything into local memory
  #if 0
  __local AnsTableEntry table[ANS_TABLE_SIZE];
  for (size_t i = get_local_id(0); i < ANS_TABLE_SIZE; i += get_local_size(0)) {
    table[i].freq = global_table[i].freq;
    table[i].cum_freq = global_table[i].cum_freq;
    table[i].symbol = global_table[i].symbol;
  }
  #endif

  __local uint normalization_mask;
  if (0 == get_local_id(0)) {
    normalization_mask = 0;
  }

  ans_decode_single(global_table, &normalization_mask, get_group_id(0), data, out_stream);
}

__kernel void ans_decode_multiple(const __global   AnsTableEntry *global_table,
                                  const            uint           num_offsets,
                                  const __global   uint          *offsets,
                                  const __global   uchar         *data,
                                        __global   uchar         *out_stream) {
  const __global uint *input_offsets = offsets + num_offsets;
  const __global uint *output_offsets = offsets;
  uint id = get_group_id(0) * get_local_size(0) * NUM_ENCODED_SYMBOLS;
  
  // Binary search...
  uint low = 0;
  uint high = num_offsets - 1;
  uint x = (high + low) / 2;

  // condition:
  // output_offsets[x] <= id < output_offsets[x + 1]
  const uint iters = 32 - clz(high - low + 1);
  for (uint i = 0; i < iters; ++i) {
    uint too_high = (uint)(id < output_offsets[x]);
    uint too_low = (uint)(x < (num_offsets - 1) && output_offsets[x + 1] <= id);

    low = too_low * max(low + 1, x) + ((1 - too_low) * low);
    high = too_high * min(high - 1, x) + ((1 - too_high) * high);
    x = (high + low) >> 1;
  }

  // Load everything into local memory
  #if 0
  __local AnsTableEntry table[ANS_TABLE_SIZE];
  for (size_t i = get_local_id(0); i < ANS_TABLE_SIZE; i += get_local_size(0)) {
    uint gidx = x * ANS_TABLE_SIZE + i;
    table[i].freq = global_table[gidx].freq;
    table[i].cum_freq = global_table[gidx].cum_freq;
    table[i].symbol = global_table[gidx].symbol;
  }
  #endif

  __local uint normalization_mask;
  if (0 == get_local_id(0)) {
    normalization_mask = 0;
  }

  ans_decode_single(global_table + x * ANS_TABLE_SIZE, &normalization_mask,
                    (id - output_offsets[x]) / (get_local_size(0) * NUM_ENCODED_SYMBOLS),
                    data + input_offsets[x],
                    out_stream + output_offsets[x]);
}
