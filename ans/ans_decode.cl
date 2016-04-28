#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// This doesn't seem to make much of a difference....
// #define USE_LOCAL_TABLE

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

__kernel void ans_decode(const __constant AnsTableEntry *global_table,
						 const __global   uchar         *data,
						       __global   uchar         *out_stream)
{
#ifdef USE_LOCAL_TABLE
	// Load everything into local memory
	__local AnsTableEntry table[ANS_TABLE_SIZE];
	for (size_t i = get_local_id(0); i < ANS_TABLE_SIZE; i += get_local_size(0)) {
		table[i].freq = global_table[i].freq;
		table[i].cum_freq = global_table[i].cum_freq;
		table[i].symbol = global_table[i].symbol;
	}
#endif

	__local uint normalization_mask;
	normalization_mask = 0;

	uint offset = ((const __global uint *)data)[get_group_id(0)];
	uint state = ((const __global uint *)(data + offset) - get_local_size(0))[get_local_id(0)];
    uint next_to_read = (offset - (get_local_size(0) * 4)) / 2;
    const __global ushort *stream_data = (const __global ushort *)data;

	barrier(CLK_LOCAL_MEM_FENCE);

	uchar result[NUM_ENCODED_SYMBOLS];
	for (int i = 0; i < NUM_ENCODED_SYMBOLS; ++i) {
		const uint symbol = state & (ANS_TABLE_SIZE - 1);
#ifdef USE_LOCAL_TABLE
		const __local AnsTableEntry *entry = table + symbol;
#else
		const __constant AnsTableEntry *entry = global_table + symbol;
#endif
		state = (state >> ANS_TABLE_SIZE_LOG) * entry->freq
			- entry->cum_freq + symbol;

		// Set the bit for this invocation...
		const uint normalization_bit =
		  ((uint)(state < ANS_DECODER_L)) << get_local_id(0);
		atomic_or(&normalization_mask, normalization_bit);

		barrier(CLK_LOCAL_MEM_FENCE);

		// If we need to renormalize, then do so...
		const uint total_to_read = popcount(normalization_mask);
		if (normalization_bit != 0) {
		  const uint up_to_me_mask = normalization_bit - 1;
		  uint num_to_skip = total_to_read;
		  num_to_skip -= popcount(normalization_mask & up_to_me_mask) + 1;
		  state = (state << 16) | stream_data[next_to_read - num_to_skip - 1];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Clear the bit in the normalization mask...
		atomic_and(&normalization_mask, ~normalization_bit);

		// Advance the read pointer by the number of shorts read
		next_to_read -= total_to_read;

		// Write the result
		result[i] = entry->symbol;
	}

	// Write a staggered result to avoid bank conflicts
	int write_idx = get_local_id(0);
	for (int i = 0; i < NUM_ENCODED_SYMBOLS; ++i) {
		const int gidx = get_global_id(0) * NUM_ENCODED_SYMBOLS + (255 - write_idx);
		out_stream[gidx] = result[write_idx];
		write_idx = (write_idx + 1) & (NUM_ENCODED_SYMBOLS - 1);
	}
}
