#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define ANS_TABLE_SIZE_LOG  11
#define MAX_NUM_SYMBOLS     256

typedef struct AnsTableEntry_Struct {
	ushort freq;
	ushort cum_freq;
	uchar  symbol;
} AnsTableEntry;

__kernel void build_table(const __constant uint *frequencies,
                          __global AnsTableEntry *table)
{
  __local uint num_symbols;
  __local uint cumulative_frequencies[MAX_NUM_SYMBOLS];

  // Read the number of symbols from the frequencies table...
  int lid = get_local_id(0);
  if (0 == lid) {
    num_symbols = frequencies[0];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Set the cumulative frequencies to the frequencies... if we have
  // more, then pad to zeros.
  if (lid < num_symbols) {
    cumulative_frequencies[lid] = frequencies[lid + 1];
  } else if (lid < MAX_NUM_SYMBOLS) {
    cumulative_frequencies[lid] = 0;
  }

  // Do a quick scan to build the cumulative frequencies...
  const int n = MAX_NUM_SYMBOLS;
  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < d) {
      int ai = offset * (2 * lid + 1) - 1;
      int bi = offset * (2 * lid + 2) - 1;
      cumulative_frequencies[bi] += cumulative_frequencies[ai];
    }
    offset *= 2;
  }

  if (lid == 0) {
    cumulative_frequencies[n - 1] = 0;
  }

  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < d) {
      int ai = offset * (2 * lid + 1) - 1;
      int bi = offset * (2 * lid + 2) - 1;

      uint t = cumulative_frequencies[ai];
      cumulative_frequencies[ai] = cumulative_frequencies[bi];
      cumulative_frequencies[bi] += t;
    }
  }

  // If our scan's done, we can build our portion of the table...
  barrier(CLK_LOCAL_MEM_FENCE);

  uint id = get_global_id(0);

  // Binary search...
  uint low = 0;
  uint high = num_symbols - 1;
  uint x = (high + low) / 2;

  // condition:
  // cumulative_frequencies[x] <= id < cumulative_frequencies[x + 1]
  for (int i = 0; i < ANS_TABLE_SIZE_LOG; ++i) {
    uint too_high = (uint)(id < cumulative_frequencies[x]);
	uint too_low = (uint)(x < num_symbols - 1 && cumulative_frequencies[x + 1] <= id);

	low = (too_low) * max(low + 1, x) + ((1 - too_low) * low);
	high = (too_high) * min(high - 1, x) + ((1 - too_high) * high);
    x = (high + low) / 2;
  }

  // Write results
  table[id].freq = frequencies[x + 1];
  table[id].cum_freq = cumulative_frequencies[x];
  table[id].symbol = x;
}
