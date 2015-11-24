#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void build_table(const __constant uint *frequencies,
						  const __constant uint *cumulative_frequencies,
					      const uint num_symbols,
                          __global ushort *table_frequencies,
						  __global ushort *table_cumulative_frequencies,
						  __global uchar *table_symbols)
{
  uint id = get_global_id(0);

  // Binary search...
  uint low = 0;
  uint high = num_symbols - 1;
  uint x = (high + low) / 2;

  // condition:
  // cumulative_frequencies[x] <= id < cumulative_frequencies[x + 1]
  int lgM = 31 - clz(cumulative_frequencies[num_symbols - 1] + frequencies[num_symbols - 1]);
  for (int i = 0; i < lgM; ++i) {
    uint too_high = (uint)(id < cumulative_frequencies[x]);
	uint too_low = (uint)(x < num_symbols - 1 && cumulative_frequencies[x + 1] <= id);

	low = (too_low) * max(low + 1, x) + ((1 - too_low) * low);
	high = (too_high) * min(high - 1, x) + ((1 - too_high) * high);
    x = (high + low) / 2;
  }

  // First initialize everything to zero...
  table_frequencies[id] = frequencies[x];
  table_cumulative_frequencies[id] = cumulative_frequencies[x];
  table_symbols[id] = x;
}
