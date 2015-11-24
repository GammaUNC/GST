#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void build_table(const __constant uint *frequencies,
						  const __constant uint *cumulative_frequencies,
					      const uint num_symbols,
                          __global ushort *table_frequencies,
						  __global ushort *table_cumulative_frequencies,
						  __global uchar *table_symbols)
{
  uint id = get_global_id(0);

  // First initialize everything to zero...
  table_frequencies[id] = 0;
  table_cumulative_frequencies[id] = 0;
  table_symbols[id] = 0;

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  // Initialize table
  if (id < num_symbols) {
	int cum_freq = cumulative_frequencies[id];
	int freq = frequencies[id];
	table_frequencies[cum_freq] = freq;
	table_cumulative_frequencies[cum_freq] = freq;
	table_symbols[cum_freq] = id;
  }

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  int lgM = 31 - clz(cumulative_frequencies[num_symbols - 1] + frequencies[num_symbols - 1]);
  for (int i = 0; i < lgM; ++i) {
	int x = table_cumulative_frequencies[id];
	int halfx = x >> 1;
	if ( halfx > 0 ) {
	  table_cumulative_frequencies[id] = halfx;
	  table_cumulative_frequencies[id + halfx] = x - halfx;
	  table_symbols[id + halfx] = table_symbols[id];
	  table_frequencies[id + halfx] = table_frequencies[id];
	}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
  table_cumulative_frequencies[id] = cumulative_frequencies[table_symbols[id]];
}
