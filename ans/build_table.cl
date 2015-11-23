#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void build_table(__read_only uint *src,
					      __constant uint num_symbols,
                          __global ushort *table_frequencies,
						  __global ushort *table_cumulative_frequencies,
						  __global uchar *table_symbol)
{
  uint id = get_global_id(0);

  if (id < num_symbols) {
	table_symbol[id] = id;
	table_cumulative_frequencies[id] = id * 2;
	table_frequencies[id] = id * 3;
  }
}
