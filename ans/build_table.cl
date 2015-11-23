#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void build_table(const __constant uint *src,
					      const uint num_symbols,
                          __global ushort *table_frequencies,
						  __global ushort *table_cumulative_frequencies,
						  __global uchar *table_symbol)
{
  uint id = get_local_id(0); // * get_local_size(0) + get_local_id(0);

  if (id < num_symbols) {
	table_symbol[id] = (ushort)id;
	table_cumulative_frequencies[id] = (ushort)(id * 2);
	table_frequencies[id] = (ushort)(id * 3);
  }
}
