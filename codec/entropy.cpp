#include "entropy.h"

#include <cmath>
#include <numeric>
#include <iostream>

#include "ans_ocl.h"
#include "data_stream.h"

namespace GenTC {

ShortEncoder::EncodeUnit::ReturnType
ShortEncoder::Encode::Run(const ShortEncoder::EncodeUnit::ArgType &in) const {
  assert(in->size() > 0);

  // Extract values larger than or equal to 255...
  std::vector<uint16_t> big_vals;
  std::vector<uint8_t> vals;
  vals.reserve(in->size());

  for (size_t i = 0; i < in->size(); ++i) {
    int16_t x = in->at(i);
    if (std::abs(x) > 127) {
      big_vals.push_back(static_cast<uint16_t>(x));
      vals.push_back(128);
    } else {
      assert(static_cast<uint8_t>(x) != 128);
      vals.push_back(static_cast<uint8_t>(x));
    }
  }

  // Num threads
  const size_t num_symbols = vals.size();
  const size_t num_threads = num_symbols / _symbols_per_thread;
  const size_t num_thread_groups = num_threads / ans::ocl::kThreadsPerEncodingGroup;

  assert(num_threads * _symbols_per_thread == num_symbols);
  assert(num_thread_groups * ans::ocl::kThreadsPerEncodingGroup == num_threads);

  std::vector<uint32_t> counts(256, 0);
  for (auto v : vals) {
    assert(static_cast<uint32_t>(v) < 256);
    counts[v]++;
  }

#ifndef NDEBUG
  for (auto v : vals) {
    assert(counts[v] < (1 << 16));
    assert(0 < counts[v]);  // This'll cause problems if it's not the case
  }
#endif

  std::vector<uint8_t> encoded_symbols;
  std::vector<uint32_t> encoded_symbol_offsets;
  encoded_symbol_offsets.reserve(num_thread_groups);

  ans::Options opts = ans::ocl::GetOpenCLOptions(counts);

  size_t symbols_encoded = 0;
  while (symbols_encoded < num_symbols) {
    auto start = vals.begin() + symbols_encoded;
    auto end = vals.begin() + symbols_encoded +
      _symbols_per_thread * ans::ocl::kThreadsPerEncodingGroup;
    std::vector<uint8_t> symbols_for_group(start, end);

    std::vector<uint8_t> group =
      ans::EncodeInterleaved(symbols_for_group, opts,
                             ans::ocl::kThreadsPerEncodingGroup);

    encoded_symbols.insert(encoded_symbols.end(), group.begin(), group.end());
    encoded_symbol_offsets.push_back(static_cast<uint32_t>(encoded_symbols.size()));
    symbols_encoded += _symbols_per_thread * ans::ocl::kThreadsPerEncodingGroup;
  }

  assert(encoded_symbol_offsets.size() == num_thread_groups);
  assert(symbols_encoded == num_symbols);

  // Write header...
  DataStream hdr;
  for (size_t i = 0; i < counts.size(); ++i) {
    hdr.WriteShort(static_cast<uint16_t>(counts[i]));
  }

  hdr.WriteShort(static_cast<uint16_t>(big_vals.size()));
  for (auto big_val : big_vals) {
    hdr.WriteShort(static_cast<uint16_t>(big_val));
  }

  hdr.WriteShort(static_cast<uint16_t>(encoded_symbol_offsets.size()));
  for (auto offset : encoded_symbol_offsets) {
    hdr.WriteShort(static_cast<uint16_t>(offset));
  }

  std::vector<uint8_t> *result = new std::vector<uint8_t>;
  result->insert(result->end(), hdr.GetData().begin(), hdr.GetData().end());
  result->insert(result->end(), encoded_symbols.begin(), encoded_symbols.end());
  return std::move(std::unique_ptr<std::vector<uint8_t> >(result));
}

ShortEncoder::DecodeUnit::ReturnType
ShortEncoder::Decode::Run(const ShortEncoder::DecodeUnit::ArgType &in) const {
  // Read header...
  DataStream hdr(*(in.get()));

  std::vector<uint32_t> counts;
  counts.reserve(256);
  for (size_t i = 0; i < 256; ++i) {
    counts.push_back(hdr.ReadShort());
  }

  uint32_t num_big_vals = hdr.ReadShort();
  std::vector<uint16_t> big_vals;
  big_vals.reserve(num_big_vals);
  for (size_t i = 0; i < num_big_vals; ++i) {
    big_vals.push_back(hdr.ReadShort());
  }

  size_t num_offsets = hdr.ReadShort();
  std::vector<uint16_t> offsets;
  offsets.reserve(num_offsets);
  for (size_t i = 0; i < num_offsets; ++i) {
    offsets.push_back(hdr.ReadShort());
  }

  const size_t num_symbols =
    num_offsets * ans::ocl::kThreadsPerEncodingGroup * _symbols_per_thread;

  ans::Options opts = ans::ocl::GetOpenCLOptions(counts);

  std::vector<uint8_t> symbols;
  size_t last_offset = hdr.BytesRead();
  size_t symbols_read = 0;
  size_t group_idx = 0;
  while (symbols_read < num_symbols) {
    size_t offset = last_offset + offsets[group_idx];
    std::vector<uint8_t> interleaved_stream(in->begin() + last_offset,
                                            in->begin() + offset);

    size_t symbols_to_read = ans::ocl::kThreadsPerEncodingGroup * _symbols_per_thread;
    std::vector<uint8_t> interleaved_symbols =
      ans::DecodeInterleaved(interleaved_stream, symbols_to_read, opts,
                             ans::ocl::kThreadsPerEncodingGroup);

    symbols.insert(symbols.end(), interleaved_symbols.begin(), interleaved_symbols.end());
    assert(interleaved_symbols.size() == symbols_to_read);

    last_offset = offset;
    symbols_read += interleaved_symbols.size();
    group_idx++;
  }

  assert(group_idx == offsets.size());
  assert(symbols_read == num_symbols);
  assert(symbols.size() == num_symbols);

  // Convert the symbols back to their representation...
  std::vector<int16_t> *result = new std::vector<int16_t>;
  result->reserve(num_symbols);

  uint32_t big_val_idx = 0;
  for (size_t i = 0; i < num_symbols; ++i) {
    if (symbols[i] == 128) {
      result->push_back(static_cast<int16_t>(big_vals[big_val_idx++]));
    } else {
      result->push_back(static_cast<int16_t>(static_cast<int8_t>(symbols[i])));
    }
  }

  return std::move(std::unique_ptr<std::vector<int16_t> >(result));
}

ByteEncoder::Base::ReturnType
ByteEncoder::EncodeBytes::Run(const ByteEncoder::Base::ArgType &in) const {
  std::vector<uint32_t> counts(256, 0);

  for (size_t i = 0; i < in->size(); ++i) {
    counts[in->at(i)]++;
  }

  // Determine size
  uint32_t non_zero_counts = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (0 != counts[i]) {
      if(i != non_zero_counts) {
        assert(!"Zero-count symbols are not supported!");
        break;
      }

      non_zero_counts++;
    }
  }

  counts.resize(non_zero_counts);

  std::vector<size_t> offsets;

  const size_t num_symbols = in->size();
  ans::Options opts = ans::ocl::GetOpenCLOptions(counts);

  std::vector<uint8_t> encoded_symbols;
  size_t num_encoded_symbols = 0;
  while (num_encoded_symbols < num_symbols) {
    size_t num_symbols_to_encode = 
      ans::ocl::kThreadsPerEncodingGroup * _symbols_per_thread;

    std::vector<uint8_t> symbols_to_encode(in->begin() + num_encoded_symbols,
                                           in->begin() + num_encoded_symbols + num_symbols_to_encode);

    std::vector<uint8_t> encoded_symbols = 
      ans::EncodeInterleaved(symbols_to_encode, opts,
                             ans::ocl::kThreadsPerEncodingGroup);

    offsets.push_back(encoded_symbols.size());
    encoded_symbols.insert(encoded_symbols.end(), encoded_symbols.begin(), encoded_symbols.end());
    num_encoded_symbols += num_symbols_to_encode;
  }

  DataStream hdr;
  hdr.WriteByte(non_zero_counts);
  for (auto c : counts) {
    assert(static_cast<uint32_t>(c) < (1 << 16));
    hdr.WriteShort(c);
  }

  hdr.WriteByte(static_cast<uint8_t>(offsets.size()));
  for (auto off : offsets) {
    assert(off < (1 << 16));
    hdr.WriteShort(static_cast<uint16_t>(off));
  }

  std::vector<uint8_t> *result = new std::vector<uint8_t>;
  result->insert(result->end(), hdr.GetData().begin(), hdr.GetData().end());
  result->insert(result->end(), encoded_symbols.begin(), encoded_symbols.end());

  return std::move(std::unique_ptr<std::vector<uint8_t> >(result));
}

ByteEncoder::Base::ReturnType
ByteEncoder::DecodeBytes::Run(const ByteEncoder::Base::ArgType &in) const {
  DataStream hdr(*(in.get()));
  uint8_t num_unique_symbols = hdr.ReadByte();

  std::vector<uint32_t> counts;
  counts.reserve(num_unique_symbols);

  for (size_t i = 0; i < num_unique_symbols; ++i) {
    counts.push_back(hdr.ReadShort());
  }

  uint32_t num_offsets = hdr.ReadByte();

  std::vector<uint32_t> offsets;
  offsets.reserve(num_offsets);

  for (size_t i = 0; i < num_offsets; ++i) {
    offsets.push_back(hdr.ReadShort());
  }

  ans::Options opts = ans::ocl::GetOpenCLOptions(counts);

  std::vector<uint8_t> *result = new std::vector<uint8_t>;
  const size_t num_symbols = num_offsets * ans::ocl::kThreadsPerEncodingGroup * _symbols_per_thread;
  size_t symbols_decoded = 0;
  size_t group_idx = 0;
  size_t last_offset = hdr.BytesRead();
  while (symbols_decoded < num_symbols) {
    size_t offset = last_offset + offsets[group_idx++];
    auto start = in->begin() + last_offset;
    auto end = in->begin() + offset;
    std::vector<uint8_t> data(start, end);

    size_t symbols_to_read = ans::ocl::kThreadsPerEncodingGroup * _symbols_per_thread;
    std::vector<uint8_t> decoded = 
      ans::DecodeInterleaved(data, symbols_to_read, opts,
                             ans::ocl::kThreadsPerEncodingGroup);

    result->insert(result->end(), decoded.begin(), decoded.end());
    symbols_decoded += decoded.size();
    last_offset = offset;
  }

  assert(symbols_decoded == num_symbols);

  return std::move(std::unique_ptr<std::vector<uint8_t> >(result));
}

}  // namespace GenTC
