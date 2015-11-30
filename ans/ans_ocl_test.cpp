#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "ans_encode.h"
#include "ans_ocl.h"
#include "bits.h"
#include "histogram.h"

using ans::OpenCLDecoder;

static class OpenCLEnvironment : public ::testing::Environment {
public:
  OpenCLEnvironment() : ::testing::Environment() { is_setup = false;  }
  virtual ~OpenCLEnvironment() { }
  virtual void SetUp() {
    _ctx = gpu::InitializeOpenCL(false);
    _device = gpu::GetAllDevicesForContext(_ctx).front();
    is_setup = true;
  }

  virtual void TearDown() {
    gpu::ShutdownOpenCL(_ctx);
    is_setup = false;
  }

  cl_context GetContext() const { assert(is_setup);  return _ctx; }
  cl_device_id GetDevice() const { assert(is_setup);  return _device; }

private:
  bool is_setup;
  cl_context _ctx;
  cl_device_id _device;
} *gTestEnv;

TEST(ANS_OpenCL, Initialization) {
  std::vector<int> F = { 3, 2, 1, 4, 3 };
  OpenCLDecoder decoder(gTestEnv->GetContext(), gTestEnv->GetDevice(), F, 1);

  std::vector<int> normalized_F = ans::GenerateHistogram(F, ans::kANSTableSize);

  std::vector<cl_uchar> expected_symbols(ans::kANSTableSize, 0);
  std::vector<cl_ushort> expected_frequencies(ans::kANSTableSize, 0);
  std::vector<cl_ushort> expected_cumulative_frequencies(ans::kANSTableSize, 0);

  int sum = 0;
  for (size_t i = 0; i < normalized_F.size(); ++i) {
    for (size_t j = 0; j < normalized_F[i]; ++j) {
      expected_symbols[sum + j] = static_cast<cl_uchar>(i);
      expected_frequencies[sum + j] = normalized_F[i];
      expected_cumulative_frequencies[sum + j] = sum;
    }
    sum += normalized_F[i];
  }
  ASSERT_EQ(sum, ans::kANSTableSize);

  std::vector<cl_uchar> symbols = std::move(decoder.GetSymbols());
  std::vector<cl_ushort> frequencies = std::move(decoder.GetFrequencies());
  std::vector<cl_ushort> cumulative_frequencies = std::move(decoder.GetCumulativeFrequencies());

  ASSERT_EQ(symbols.size(), expected_symbols.size());
  for (size_t i = 0; i < symbols.size(); ++i) {
    EXPECT_EQ(expected_symbols[i], symbols[i]) << "Symbols differ at index " << i;
  }

  ASSERT_EQ(frequencies.size(), expected_frequencies.size());
  for (size_t i = 0; i < frequencies.size(); ++i) {
    EXPECT_EQ(expected_frequencies[i], frequencies[i]) << "Frequencies differ at index " << i;
  }

  ASSERT_EQ(cumulative_frequencies.size(), expected_cumulative_frequencies.size());
  for (size_t i = 0; i < cumulative_frequencies.size(); ++i) {
    EXPECT_EQ(expected_cumulative_frequencies[i], cumulative_frequencies[i])
      << "Cumulative frequencies differ at index " << i;
  }
}

TEST(ANS_OpenCL, TableRebuilding) {
  std::vector<int> F = { 3, 2, 1, 4, 3, 406 };
  std::vector<int> new_F = { 80, 300, 2, 14, 1, 1, 1, 20 };
  std::vector<int> normalized_F = ans::GenerateHistogram(new_F, ans::kANSTableSize);

  OpenCLDecoder decoder(gTestEnv->GetContext(), gTestEnv->GetDevice(), F, 1);
  decoder.RebuildTable(new_F);

  std::vector<cl_uchar> expected_symbols(ans::kANSTableSize, 0);
  std::vector<cl_ushort> expected_frequencies(ans::kANSTableSize, 0);
  std::vector<cl_ushort> expected_cumulative_frequencies(ans::kANSTableSize, 0);

  int sum = 0;
  for (size_t i = 0; i < normalized_F.size(); ++i) {
    for (size_t j = 0; j < normalized_F[i]; ++j) {
      expected_symbols[sum + j] = static_cast<cl_uchar>(i);
      expected_frequencies[sum + j] = normalized_F[i];
      expected_cumulative_frequencies[sum + j] = sum;
    }
    sum += normalized_F[i];
  }
  ASSERT_EQ(sum, ans::kANSTableSize);

  ASSERT_EQ(expected_symbols.size(), ans::kANSTableSize);
  ASSERT_EQ(expected_frequencies.size(), ans::kANSTableSize);
  ASSERT_EQ(expected_cumulative_frequencies.size(), ans::kANSTableSize);

  std::vector<cl_uchar> symbols = std::move(decoder.GetSymbols());
  std::vector<cl_ushort> frequencies = std::move(decoder.GetFrequencies());
  std::vector<cl_ushort> cumulative_frequencies = std::move(decoder.GetCumulativeFrequencies());

  ASSERT_EQ(symbols.size(), expected_symbols.size());
  for (size_t i = 0; i < symbols.size(); ++i) {
    EXPECT_EQ(expected_symbols[i], symbols[i]) << "Symbols differ at index " << i;
  }

  ASSERT_EQ(frequencies.size(), expected_frequencies.size());
  for (size_t i = 0; i < frequencies.size(); ++i) {
    EXPECT_EQ(expected_frequencies[i], frequencies[i]) << "Frequencies differ at index " << i;
  }

  ASSERT_EQ(cumulative_frequencies.size(), expected_cumulative_frequencies.size());
  for (size_t i = 0; i < cumulative_frequencies.size(); ++i) {
    EXPECT_EQ(expected_cumulative_frequencies[i], cumulative_frequencies[i])
      << "Cumulative frequencies differ at index " << i;
  }
}

TEST(ANS_OpenCL, DecodeSingleNPOTStream) {
  std::vector<uint32_t> F = { 12, 14, 17, 1, 1, 2, 372 };

  const size_t num_symbols = 256;
  std::vector<int> symbols;
  symbols.reserve(num_symbols);

  ans::Encoder<1 << 16, 1 << 16> encoder(F);
  std::vector<uint8_t> stream(10);

  size_t bytes_written = 0;
  srand(0);
  for (int i = 0; i < num_symbols; ++i) {
    uint32_t M = std::accumulate(F.begin(), F.end(), 0);
    int r = rand() % M;
    int symbol = 0;
    int freq = 0;
    for (auto f : F) {
      freq += f;
      if (r < freq) {
        break;
      }
      symbol++;
    }
    ASSERT_LT(symbol, F.size());
    symbols.push_back(symbol);

    ans::BitWriter w(stream.data() + bytes_written);
    encoder.Encode(symbol, w);

    bytes_written += w.BytesWritten();
    if (bytes_written > (stream.size() / 2)) {
      stream.resize(stream.size() * 2);
    }
  }
  stream.resize(bytes_written);
  ASSERT_EQ(symbols.size(), num_symbols);

  ans::OpenCLDecoder decoder(gTestEnv->GetContext(), gTestEnv->GetDevice(), F, 1);
  std::vector<std::vector<uint32_t> > decoded_symbols;
  std::vector<uint32_t> encoded_states(1, encoder.GetState());

  decoder.Decode(&decoded_symbols, encoded_states, stream);

  ASSERT_EQ(decoded_symbols.size(), 1);
  ASSERT_EQ(decoded_symbols[0].size(), num_symbols);
  for (size_t i = 0; i < num_symbols; ++i) {
    EXPECT_EQ(decoded_symbols[0][i], symbols[i]) << "Symbols differ at index " << i;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gTestEnv = dynamic_cast<OpenCLEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new OpenCLEnvironment));
  assert(NULL != gTestEnv);
  return RUN_ALL_TESTS();
}
