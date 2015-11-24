#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "ans_ocl.h"

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
  std::vector<uint32_t> F = { 3, 2, 1, 4, 3 };
  const uint32_t M = std::accumulate(F.begin(), F.end(), 0);
  ans::OpenCLDecoder decoder(gTestEnv->GetContext(), gTestEnv->GetDevice(), F, 1);

  std::vector<cl_uchar> expected_symbols = { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4 };
  std::vector<cl_ushort> expected_frequencies = { 3, 3, 3, 2, 2, 1, 4, 4, 4, 4, 3, 3, 3 };
  std::vector<cl_ushort> expected_cumulative_frequencies = { 0, 0, 0, 3, 3, 5, 6, 6, 6, 6, 10, 10, 10 };
  ASSERT_EQ(expected_symbols.size(), M);
  ASSERT_EQ(expected_frequencies.size(), M);
  ASSERT_EQ(expected_cumulative_frequencies.size(), M);

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
  std::vector<uint32_t> F = { 3, 2, 1, 4, 3, 406 };
  std::vector<uint32_t> new_F = { 80, 300, 2, 14, 1, 1, 1, 20 };
  const uint32_t M = std::accumulate(F.begin(), F.end(), 0);
  ASSERT_EQ(std::accumulate(new_F.begin(), new_F.end(), 0), M);

  ans::OpenCLDecoder decoder(gTestEnv->GetContext(), gTestEnv->GetDevice(), F, 1);
  decoder.RebuildTable(new_F);

  std::vector<cl_uchar> expected_symbols(M, 0);
  std::vector<cl_ushort> expected_frequencies(M, 0);
  std::vector<cl_ushort> expected_cumulative_frequencies(M, 0);

  int sum = 0;
  for (size_t i = 0; i < new_F.size(); ++i) {
    for (size_t j = 0; j < new_F[i]; ++j) {
      expected_symbols[sum + j] = static_cast<cl_uchar>(i);
      expected_frequencies[sum + j] = new_F[i];
      expected_cumulative_frequencies[sum + j] = sum;
    }
    sum += new_F[i];
  }
  ASSERT_EQ(sum, M);

  ASSERT_EQ(expected_symbols.size(), M);
  ASSERT_EQ(expected_frequencies.size(), M);
  ASSERT_EQ(expected_cumulative_frequencies.size(), M);

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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gTestEnv = dynamic_cast<OpenCLEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new OpenCLEnvironment));
  assert(NULL != gTestEnv);
  return RUN_ALL_TESTS();
}