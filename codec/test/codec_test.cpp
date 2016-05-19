#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "encoder.h"
#include "decoder.h"
#include "dxt_image.h"
#include "test_config.h"

static class OpenCLEnvironment : public ::testing::Environment {
public:
  OpenCLEnvironment() : ::testing::Environment() { is_setup = false;  }
  virtual ~OpenCLEnvironment() { }
  virtual void SetUp() {
    _ctx = std::move(gpu::GPUContext::InitializeOpenCL(false));
    // Make sure to always initialize random number generator for
    // deterministic tests
    srand(0);
    GenTC::InitializeDecoder(_ctx);
    is_setup = true;
  }

  virtual void TearDown() {
    _ctx = nullptr;
    is_setup = false;
  }

  const std::unique_ptr<gpu::GPUContext> &GetContext() const { assert(is_setup);  return _ctx; }

private:
  bool is_setup;
  std::unique_ptr<gpu::GPUContext> _ctx;
} *gTestEnv;

TEST(GenTC, CanCompressAndDecompressImage) {
  std::string dir(CODEC_TEST_DIR);
  std::string fname = dir + std::string("/") + std::string("test1.png");

  GenTC::DXTImage dxt_img(fname.c_str(), NULL);
  std::vector<uint8_t> cmp_data = std::move(GenTC::CompressDXT(dxt_img));
  GenTC::DXTImage cmp_img = std::move(GenTC::DecompressDXT(gTestEnv->GetContext(), cmp_data));

  const std::vector<GenTC::PhysicalDXTBlock> &blks = dxt_img.PhysicalBlocks();
  for (size_t i = 0; i < blks.size(); ++i) {
    EXPECT_EQ(blks[i].dxt_block, cmp_img.PhysicalBlocks()[i].dxt_block) << "Index: " << i;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gTestEnv = dynamic_cast<OpenCLEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new OpenCLEnvironment));
  assert(NULL != gTestEnv);
  return RUN_ALL_TESTS();
}
