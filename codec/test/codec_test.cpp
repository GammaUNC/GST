#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "codec.h"
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
  EXPECT_TRUE(GenTC::TestDXT(gTestEnv->GetContext(), fname.c_str(), NULL));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gTestEnv = dynamic_cast<OpenCLEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new OpenCLEnvironment));
  assert(NULL != gTestEnv);
  return RUN_ALL_TESTS();
}
