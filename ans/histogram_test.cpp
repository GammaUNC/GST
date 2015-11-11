#include "gtest/gtest.h"

#include "histogram.h"

template<typename T, typename U> ::testing::AssertionResult
VectorsAreEqual(const std::vector<T> &vec_one,
                const std::vector<U> &vec_two) {
  if (vec_one.size() != vec_two.size()) {
    return ::testing::AssertionFailure() << "Vectors of different size!";
  }

  bool failed = false;
  ::testing::AssertionResult result = ::testing::AssertionSuccess();
  for (size_t i = 0; i < vec_one.size(); ++i) {
    if (vec_one[i] != static_cast<U>(vec_two[i])) {
      if (!failed) {
        result = ::testing::AssertionFailure();
        failed = true;
      } 
      result << std::endl;
      result <<
        "Vectors differ at element " << i << " -- " <<
        "vec_one[" << i << "]: " << vec_one[i] << ", " <<
        "vec_two[" << i << "]: " << vec_two[i];
    }
  }

  return result;
}                                           

TEST(Histogram, HandlesAlreadyNormalized) {
  std::vector<int> counts(10);
  for (int j = 1; j < 50; ++j) {
    for (int i = 0; i < 10; ++i) {
      counts[i] = j;
    }

    std::vector<int> hist;
    ans::GenerateHistogram(&hist, counts, 10*j);
    EXPECT_TRUE(VectorsAreEqual(hist, counts));
  }
}

TEST(Histogram, HandlesEmptyFreqs) {
  std::vector<int> counts(10, 0);
  std::vector<int> hist;
#ifndef NDEBUG
  ASSERT_DEATH(ans::GenerateHistogram(&hist, counts, 10), "No symbols have any frequency!");
#else
  ans::GenerateHistogram(&hist, counts, 10);
  EXPECT_TRUE(VectorsAreEqual(hist, std::vector<int>()));
#endif
}

TEST(Histogram, HandlesNULL) {
  std::vector<int> counts(10, 0);
#ifndef NDEBUG
  ASSERT_DEATH(ans::GenerateHistogram(NULL, counts, 10), "NULL passed for output!");
#else
  ASSERT_NO_FATAL_FAILURE(ans::GenerateHistogram(NULL, counts, 10));
#endif
}

TEST(Histogram, HandlesImproperTargetSum) {
  std::vector<int> counts(10, 0); 
 std::vector<int> hist;
#ifndef NDEBUG
  EXPECT_DEATH(ans::GenerateHistogram(&hist, counts, 0), "Improper target sum for frequencies!");
  EXPECT_DEATH(ans::GenerateHistogram(&hist, counts, -4), "Improper target sum for frequencies!");
#else
  ASSERT_NO_FATAL_FAILURE(ans::GenerateHistogram(&hist, counts, 0));
  ASSERT_NO_FATAL_FAILURE(ans::GenerateHistogram(&hist, counts, -4));
#endif
}

TEST(Histogram, ProperlyDistributesPOTFreqs) {
  std::vector<int> counts = { 1, 1, 2 };
  std::vector<int> hist;
  ans::GenerateHistogram(&hist, counts, 256);

  std::vector<int> expected = { 64, 64, 128 };
  EXPECT_TRUE(VectorsAreEqual(hist, expected));
}

TEST(Histogram, ProperlyDistributesFreqs) {
  std::vector<int> counts = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  std::vector<int> hist;
  ans::GenerateHistogram(&hist, counts, 256);

  std::vector<int> expected = { 5, 9, 14, 19, 23, 28, 33, 37, 42, 46 };
  EXPECT_TRUE(VectorsAreEqual(hist, expected));
}

TEST(Histogram, ProperlyDistributesFreqsNPOT) {
  std::vector<int> counts = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  std::vector<int> hist;
  ans::GenerateHistogram(&hist, counts, 11);

  std::vector<int> expected = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 };
  EXPECT_TRUE(VectorsAreEqual(hist, expected));
}
