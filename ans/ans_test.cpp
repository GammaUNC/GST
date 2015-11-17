#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "ans_encode.h"
#include "ans_decode.h"
#include "gtest/gtest.h"

TEST(Codec, CanEncodeValues) {
  std::vector<uint32_t> F = { 2, 1, 1 };
  ans::Encoder<(1 << 16), 2> enc(F);

  uint32_t x = 0;
  ans::BitWriter w(reinterpret_cast<unsigned char *>(&x));

  uint32_t state = enc.GetState();
  enc.Encode(0, w);
  EXPECT_LE(state, enc.GetState());
  uint32_t state1 = enc.GetState();

  enc.Encode(1, w);
  EXPECT_LE(state1, enc.GetState());
  uint32_t state2 = enc.GetState();

  enc.Encode(0, w);
  EXPECT_LE(state2, enc.GetState());
  uint32_t state3 = enc.GetState();

  enc.Encode(2, w);
  EXPECT_LE(state3, enc.GetState());
  uint32_t state4 = enc.GetState();

  // We really shouldn't have written anything to x...
  EXPECT_EQ(0, x);

  ans::BitReader r(reinterpret_cast<unsigned char *>(&x));
  ans::Decoder<(1 << 16), 2> dec(state4, F);
  EXPECT_EQ(dec.Decode(r), 2);
  EXPECT_EQ(dec.GetState(), state3);
  EXPECT_EQ(dec.Decode(r), 0);
  EXPECT_EQ(dec.GetState(), state2);
  EXPECT_EQ(dec.Decode(r), 1);
  EXPECT_EQ(dec.GetState(), state1);
  EXPECT_EQ(dec.Decode(r), 0);
  EXPECT_EQ(dec.GetState(), state);
}

TEST(Codec, CanEncodeValuesWithRenormalization) {
  const int num_copies = 6;
  std::vector<uint32_t> F = { 2, 1, 1 };
  ans::Encoder<(1 << 8), 2> enc(F);

  std::vector<uint32_t> states;
  states.reserve(num_copies * 4);

  uint32_t x = 0;
  unsigned char *xptr = reinterpret_cast<unsigned char *>(&x);
  ans::BitWriter w(xptr);

  for (int i = 0; i < num_copies; ++i) {
    states.push_back(enc.GetState());
    enc.Encode(0, w);

    states.push_back(enc.GetState());
    enc.Encode(1, w);

    states.push_back(enc.GetState());
    enc.Encode(0, w);

    states.push_back(enc.GetState());
    enc.Encode(2, w);
  }

  ASSERT_EQ(w.BytesWritten(), 4);
  uint32_t final = enc.GetState();

  // We really should have written something to x...
  EXPECT_NE(0, x);

  // Reverse the bytes in x since we need to unwind a stack...
  std::swap(xptr[0], xptr[3]);
  std::swap(xptr[1], xptr[2]);

  ans::BitReader r(xptr);
  ans::Decoder<(1 << 8), 2> dec(final, F);

  auto rev_iter = states.rbegin();
  for (int i = 0; i < num_copies; ++i) {
    EXPECT_EQ(dec.Decode(r), 2);
    EXPECT_EQ(dec.GetState(), *(rev_iter++));
    EXPECT_EQ(dec.Decode(r), 0);
    EXPECT_EQ(dec.GetState(), *(rev_iter++));
    EXPECT_EQ(dec.Decode(r), 1);
    EXPECT_EQ(dec.GetState(), *(rev_iter++));
    EXPECT_EQ(dec.Decode(r), 0);
    EXPECT_EQ(dec.GetState(), *(rev_iter++));
  }
}

TEST(Codec, CanEncodeValuesWithRenormalization_Robust) {
  // Make sure to initialize the random number generator
  // with a known value in order to make it deterministic
  srand(0);
  struct TestCase {
    const int num_symbols;
    const std::vector<uint32_t> F;
  } test_cases[] = {
    1024, { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 },
    1024, { 80, 15 },
    257, { 80, 15, 10, 7, 5, 3, 3, 33, 2, 2, 2, 2, 1 },
    10, { 80, 15, 10, 7, 5, 3, 3, 3, 3, 22, 2, 2, 1 },
    1, { 80, 15 }
  };

  size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
  for (size_t i = 0; i < num_cases; ++i) {
    const TestCase &test = test_cases[i];
    uint32_t M = std::accumulate(test.F.begin(), test.F.end(), 0);

    size_t bytes_written = 0;
    std::vector<unsigned char> stream(10, 0);

    std::vector<uint32_t> symbols;
    symbols.reserve(test.num_symbols);

    std::vector<uint32_t> states;
    states.reserve(test.num_symbols);

    ans::Encoder<256, 2> enc(test.F);
    for (int i = 0; i < test.num_symbols; ++i) {
      int r = rand() % M;
      int symbol = 0;
      int freq = 0;
      for (auto f : test.F) {
        freq += f;
        if (r < freq) {
          break;
        }
        symbol++;
      }
      ASSERT_LT(symbol, test.F.size());
      symbols.push_back(symbol);

      ans::BitWriter w(stream.data() + bytes_written);
      states.push_back(enc.GetState());
      enc.Encode(symbol, w);

      bytes_written += w.BytesWritten();
      if (bytes_written > (stream.size() / 2)) {
        stream.resize(stream.size() * 2);
      }
    }

    uint32_t final_state = enc.GetState();
    ans::Decoder<256, 2> dec(final_state, test.F);

    stream.resize(bytes_written);
    std::reverse(stream.begin(), stream.end());
    std::reverse(symbols.begin(), symbols.end());
    std::reverse(states.begin(), states.end());

    ans::BitReader r(stream.data());
    for (int i = 0; i < test.num_symbols; ++i) {
      EXPECT_EQ(dec.Decode(r), symbols[i]);
      EXPECT_EQ(dec.GetState(), states[i]);
    }
  }
}
