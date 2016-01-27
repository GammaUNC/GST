#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "ans.h"
#include "gtest/gtest.h"

TEST(Codec, CanEncodeValues) {
  ans::Options opts;
  opts.b = (1 << 16);
  opts.k = 2;
  opts.M = 32;
  opts.Fs = { 2, 1, 1 };
  for (auto ty : { ans::eType_rANS, ans::eType_tANS }) {
    opts.type = ty;
    std::unique_ptr<ans::Encoder> enc = ans::Encoder::Create(opts);

    uint32_t x = 0;
    ans::BitWriter w(reinterpret_cast<unsigned char *>(&x));

    uint32_t state = enc->GetState();
    enc->Encode(0, &w);
    EXPECT_LE(state, enc->GetState());
    uint32_t state1 = enc->GetState();

    enc->Encode(1, &w);
    EXPECT_LE(state1, enc->GetState());
    uint32_t state2 = enc->GetState();

    enc->Encode(0, &w);
    EXPECT_LE(state2, enc->GetState());
    uint32_t state3 = enc->GetState();

    enc->Encode(2, &w);
    EXPECT_LE(state3, enc->GetState());
    uint32_t state4 = enc->GetState();

    // We really shouldn't have written anything to x...
    EXPECT_EQ(0, x);

    ans::BitReader r(reinterpret_cast<unsigned char *>(&x));
    std::unique_ptr<ans::Decoder> dec = ans::Decoder::Create(state4, opts);
    EXPECT_EQ(dec->Decode(&r), 2);
    EXPECT_EQ(dec->GetState(), state3);
    EXPECT_EQ(dec->Decode(&r), 0);
    EXPECT_EQ(dec->GetState(), state2);
    EXPECT_EQ(dec->Decode(&r), 1);
    EXPECT_EQ(dec->GetState(), state1);
    EXPECT_EQ(dec->Decode(&r), 0);
    EXPECT_EQ(dec->GetState(), state);
  }
}

TEST(Codec, CanEncodeValuesWithRenormalization) {
  const int num_copies = 6;

  ans::Options opts;
  opts.b = (1 << 8);
  opts.k = 2;
  opts.M = 32;
  opts.Fs = { 2, 1, 1 };
  for (auto ty : { ans::eType_rANS, ans::eType_tANS }) {
    opts.type = ty;

    std::unique_ptr<ans::Encoder> enc = ans::Encoder::Create(opts);

    std::vector<uint32_t> states;
    states.reserve(num_copies * 4);

    uint32_t x = 0;
    unsigned char *xptr = reinterpret_cast<unsigned char *>(&x);
    ans::BitWriter w(xptr);

    for (int i = 0; i < num_copies; ++i) {
      states.push_back(enc->GetState());
      enc->Encode(0, &w);

      states.push_back(enc->GetState());
      enc->Encode(1, &w);

      states.push_back(enc->GetState());
      enc->Encode(0, &w);

      states.push_back(enc->GetState());
      enc->Encode(2, &w);
    }

    ASSERT_EQ(w.BytesWritten(), 4);
    uint32_t final = enc->GetState();

    // We really should have written something to x...
    EXPECT_NE(0, x);

    // Reverse the bytes in x since we need to unwind a stack...
    std::swap(xptr[0], xptr[3]);
    std::swap(xptr[1], xptr[2]);

    ans::BitReader r(xptr);
    std::unique_ptr<ans::Decoder> dec = ans::Decoder::Create(final, opts);

    auto rev_iter = states.rbegin();
    for (int i = 0; i < num_copies; ++i) {
      EXPECT_EQ(dec->Decode(&r), 2);
      EXPECT_EQ(dec->GetState(), *(rev_iter++));
      EXPECT_EQ(dec->Decode(&r), 0);
      EXPECT_EQ(dec->GetState(), *(rev_iter++));
      EXPECT_EQ(dec->Decode(&r), 1);
      EXPECT_EQ(dec->GetState(), *(rev_iter++));
      EXPECT_EQ(dec->Decode(&r), 0);
      EXPECT_EQ(dec->GetState(), *(rev_iter++));
    }
  }
}

TEST(Codec, CanEncodeValuesWithRenormalization_Robust) {
  // Make sure to initialize the random number generator
  // with a known value in order to make it deterministic
  srand(0);
  struct TestCase {
    const ans::EType type;
    const int num_symbols;
    const std::vector<uint32_t> F;
  } test_cases[] = {
    { ans::eType_rANS, 1024,  { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 } },
    { ans::eType_rANS, 65535, { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 } },
    { ans::eType_rANS, 1024,  { 80, 15 } },
    { ans::eType_rANS, 257,   { 80, 15, 10, 7, 5, 3, 3, 33, 2, 2, 2, 2, 1 } },
    { ans::eType_rANS, 10,    { 80, 15, 10, 7, 5, 3, 3, 3, 3, 22, 2, 2, 1 } },
    { ans::eType_rANS, 1,     { 80, 15 } },
    { ans::eType_tANS, 1024,  { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 } },
    { ans::eType_tANS, 65535, { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 } },
    { ans::eType_tANS, 1024,  { 80, 15 } },
    { ans::eType_tANS, 257,   { 80, 15, 10, 7, 5, 3, 3, 33, 2, 2, 2, 2, 1 } },
    { ans::eType_tANS, 10,    { 80, 15, 10, 7, 5, 3, 3, 3, 3, 22, 2, 2, 1 } },
    { ans::eType_tANS, 1,     { 80, 15 } },
    { ans::eType_tANS, 1024,  { 2, 10, 7, 5, 53, 3, 33, 2, 2, 1, 1, 1, 1, 1 } }
  };

  ans::Options opts;
  opts.b = 256;
  opts.k = 2;

  size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
  for (size_t i = 0; i < num_cases; ++i) {
    const TestCase &test = test_cases[i];
    opts.Fs = test.F;
    opts.M = std::accumulate(test.F.begin(), test.F.end(), 0);

    size_t bytes_written = 0;
    std::vector<unsigned char> stream(10, 0);

    std::vector<uint32_t> symbols;
    symbols.reserve(test.num_symbols);

    std::vector<uint32_t> states;
    states.reserve(test.num_symbols);

    opts.type = test.type;
    std::unique_ptr<ans::Encoder> enc = ans::Encoder::Create(opts);
    for (int i = 0; i < test.num_symbols; ++i) {
      int r = rand() % opts.M;
      int symbol = 0;
      int freq = 0;
      for (auto f : opts.Fs) {
        freq += f;
        if (r < freq) {
          break;
        }
        symbol++;
      }
      ASSERT_LT(symbol, test.F.size());
      symbols.push_back(symbol);

      ans::BitWriter w(stream.data() + bytes_written);
      states.push_back(enc->GetState());
      enc->Encode(symbol, &w);

      bytes_written += w.BytesWritten();
      if (bytes_written > (stream.size() / 2)) {
        stream.resize(stream.size() * 2);
      }
    }

    uint32_t final_state = enc->GetState();
    std::unique_ptr<ans::Decoder> dec = ans::Decoder::Create(final_state, opts);

    stream.resize(bytes_written);
    std::reverse(stream.begin(), stream.end());
    std::reverse(symbols.begin(), symbols.end());
    std::reverse(states.begin(), states.end());

    ans::BitReader r(stream.data());
    for (int i = 0; i < test.num_symbols; ++i) {
      EXPECT_EQ(dec->Decode(&r), symbols[i]);
      EXPECT_EQ(dec->GetState(), states[i]);
    }
  }
}

TEST(Codec, CanInterleaveIdenticalStreams) {
  // Make sure to initialize the random number generator
  // with a known value in order to make it deterministic
  srand(0);
  const int num_symbols = 1024;
  struct TestCase {
    const std::vector<uint32_t> F;
    std::vector<uint32_t> symbols;
    std::vector<uint32_t> states;
  } test_cases[] = {
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
  };

  const size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

  ans::Options opts[num_cases];
  for (size_t i = 0; i < num_cases; ++i) {
    opts[i].b = 256;
    opts[i].k = 2;
    opts[i].Fs = test_cases[i].F;
    opts[i].M = std::accumulate(opts[i].Fs.begin(), opts[i].Fs.end(), 0);
  }

  for (auto ty : { ans::eType_rANS, ans::eType_tANS }) {
    std::vector<std::unique_ptr<ans::Encoder> > encoders;
    encoders.reserve(num_cases);

    for (size_t i = 0; i < num_cases; ++i) {
      opts[i].type = ty;

      TestCase &test = test_cases[i];
      test.symbols.reserve(num_symbols);
      test.states.reserve(num_symbols);
      encoders.push_back(ans::Encoder::Create(opts[i]));
    }

    size_t bytes_written = 0;
    std::vector<unsigned char> stream(10, 0);
    for (size_t i = 0; i < num_symbols * num_cases; ++i) {
      size_t test_idx = i % num_cases;
      TestCase &test = test_cases[test_idx];

      int r = rand() % opts[test_idx].M;
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
      test.symbols.push_back(symbol);

      ans::BitWriter w(stream.data() + bytes_written);
      test.states.push_back(encoders[test_idx]->GetState());
      encoders[test_idx]->Encode(symbol, &w);

      bytes_written += w.BytesWritten();
      if (bytes_written > (stream.size() / 2)) {
        stream.resize(stream.size() * 2);
      }
    }

    std::vector<std::unique_ptr<ans::Decoder> > decoders;
    decoders.reserve(num_cases);

    for (size_t i = 0; i < num_cases; ++i) {
      TestCase &test = test_cases[i];
      std::reverse(test.symbols.begin(), test.symbols.end());
      std::reverse(test.states.begin(), test.states.end());

      uint32_t enc_state = encoders[i]->GetState();
      decoders.push_back(ans::Decoder::Create(enc_state, opts[i]));
    }

    stream.resize(bytes_written);
    std::reverse(stream.begin(), stream.end());
    ans::BitReader r(stream.data());

    for (size_t i = 0; i < num_symbols * num_cases; ++i) {
      const int test_idx = num_cases - 1 - (i % num_cases);
      const TestCase &test = test_cases[test_idx];
      EXPECT_EQ(decoders[test_idx]->Decode(&r), test.symbols[i / num_cases]);
      EXPECT_EQ(decoders[test_idx]->GetState(), test.states[i / num_cases]);
    }
  }
}

TEST(Codec, CanInterleaveStreamsWithDifferentDistributions) {
  // Make sure to initialize the random number generator
  // with a known value in order to make it deterministic
  srand(0);
  const int num_symbols = 1024;
  struct TestCase {
    const std::vector<uint32_t> F;
    std::vector<uint32_t> symbols;
    std::vector<uint32_t> states;
  } test_cases[] = {
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 3, 14, 7, 5, 5, 3, 13, 2, 2, 2, 1, 8, 1, 1 }, { }, { } },
    { { 80, 10, 7, 5, 53, 3, 33, 2, 2, 1, 1, 1, 1, 1 }, { }, { } },
    { { 2, 10, 7, 5, 53, 3, 33, 2, 2, 1, 1, 1, 1, 1 }, { }, { } },
  };

  const size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

  ans::Options opts[num_cases];

  for (size_t i = 0; i < num_cases; ++i) {
    opts[i].b = 256;
    opts[i].k = 2;
    opts[i].Fs = test_cases[i].F;
    opts[i].M = std::accumulate(opts[i].Fs.begin(), opts[i].Fs.end(), 0);
  }

  for (auto ty : { ans::eType_rANS, ans::eType_tANS }) {

    std::vector<std::unique_ptr<ans::Encoder> > encoders;
    encoders.reserve(num_cases);

    for (size_t i = 0; i < num_cases; ++i) {
      opts[i].type = ty;

      TestCase &test = test_cases[i];
      test.symbols.reserve(num_symbols);
      test.states.reserve(num_symbols);
      encoders.push_back(ans::Encoder::Create(opts[i]));
    }

    size_t bytes_written = 0;
    std::vector<unsigned char> stream(10, 0);
    for (size_t i = 0; i < num_symbols * num_cases; ++i) {
      size_t test_idx = i % num_cases;
      TestCase &test = test_cases[test_idx];

      int r = rand() % opts[test_idx].M;
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
      test.symbols.push_back(symbol);

      ans::BitWriter w(stream.data() + bytes_written);
      test.states.push_back(encoders[test_idx]->GetState());
      encoders[test_idx]->Encode(symbol, &w);

      bytes_written += w.BytesWritten();
      if (bytes_written > (stream.size() / 2)) {
        stream.resize(stream.size() * 2);
      }
    }

    std::vector<std::unique_ptr<ans::Decoder> > decoders;
    decoders.reserve(num_cases);

    for (size_t i = 0; i < num_cases; ++i) {
      TestCase &test = test_cases[i];
      std::reverse(test.symbols.begin(), test.symbols.end());
      std::reverse(test.states.begin(), test.states.end());

      uint32_t enc_state = encoders[i]->GetState();
      decoders.push_back(ans::Decoder::Create(enc_state, opts[i]));
    }

    stream.resize(bytes_written);
    std::reverse(stream.begin(), stream.end());
    ans::BitReader r(stream.data());

    for (size_t i = 0; i < num_symbols * num_cases; ++i) {
      const int test_idx = num_cases - 1 - (i % num_cases);
      const TestCase &test = test_cases[test_idx];
      EXPECT_EQ(decoders[test_idx]->Decode(&r), test.symbols[i / num_cases]);
      EXPECT_EQ(decoders[test_idx]->GetState(), test.states[i / num_cases]);
    }
  }
}

TEST(Codec, CanInterleaveStreamsWithDifferentSymbolsFromDistributions) {
  // Make sure to initialize the random number generator
  // with a known value in order to make it deterministic
  srand(0);
  struct TestCase {
    const std::vector<uint32_t> F;
    std::vector<uint32_t> symbols;
    std::vector<uint32_t> states;
  } test_cases[] = {
    { { 80, 15, 10, 7, 5, 3, 3, 3, 3, 2, 2, 2, 2, 1 }, { }, { } },
    { { 3, 14, 7, 5, 5, 3, 13, 2, 2, 2, 1, 8, 1, 1 }, { }, { } },
    { { 80, 10, 7, 5, 53, 3, 33, 2, 2, 1, 1, 1, 1, 1 }, { }, { } },
    { { 2, 10, 7, 5, 53, 3, 33, 2, 2, 1, 1, 1, 1, 1 }, { }, { } },
  };

  const size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
  const size_t num_symbols[num_cases] = { 1024, 3, 14, 256 };

  ans::Options opts[num_cases];
  for (size_t i = 0; i < num_cases; ++i) {
    opts[i].b = 256;
    opts[i].k = 2;
    opts[i].Fs = test_cases[i].F;
    opts[i].M = std::accumulate(opts[i].Fs.begin(), opts[i].Fs.end(), 0);
  }

  for (auto ty : { ans::eType_rANS, ans::eType_tANS }) {
    std::vector<std::unique_ptr<ans::Encoder> > encoders;
    encoders.reserve(num_cases);

    size_t max_num_symbols = 0;
    for (size_t i = 0; i < num_cases; ++i) {
      opts[i].type = ty;

      TestCase &test = test_cases[i];
      test.symbols.reserve(num_symbols[i]);
      test.states.reserve(num_symbols[i]);
      encoders.push_back(ans::Encoder::Create(opts[i]));
      max_num_symbols = std::max(max_num_symbols, num_symbols[i]);
    }

    // Still a single stream...
    size_t bytes_written = 0;
    std::vector<unsigned char> stream(10, 0);

    for (size_t i = 0; i < max_num_symbols; ++i) {
      for (size_t j = 0; j < num_cases; ++j) {
        if (i >= num_symbols[j]) {
          continue;
        }

        TestCase &test = test_cases[j];

        int r = rand() % opts[j].M;
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
        test.symbols.push_back(symbol);

        ans::BitWriter w(stream.data() + bytes_written);
        test.states.push_back(encoders[j]->GetState());
        encoders[j]->Encode(symbol, &w);

        bytes_written += w.BytesWritten();
        if (bytes_written > (stream.size() / 2)) {
          stream.resize(stream.size() * 2);
        }
      }
    }

    std::vector<std::unique_ptr<ans::Decoder> > decoders;
    decoders.reserve(num_cases);

    for (size_t i = 0; i < num_cases; ++i) {
      uint32_t enc_state = encoders[i]->GetState();
      decoders.push_back(ans::Decoder::Create(enc_state, opts[i]));
    }

    stream.resize(bytes_written);
    std::reverse(stream.begin(), stream.end());
    ans::BitReader r(stream.data());

    for (size_t i = 0; i < max_num_symbols; ++i) {
      size_t symbol_idx = max_num_symbols - 1 - i;
      for (size_t j = 0; j < num_cases; ++j) {
        const size_t test_idx = num_cases - 1 - j;
        if (symbol_idx >= num_symbols[test_idx]) {
          continue;
        }

        const TestCase &test = test_cases[test_idx];
        EXPECT_EQ(decoders[test_idx]->Decode(&r), test.symbols[symbol_idx]);
        EXPECT_EQ(decoders[test_idx]->GetState(), test.states[symbol_idx]);
      }
    }

    for (size_t j = 0; j < num_cases; ++j) {
      test_cases[j].symbols.clear();
      test_cases[j].states.clear();
    }
  }
}
