#ifndef __TCAR_PIXEL_TRAITS_H__
#define __TCAR_PIXEL_TRAITS_H__

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>

namespace GenTC {

template <unsigned NumBits>
struct SignedBits {
  static_assert(NumBits > 0, "Must have at least some bits!");
  static_assert(NumBits <= 64, "Can't hold more than 64 bits!");
  static const size_t kNumBits = NumBits;
  int64_t x;

  SignedBits(): x(0) { }
  SignedBits(int64_t _x): x(_x) {
    assert(_x < (1LL << (NumBits - 1)));
    assert(_x >= -(1LL << (NumBits - 1)));
  }
  operator int64_t() const { return x; }
};

template <unsigned NumBits>
struct UnsignedBits {
  static_assert(NumBits > 0, "Must have at least some bits!");
  static_assert(NumBits <= 64, "Can't hold more than 64 bits!");
  static const size_t kNumBits = NumBits;
  uint64_t x;

  UnsignedBits(): x(0) { }
  UnsignedBits(uint64_t _x): x(_x) {
    assert(_x < (1 << NumBits));
  }
  operator uint64_t() const { return x; }
};

////////////////////////////////////////////////////////////////////////////////

namespace PixelTraits {

////////////////////////////////////////////////////////////

template <typename T>
struct NumChannels {
  static const size_t value = 1;
};

template <class... Types>
struct NumChannels<std::tuple<Types...> > {
  static const size_t value = std::tuple_size<std::tuple<Types...> >::value;
};

////////////////////////////////////////////////////////////

template <typename T>
struct IsSigned {
  static const bool value = std::is_signed<T>::value;
};

template <unsigned NumBits>
struct IsSigned<UnsignedBits<NumBits> > {
  static const bool value = false;
};

template <unsigned NumBits>
struct IsSigned<SignedBits<NumBits> > {
  static const bool value = true;
};

////////////////////////////////////////////////////////////

template <typename T>
struct BitsUsed { };

template <unsigned NumBits>
struct BitsUsed<UnsignedBits<NumBits> > {
  static const size_t value = NumBits;
};

template <unsigned NumBits>
struct BitsUsed<SignedBits<NumBits> > {
  static const size_t value = NumBits;
};

template <typename T1, typename T2, typename T3>
struct BitsUsed<std::tuple<T1, T2, T3> > {
  static const size_t value =
    BitsUsed<T1>::value + BitsUsed<T2>::value + BitsUsed<T3>::value;
};

template <typename T1, typename T2, typename T3, typename T4>
struct BitsUsed<std::tuple<T1, T2, T3, T4> > {
  static const size_t value =
    BitsUsed<T1>::value + BitsUsed<T2>::value + BitsUsed<T3>::value + BitsUsed<T4>::value;
};

#ifdef BITS_USED_FOR_BASE_TYPE
#error "Already defined??"
#endif
#define BITS_USED_FOR_BASE_TYPE(num_bits) \
  template <>                                   \
  struct BitsUsed<int##num_bits##_t> {          \
    static const size_t value = num_bits;       \
  };                                            \
                                                \
  template <>                                   \
  struct BitsUsed<uint##num_bits##_t> {         \
    static const size_t value = num_bits;       \
  };                                            \

BITS_USED_FOR_BASE_TYPE(8)
BITS_USED_FOR_BASE_TYPE(16)
BITS_USED_FOR_BASE_TYPE(32)
BITS_USED_FOR_BASE_TYPE(64)
#undef BITS_USED_FOR_BASE_TYPE

////////////////////////////////////////////////////////////

template <typename T>
struct Max {
  static const T value = std::numeric_limits<T>::max();
};

template <unsigned NumBits>
struct Max<UnsignedBits<NumBits> > {
  static const uint64_t value = (1ULL << NumBits) - 1;
};

template <unsigned NumBits>
struct Max<SignedBits<NumBits> > {
  static const int64_t value = (1ULL << (NumBits - 1)) - 1;
};

////////////////////////////////////////////////////////////

template <typename T>
struct Min {
  static const T value = std::numeric_limits<T>::min();
};

template <unsigned NumBits>
struct Min<UnsignedBits<NumBits> > {
  static const uint64_t value = 0;
};

template <unsigned NumBits>
struct Min<SignedBits<NumBits> > {
  static const int64_t value = -(1LL << (NumBits - 1));
};

////////////////////////////////////////////////////////////

template <typename T>
struct ConvertUnsigned {
  static T cvt(uint64_t x) {
    if (std::is_signed<T>::value) {
      static const uint64_t mask = 1ULL << (BitsUsed<T>::value - 1);
      if (mask & x) {
		    int64_t v = (-1LL & (~(mask - 1))) | x;
		    assert(v <= std::numeric_limits<T>::max());
		    assert(v >= std::numeric_limits<T>::min());
		    return static_cast<T>(v);
      }
    }
    return static_cast<T>(x);
  }
};

template <unsigned NumBits>
struct ConvertUnsigned<UnsignedBits<NumBits> > {
  static UnsignedBits<NumBits> cvt(uint64_t x) {
    return x;
  }
};

template <unsigned NumBits>
struct ConvertUnsigned<SignedBits<NumBits> > {
  static SignedBits<NumBits> cvt(uint64_t x) {
    static const uint64_t mask = 1ULL << (NumBits - 1);
    if (mask & x) {
      return (-1LL & (~(mask - 1))) | x;
    }

    return x;
  }
};

////////////////////////////////////////////////////////////

template <typename T>
struct BitPacker {
  static void pack(T p, uint8_t *dst, size_t *bit_offset) {
    const int num_bits = static_cast<int>(BitsUsed<T>::value);
    const uint64_t x = static_cast<uint64_t>(p);
    uint8_t *byte_to_write = dst + (*bit_offset / 8);
    int bits_to_write = static_cast<int>(num_bits);
    while (bits_to_write > 0) {
      const int local_offset = static_cast<int>(*bit_offset % 8);
      const int bits_to_write_in_byte = static_cast<int>(8 - local_offset);

      uint8_t bits = 0;
      if (bits_to_write_in_byte >= bits_to_write) {
        int shift = bits_to_write_in_byte - bits_to_write;
        assert(shift < 8);
        bits = static_cast<uint8_t>(x << shift);
        *bit_offset += bits_to_write;
      } else {
        int shift = bits_to_write - bits_to_write_in_byte;
        bits = static_cast<uint8_t>(x >> shift);
        *bit_offset += bits_to_write_in_byte;
      }

      assert(bits < (1 << bits_to_write_in_byte));

      *byte_to_write &= ~(((1 << bits_to_write_in_byte) - 1) - 1);
      *byte_to_write |= bits;

      bits_to_write -= bits_to_write_in_byte;
      ++byte_to_write;
    }
  }
};

template <typename T1, typename T2, typename T3>
struct BitPacker<std::tuple<T1, T2, T3> > {
  static void pack(std::tuple<T1, T2, T3> p, uint8_t *dst, size_t *bit_offset) {
    BitPacker<T1>::pack(std::get<0>(p), dst, bit_offset);
    BitPacker<T2>::pack(std::get<1>(p), dst, bit_offset);
    BitPacker<T3>::pack(std::get<2>(p), dst, bit_offset);
  }
};

template <typename T1, typename T2, typename T3, typename T4>
struct BitPacker<std::tuple<T1, T2, T3, T4> > {
  static void pack(std::tuple<T1, T2, T3, T4> p, uint8_t *dst, size_t *bit_offset) {
    BitPacker<T1>::pack(std::get<0>(p), dst, bit_offset);
    BitPacker<T2>::pack(std::get<1>(p), dst, bit_offset);
    BitPacker<T3>::pack(std::get<2>(p), dst, bit_offset);
    BitPacker<T4>::pack(std::get<3>(p), dst, bit_offset);
  }
};

////////////////////////////////////////////////////////////

template <typename T>
struct ToUnsigned {
  static uint64_t cvt(T x) {
    if (IsSigned<T>::value) {
      if (x < 0) {
        uint64_t sub_zero = static_cast<uint64_t>(-x);
        return (static_cast<uint64_t>(Max<T>::value) / 2) - sub_zero;
      } else {
        uint64_t above_zero = static_cast<uint64_t>(x);
        return (static_cast<uint64_t>(Max<T>::value) / 2) + above_zero;
      }
    }
    return static_cast<uint64_t>(x);
  }
};

template <typename T1, typename T2, typename T3>
struct ToUnsigned<std::tuple<T1, T2, T3> > {
  static std::tuple<T1, T2, T3> cvt(std::tuple<T1, T2, T3> p) {
    std::tuple<T1, T2, T3> result;
    std::get<0>(result) = ToUnsigned<T1>::cvt(std::get<0>(p));
    std::get<1>(result) = ToUnsigned<T2>::cvt(std::get<1>(p));
    std::get<2>(result) = ToUnsigned<T3>::cvt(std::get<2>(p));
    return result;
  }
};

template <typename T1, typename T2, typename T3, typename T4>
struct ToUnsigned<std::tuple<T1, T2, T3, T4> > {
  static std::tuple<T1, T2, T3, T4> cvt(std::tuple<T1, T2, T3, T4> p) {
    std::tuple<T1, T2, T3, T4> result;
    std::get<0>(result) = ToUnsigned<T1>::cvt(std::get<0>(p));
    std::get<1>(result) = ToUnsigned<T2>::cvt(std::get<1>(p));
    std::get<2>(result) = ToUnsigned<T3>::cvt(std::get<2>(p));
    std::get<3>(result) = ToUnsigned<T4>::cvt(std::get<3>(p));
    return result;
  }
};

////////////////////////////////////////////////////////////

template <unsigned NumBits>
struct SignedTypeForBits {
  typedef SignedBits<NumBits> Ty;
};

template <>
struct SignedTypeForBits<8> {
  typedef int8_t Ty;
};

template <>
struct SignedTypeForBits<16> {
  typedef int16_t Ty;
};

template <>
struct SignedTypeForBits<32> {
  typedef int32_t Ty;
};

template <>
struct SignedTypeForBits<64> {
  typedef int64_t Ty;
};

////////////////////////////////////////////////////////////

template <unsigned NumBits>
struct UnsignedTypeForBits {
  typedef UnsignedBits<NumBits> Ty;
};

template <>
struct UnsignedTypeForBits<8> {
  typedef uint8_t Ty;
};

template <>
struct UnsignedTypeForBits<16> {
  typedef uint16_t Ty;
};

template <>
struct UnsignedTypeForBits<32> {
  typedef uint32_t Ty;
};

template <>
struct UnsignedTypeForBits<64> {
  typedef uint64_t Ty;
};

////////////////////////////////////////////////////////////

template <typename T>
struct SignedForUnsigned {
  typedef typename SignedTypeForBits<PixelTraits::BitsUsed<T>::value>::Ty Ty;
};

template <typename T>
struct UnsignedForSigned {
  typedef typename UnsignedTypeForBits<PixelTraits::BitsUsed<T>::value>::Ty Ty;
};

}  // namespace PixelTraits

}  // namespace GenTC

#endif  //__TCAR_PIXEL_TRAITS_H__
