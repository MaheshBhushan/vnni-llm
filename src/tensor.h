#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace mk {

enum class DType { F32, F16, BF16, I64, Q4_0, Q8_0 };

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        case DType::I64:  return "I64";
        case DType::Q4_0: return "Q4_0";
        case DType::Q8_0: return "Q8_0";
    }
    return "?";
}

// Bytes for `n` elements. Quantized types are only valid for n % 32 == 0.
inline size_t dtype_bytes(DType d, size_t n) {
    switch (d) {
        case DType::F32:  return n * 4;
        case DType::F16:  return n * 2;
        case DType::BF16: return n * 2;
        case DType::I64:  return n * 8;
        case DType::Q4_0: return (n / 32) * 18;  // 2-byte f16 scale + 16 bytes of nibbles
        case DType::Q8_0: return (n / 32) * 34;  // 2-byte f16 scale + 32 int8
    }
    return 0;
}

// Non-owning view over tensor memory (owner is the mmap or an arena).
struct TensorView {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const void* data = nullptr;

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    const float* f32() const { return static_cast<const float*>(data); }
};

inline float bf16_to_f32(uint16_t x) {
    uint32_t y = static_cast<uint32_t>(x) << 16;
    float f;
    __builtin_memcpy(&f, &y, 4);
    return f;
}

inline float f16_to_f32(uint16_t h) {
    // Scalar IEEE half → float; hot paths use F16C intrinsics instead.
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);  // flush denormals
    if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

} // namespace mk
