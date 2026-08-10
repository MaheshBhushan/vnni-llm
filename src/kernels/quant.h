#pragma once
#include <cstdint>

namespace mk {

// Block quantization formats, layout-compatible with llama.cpp's Q4_0 / Q8_0
// so benchmarks compare like-for-like.
constexpr int QK = 32;  // elements per block

// 4-bit weights: per-block f16 scale + 32 nibbles (packed 2/byte).
// Dequant: w[i] = d * (nibble[i] - 8), nibble in [0,15]. 18 bytes/block.
struct BlockQ4_0 {
    uint16_t d;             // f16 scale
    uint8_t qs[QK / 2];     // low nibble = even index, high nibble = odd index
};
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must be 18 bytes");

// 8-bit activations: per-block f16 scale + 32 int8 + precomputed sum(q).
// The 16-bit sum lets the VNNI kernel apply the -8 nibble bias per block
// without a second pass. 36 bytes/block (llama.cpp stores 34; the extra
// 2 bytes are our own optimization and never touch disk).
struct BlockQ8_0 {
    uint16_t d;             // f16 scale
    int16_t sum;            // sum of the 32 int8 values
    int8_t qs[QK];
};
static_assert(sizeof(BlockQ8_0) == 36, "Q8_0 block must be 36 bytes");

// Quantize `n` f32 weights (n % 32 == 0) into `out` (n/32 blocks).
void quantize_q4_0(BlockQ4_0* out, const float* src, int n);

// Quantize `n` f32 activations into `out` (n/32 blocks).
void quantize_q8_0(BlockQ8_0* out, const float* src, int n);

// Dequantize one Q4_0 block row of `n` elements back to f32 (embedding lookup).
void dequantize_q4_0_row(float* dst, const BlockQ4_0* src, int n);

// dst[r] = dot(weight_row_r, x) (+bias) for r in [0, rows).
// w is `rows` consecutive Q4_0 rows of `cols` elements (cols/32 blocks each);
// x is pre-quantized to Q8_0 (cols/32 blocks). Reference + VNNI variants.
void matvec_q4_0_q8_0_ref(float* dst, const BlockQ4_0* w, const BlockQ8_0* x,
                          const float* bias, int rows, int cols);
void matvec_q4_0_q8_0(float* dst, const BlockQ4_0* w, const BlockQ8_0* x,
                      const float* bias, int rows, int cols);

} // namespace mk
