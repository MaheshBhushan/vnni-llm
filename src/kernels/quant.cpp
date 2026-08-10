#include "quant.h"
#include "../tensor.h"

#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace mk {

// Hardware half->float (F16C). ~4x faster than the scalar fallback and this is
// called millions of times per token in the quantized matmuls.
#if defined(__F16C__)
static inline float half_to_float(uint16_t h) { return _cvtsh_ss(h); }
#else
static inline float half_to_float(uint16_t h) { return f16_to_f32(h); }
#endif

void quantize_q4_0(BlockQ4_0* out, const float* src, int n) {
    const int nb = n / QK;
    for (int b = 0; b < nb; ++b) {
        const float* x = src + b * QK;
        // Symmetric quant around 0: scale by the largest-magnitude value so it
        // maps to nibble -8 (the most negative representable level).
        float amax = 0.f, max = 0.f;
        for (int i = 0; i < QK; ++i) {
            if (std::fabs(x[i]) > amax) { amax = std::fabs(x[i]); max = x[i]; }
        }
        float d = max / -8.f;
        float id = d ? 1.f / d : 0.f;
        out[b].d = f32_to_f16(d);
        for (int i = 0; i < QK / 2; ++i) {
            int8_t x0 = static_cast<int8_t>(std::lround(x[i] * id) + 8);
            int8_t x1 = static_cast<int8_t>(std::lround(x[i + QK / 2] * id) + 8);
            if (x0 > 15) x0 = 15; if (x0 < 0) x0 = 0;
            if (x1 > 15) x1 = 15; if (x1 < 0) x1 = 0;
            out[b].qs[i] = static_cast<uint8_t>(x0 | (x1 << 4));
        }
    }
}

void quantize_q8_0(BlockQ8_0* out, const float* src, int n) {
    const int nb = n / QK;
    for (int b = 0; b < nb; ++b) {
        const float* x = src + b * QK;
        float amax = 0.f;
        for (int i = 0; i < QK; ++i) amax = std::fmax(amax, std::fabs(x[i]));
        float d = amax / 127.f;
        float id = d ? 1.f / d : 0.f;
        out[b].d = f32_to_f16(d);
        int sum = 0;
        for (int i = 0; i < QK; ++i) {
            int q = static_cast<int>(std::lround(x[i] * id));
            if (q > 127) q = 127; if (q < -127) q = -127;
            out[b].qs[i] = static_cast<int8_t>(q);
            sum += q;
        }
        out[b].sum = static_cast<int16_t>(sum);
    }
}

void dequantize_q4_0_row(float* dst, const BlockQ4_0* src, int n) {
    const int nb = n / QK;
    for (int b = 0; b < nb; ++b) {
        float d = half_to_float(src[b].d);
        for (int i = 0; i < QK / 2; ++i) {
            int lo = (src[b].qs[i] & 0x0F) - 8;
            int hi = (src[b].qs[i] >> 4) - 8;
            dst[b * QK + i] = d * lo;
            dst[b * QK + i + QK / 2] = d * hi;
        }
    }
}

// -------- reference matmul --------
// Per block: sum((nibble-8) * q_i) * d_w * d_x. The nibble packing puts index
// i in the low half and i+16 in the high half, matching Q8_0's linear order.
void matvec_q4_0_q8_0_ref(float* dst, const BlockQ4_0* w, const BlockQ8_0* x,
                          const float* bias, int rows, int cols) {
    const int nb = cols / QK;
    for (int r = 0; r < rows; ++r) {
        const BlockQ4_0* wr = w + static_cast<int64_t>(r) * nb;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            int isum = 0;
            for (int i = 0; i < QK / 2; ++i) {
                int lo = (wr[b].qs[i] & 0x0F) - 8;
                int hi = (wr[b].qs[i] >> 4) - 8;
                isum += lo * x[b].qs[i] + hi * x[b].qs[i + QK / 2];
            }
            acc += isum * half_to_float(wr[b].d) * half_to_float(x[b].d);
        }
        dst[r] = bias ? acc + bias[r] : acc;
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)

static inline float hsum_f32_8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// dpbusd is unsigned*signed. We keep nibbles unsigned in [0,15] and correct the
// -8 bias analytically: sum((n-8)*q) = sum(n*q) - 8*sum(q). sum(q) is the
// precomputed per-block BlockQ8_0::sum, so no second dot product is needed.
//
// The 8 int32 lanes of each block's dpbusd result share one block scale, so we
// convert them to float and FMA into an 8-wide float accumulator directly —
// no per-block horizontal reduction. The -8*sum(q)*scale bias is a cheap scalar
// running sum. One hsum at the very end. This is the throughput-critical path.
static inline void unpack_nibbles(const BlockQ4_0& blk, __m128i nib_mask, __m256i& out) {
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.qs));
    __m128i lo = _mm_and_si128(packed, nib_mask);                    // indices 0..15
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib_mask); // indices 16..31
    out = _mm256_set_m128i(hi, lo);
}

// Processes ROWS_AT rows per pass so the loads of the shared activation block
// `q` are reused and, more importantly, ROWS_AT independent dpbusd/FMA chains
// run in parallel — hiding the ~4-cycle FMA and dpbusd latencies that
// bottleneck a single-accumulator kernel. Decode is compute-bound until this
// exposes enough ILP to reach the memory-bandwidth ceiling.
void matvec_q4_0_q8_0(float* dst, const BlockQ4_0* w, const BlockQ8_0* x,
                      const float* bias, int rows, int cols) {
    const int nb = cols / QK;
    const __m128i nib_mask = _mm_set1_epi8(0x0F);
    constexpr int R = 4;

    int r = 0;
    for (; r + R <= rows; r += R) {
        const BlockQ4_0* wr[R];
        __m256 acc[R];
        float bias_acc[R];
        for (int j = 0; j < R; ++j) {
            wr[j] = w + static_cast<int64_t>(r + j) * nb;
            acc[j] = _mm256_setzero_ps();
            bias_acc[j] = 0.f;
        }
        for (int b = 0; b < nb; ++b) {
            __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].qs));
            float xd = half_to_float(x[b].d);
            int xsum = x[b].sum;
            for (int j = 0; j < R; ++j) {
                __m256i nibbles;
                unpack_nibbles(wr[j][b], nib_mask, nibbles);
                __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), nibbles, q);
                float scale = half_to_float(wr[j][b].d) * xd;
                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(scale), acc[j]);
                bias_acc[j] += 8.f * xsum * scale;
            }
        }
        for (int j = 0; j < R; ++j) {
            float v = hsum_f32_8(acc[j]) - bias_acc[j];
            dst[r + j] = bias ? v + bias[r + j] : v;
        }
    }
    // scalar-ish tail for the remaining <R rows
    for (; r < rows; ++r) {
        const BlockQ4_0* wrow = w + static_cast<int64_t>(r) * nb;
        __m256 acc = _mm256_setzero_ps();
        float bias_acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            __m256i nibbles;
            unpack_nibbles(wrow[b], nib_mask, nibbles);
            __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].qs));
            __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), nibbles, q);
            float scale = half_to_float(wrow[b].d) * half_to_float(x[b].d);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(scale), acc);
            bias_acc += 8.f * x[b].sum * scale;
        }
        float v = hsum_f32_8(acc) - bias_acc;
        dst[r] = bias ? v + bias[r] : v;
    }
}

#else

void matvec_q4_0_q8_0(float* dst, const BlockQ4_0* w, const BlockQ8_0* x,
                      const float* bias, int rows, int cols) {
    matvec_q4_0_q8_0_ref(dst, w, x, bias, rows, cols);
}

#endif

} // namespace mk
