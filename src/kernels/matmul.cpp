#include "matmul.h"
#include "../tensor.h"

#include <immintrin.h>

namespace mk {

void matvec_bf16_ref(float* dst, const uint16_t* w, const float* x,
                     const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* row = w + static_cast<int64_t>(r) * cols;
        float acc = 0.f;
        for (int c = 0; c < cols; ++c) acc += bf16_to_f32(row[c]) * x[c];
        dst[r] = bias ? acc + bias[r] : acc;
    }
}

#if defined(__AVX512F__)

// Load 16 bf16, widen to f32: zero-extend to 32-bit then shift into the
// high half of each lane. bf16 is literally the top 16 bits of an f32.
static inline __m512 load_bf16x16(const uint16_t* p) {
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
    return _mm512_castsi512_ps(w);
}

void matvec_bf16(float* dst, const uint16_t* w, const float* x,
                 const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* row = w + static_cast<int64_t>(r) * cols;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            acc0 = _mm512_fmadd_ps(load_bf16x16(row + c), _mm512_loadu_ps(x + c), acc0);
            acc1 = _mm512_fmadd_ps(load_bf16x16(row + c + 16), _mm512_loadu_ps(x + c + 16), acc1);
        }
        float acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        for (; c < cols; ++c) acc += bf16_to_f32(row[c]) * x[c];
        dst[r] = bias ? acc + bias[r] : acc;
    }
}

#else

void matvec_bf16(float* dst, const uint16_t* w, const float* x,
                 const float* bias, int rows, int cols) {
    matvec_bf16_ref(dst, w, x, bias, rows, cols);
}

#endif

void matvec_f32(float* dst, const float* w, const float* x,
                const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = w + static_cast<int64_t>(r) * cols;
        float acc = 0.f;
        for (int c = 0; c < cols; ++c) acc += row[c] * x[c];
        dst[r] = bias ? acc + bias[r] : acc;
    }
}

} // namespace mk
