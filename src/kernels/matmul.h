#pragma once
#include <cstdint>

namespace mk {

// dst[r] = dot(w[r, :], x) (+ bias[r] if bias) for r in [0, rows).
// w is row-major [rows, cols].
void matvec_bf16(float* dst, const uint16_t* w, const float* x,
                 const float* bias, int rows, int cols);

// Scalar reference used by unit tests.
void matvec_bf16_ref(float* dst, const uint16_t* w, const float* x,
                     const float* bias, int rows, int cols);

// f32 weights variant (activation-side small matrices, tests).
void matvec_f32(float* dst, const float* w, const float* x,
                const float* bias, int rows, int cols);

} // namespace mk
