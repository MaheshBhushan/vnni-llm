#pragma once
#include <cstddef>
#include <cstdint>

namespace mk {

// y = x * rsqrt(mean(x^2) + eps) * weight      (x, y may alias)
void rmsnorm(float* y, const float* x, const float* weight, int n, float eps);

// In-place softmax over x[0..n)
void softmax(float* x, int n);

// gate = silu(gate) * up, elementwise, in-place into gate
void silu_mul(float* gate, const float* up, int n);

// NeoX-style rotary embedding applied in-place to one head of size head_dim
// at absolute position pos. Pairs are (i, i + head_dim/2).
void rope(float* head, int head_dim, int pos, float theta_base);

// dst[i] = bf16_to_f32(src[i])
void bf16_row_to_f32(float* dst, const uint16_t* src, int n);

} // namespace mk
