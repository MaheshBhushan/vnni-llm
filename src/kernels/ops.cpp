#include "ops.h"
#include "../tensor.h"

#include <cmath>

namespace mk {

void rmsnorm(float* y, const float* x, const float* weight, int n, float eps) {
    // HF Qwen2RMSNorm computes the mean in the input dtype's f32 upcast; plain
    // double-free f32 accumulation matches to ~1e-7 at n=2048.
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * weight[i];
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void silu_mul(float* gate, const float* up, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        gate[i] = g / (1.f + std::exp(-g)) * up[i];
    }
}

void rope(float* head, int head_dim, int pos, float theta_base) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = std::pow(theta_base, -2.f * i / head_dim);
        float angle = pos * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float a = head[i], b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
    }
}

void bf16_row_to_f32(float* dst, const uint16_t* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}

} // namespace mk
