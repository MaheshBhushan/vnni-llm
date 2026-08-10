#include "sampler.h"
#include "kernels/ops.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace mk {

Sampler::Sampler(const SamplerConfig& cfg) : cfg_(cfg) {
    rng_state_ = cfg.seed ? cfg.seed
        : static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
}

float Sampler::next_uniform() {
    // xorshift64*
    uint64_t x = rng_state_;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state_ = x;
    return static_cast<float>((x * 0x2545F4914F6CDD1Dull) >> 40) / static_cast<float>(1ull << 24);
}

int Sampler::sample(float* logits, int vocab) {
    if (cfg_.temperature <= 0.f) {
        int best = 0;
        for (int i = 1; i < vocab; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    float inv_t = 1.f / cfg_.temperature;
    for (int i = 0; i < vocab; ++i) logits[i] *= inv_t;
    softmax(logits, vocab);

    // top-p: sort candidate indices by probability, keep the smallest prefix
    // with cumulative mass >= top_p, renormalize, sample.
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
    float cum = 0.f;
    int cutoff = vocab;
    for (int i = 0; i < vocab; ++i) {
        cum += logits[idx[i]];
        if (cum >= cfg_.top_p) { cutoff = i + 1; break; }
    }
    float r = next_uniform() * cum;
    float acc = 0.f;
    for (int i = 0; i < cutoff; ++i) {
        acc += logits[idx[i]];
        if (acc >= r) return idx[i];
    }
    return idx[cutoff - 1];
}

} // namespace mk
