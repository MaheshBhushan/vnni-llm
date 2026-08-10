#pragma once
#include <cstdint>

namespace mk {

struct SamplerConfig {
    float temperature = 0.7f;
    float top_p = 0.8f;
    uint64_t seed = 0;   // 0 = nondeterministic
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg);
    // Picks a token from logits[0..vocab). temperature == 0 → greedy argmax.
    int sample(float* logits, int vocab);

private:
    SamplerConfig cfg_;
    uint64_t rng_state_;
    float next_uniform();
};

} // namespace mk
