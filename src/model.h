#pragma once
#include "kernels/quant.h"
#include "kv_cache.h"
#include "safetensors.h"
#include "tensor.h"
#include "threadpool.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mk {

struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;
    bool qkv_bias;          // Qwen2: true, Llama: false
    int eos_token_id;

    static ModelConfig from_json_file(const std::string& path);
};

// A projection matrix [rows, cols], stored either as a BF16 mmap view or as
// owned Q4_0 blocks depending on the model's quantization mode.
struct QWeight {
    int rows = 0, cols = 0;
    const uint16_t* bf16 = nullptr;         // non-null in BF16 mode
    std::vector<BlockQ4_0> q4;              // populated in Q4_0 mode
};

// Per-layer weights. Small vectors (norms, biases) are always owned f32.
struct LayerWeights {
    std::vector<float> input_norm, post_attn_norm;
    std::vector<float> q_bias, k_bias, v_bias;
    QWeight wq, wk, wv, wo;
    QWeight w_gate, w_up, w_down;
};

class Model {
public:
    // quantize=true converts all projection matrices to Q4_0 (activations are
    // quantized to Q8_0 on the fly). false keeps BF16 for exact verification.
    explicit Model(const std::string& model_dir, bool quantize = false, int n_threads = 1);

    const ModelConfig& config() const { return cfg_; }
    bool quantized() const { return quantize_; }

    // Runs one token at position kv.seq_len(), appends to the cache and
    // advances it. If logits != nullptr, writes vocab_size logits.
    void forward(int token, KVCache& kv, float* logits);

    // Optional per-layer trace for verification against the HF dump.
    std::function<void(const std::string&, int, const float*, int)> trace;

private:
    // dst[0..w.rows) = w @ x (+bias). Threaded; dispatches on storage mode.
    // In Q4 mode x is quantized into x_q8_ first (caller passes x length = w.cols).
    void matmul(float* dst, const QWeight& w, const float* x, const float* bias);
    QWeight load_qweight(const std::string& name, int rows, int cols);

    ModelConfig cfg_;
    bool quantize_;
    std::unique_ptr<SafetensorsFile> st_;
    std::unique_ptr<ThreadPool> pool_;
    std::vector<LayerWeights> layers_;
    std::vector<float> final_norm_;
    const uint16_t* embed_;       // [vocab, hidden] bf16, for token lookup
    std::vector<uint16_t> embed_owned_;  // heap copy in quant mode (mmap released)
    QWeight lm_head_;             // tied to embeddings; Q4 copy in quant mode

    // scratch buffers reused across tokens
    std::vector<float> x_, xb_, q_, k_, v_, attn_out_, gate_, up_, scores_;
    std::vector<BlockQ8_0> x_q8_;   // quantized activation scratch (max cols)
};

} // namespace mk
