#include "model.h"
#include "kernels/matmul.h"
#include "kernels/ops.h"
#include "kernels/quant.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mk {

ModelConfig ModelConfig::from_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);

    std::string type = j.at("model_type").get<std::string>();
    if (type != "qwen2" && type != "llama")
        throw std::runtime_error("unsupported model_type: " + type);

    ModelConfig c{};
    c.hidden_size = j.at("hidden_size");
    c.intermediate_size = j.at("intermediate_size");
    c.n_layers = j.at("num_hidden_layers");
    c.n_heads = j.at("num_attention_heads");
    c.n_kv_heads = j.at("num_key_value_heads");
    c.head_dim = j.value("head_dim", c.hidden_size / c.n_heads);
    c.vocab_size = j.at("vocab_size");
    c.max_position_embeddings = j.at("max_position_embeddings");
    c.rms_norm_eps = j.at("rms_norm_eps");
    c.rope_theta = j.value("rope_theta", 10000.f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.qkv_bias = (type == "qwen2");
    // eos may be an int or a list; take the first.
    auto& eos = j.at("eos_token_id");
    c.eos_token_id = eos.is_array() ? eos[0].get<int>() : eos.get<int>();
    return c;
}

static std::vector<float> load_f32_vec(const SafetensorsFile& st, const std::string& name) {
    const TensorView& t = st.get(name);
    std::vector<float> out(t.numel());
    if (t.dtype == DType::BF16)
        bf16_row_to_f32(out.data(), static_cast<const uint16_t*>(t.data), out.size());
    else if (t.dtype == DType::F32)
        std::memcpy(out.data(), t.data, out.size() * 4);
    else
        throw std::runtime_error("unexpected dtype for " + name);
    return out;
}

static const uint16_t* bf16_mat(const SafetensorsFile& st, const std::string& name,
                                int64_t rows, int64_t cols) {
    const TensorView& t = st.get(name);
    if (t.dtype != DType::BF16)
        throw std::runtime_error(name + ": expected BF16");
    if (t.shape.size() != 2 || t.shape[0] != rows || t.shape[1] != cols)
        throw std::runtime_error(name + ": unexpected shape");
    return static_cast<const uint16_t*>(t.data);
}

QWeight Model::load_qweight(const std::string& name, int rows, int cols) {
    QWeight w;
    w.rows = rows;
    w.cols = cols;
    const uint16_t* bf = bf16_mat(*st_, name, rows, cols);
    if (!quantize_) { w.bf16 = bf; return w; }
    // Quantize BF16 -> Q4_0 row by row, one f32 row at a time.
    const int nb = cols / QK;
    w.q4.resize(static_cast<size_t>(rows) * nb);
    std::vector<float> row(cols);
    for (int r = 0; r < rows; ++r) {
        bf16_row_to_f32(row.data(), bf + static_cast<int64_t>(r) * cols, cols);
        quantize_q4_0(w.q4.data() + static_cast<size_t>(r) * nb, row.data(), cols);
    }
    return w;
}

Model::Model(const std::string& model_dir, bool quantize, int n_threads)
    : cfg_(ModelConfig::from_json_file(model_dir + "/config.json")),
      quantize_(quantize),
      st_(std::make_unique<SafetensorsFile>(model_dir)),
      pool_(std::make_unique<ThreadPool>(n_threads)) {
    const auto& c = cfg_;
    int q_dim = c.n_heads * c.head_dim;
    int kv_dim = c.n_kv_heads * c.head_dim;
    if (c.hidden_size % QK || c.intermediate_size % QK || q_dim % QK)
        throw std::runtime_error("dims must be divisible by 32 for Q4_0");

    embed_ = bf16_mat(*st_, "model.embed_tokens.weight", c.vocab_size, c.hidden_size);
    final_norm_ = load_f32_vec(*st_, "model.norm.weight");
    if (!c.tie_word_embeddings && st_->has("lm_head.weight"))
        throw std::runtime_error("untied lm_head not implemented yet");
    lm_head_ = load_qweight("model.embed_tokens.weight", c.vocab_size, c.hidden_size);

    layers_.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        LayerWeights& w = layers_[l];
        w.input_norm = load_f32_vec(*st_, p + "input_layernorm.weight");
        w.post_attn_norm = load_f32_vec(*st_, p + "post_attention_layernorm.weight");
        w.wq = load_qweight(p + "self_attn.q_proj.weight", q_dim, c.hidden_size);
        w.wk = load_qweight(p + "self_attn.k_proj.weight", kv_dim, c.hidden_size);
        w.wv = load_qweight(p + "self_attn.v_proj.weight", kv_dim, c.hidden_size);
        w.wo = load_qweight(p + "self_attn.o_proj.weight", c.hidden_size, q_dim);
        if (c.qkv_bias) {
            w.q_bias = load_f32_vec(*st_, p + "self_attn.q_proj.bias");
            w.k_bias = load_f32_vec(*st_, p + "self_attn.k_proj.bias");
            w.v_bias = load_f32_vec(*st_, p + "self_attn.v_proj.bias");
        }
        w.w_gate = load_qweight(p + "mlp.gate_proj.weight", c.intermediate_size, c.hidden_size);
        w.w_up = load_qweight(p + "mlp.up_proj.weight", c.intermediate_size, c.hidden_size);
        w.w_down = load_qweight(p + "mlp.down_proj.weight", c.hidden_size, c.intermediate_size);
    }

    x_.resize(c.hidden_size);
    xb_.resize(c.hidden_size);
    q_.resize(q_dim);
    k_.resize(kv_dim);
    v_.resize(kv_dim);
    attn_out_.resize(q_dim);
    gate_.resize(c.intermediate_size);
    up_.resize(c.intermediate_size);
    scores_.resize(c.max_position_embeddings);
    int max_cols = std::max({c.hidden_size, c.intermediate_size, q_dim});
    x_q8_.resize(max_cols / QK);

    // In quant mode every projection is now owned Q4; the only remaining use of
    // the mmap is token-embedding lookup. Copy those rows to the heap and
    // release the 5.7 GB mapping so it stops competing with the Q4 working set
    // for RAM (this is what was thrashing the benchmark).
    if (quantize_) {
        size_t n = static_cast<size_t>(c.vocab_size) * c.hidden_size;
        embed_owned_.assign(embed_, embed_ + n);
        embed_ = embed_owned_.data();
        st_.reset();
    }
}

void Model::matmul(float* dst, const QWeight& w, const float* x, const float* bias) {
    if (quantize_) {
        quantize_q8_0(x_q8_.data(), x, w.cols);
        const BlockQ8_0* xq = x_q8_.data();
        pool_->parallel_for(w.rows, [&](int r0, int r1) {
            const int nb = w.cols / QK;
            matvec_q4_0_q8_0(dst + r0, w.q4.data() + static_cast<size_t>(r0) * nb,
                             xq, bias ? bias + r0 : nullptr, r1 - r0, w.cols);
        });
    } else {
        pool_->parallel_for(w.rows, [&](int r0, int r1) {
            matvec_bf16(dst + r0, w.bf16 + static_cast<int64_t>(r0) * w.cols, x,
                        bias ? bias + r0 : nullptr, r1 - r0, w.cols);
        });
    }
}

void Model::forward(int token, KVCache& kv, float* logits) {
    const auto& c = cfg_;
    const int pos = kv.seq_len();
    const int hd = c.head_dim;
    const int q_dim = c.n_heads * hd;
    const int kv_dim = c.n_kv_heads * hd;
    const int group = c.n_heads / c.n_kv_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));

    if (token < 0 || token >= c.vocab_size) throw std::runtime_error("token out of range");
    if (pos >= kv.max_seq()) throw std::runtime_error("kv cache full");

    bf16_row_to_f32(x_.data(), embed_ + static_cast<int64_t>(token) * c.hidden_size, c.hidden_size);

    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& w = layers_[l];

        // --- attention block ---
        rmsnorm(xb_.data(), x_.data(), w.input_norm.data(), c.hidden_size, c.rms_norm_eps);
        matmul(q_.data(), w.wq, xb_.data(), c.qkv_bias ? w.q_bias.data() : nullptr);
        // K and V go straight into their cache slot for this position.
        float* k_slot = kv.k_at(l, pos);
        float* v_slot = kv.v_at(l, pos);
        matmul(k_slot, w.wk, xb_.data(), c.qkv_bias ? w.k_bias.data() : nullptr);
        matmul(v_slot, w.wv, xb_.data(), c.qkv_bias ? w.v_bias.data() : nullptr);

        for (int h = 0; h < c.n_heads; ++h) rope(q_.data() + h * hd, hd, pos, c.rope_theta);
        for (int h = 0; h < c.n_kv_heads; ++h) rope(k_slot + h * hd, hd, pos, c.rope_theta);

        for (int h = 0; h < c.n_heads; ++h) {
            const float* qh = q_.data() + h * hd;
            const int kvh_off = (h / group) * hd;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = kv.k_at(l, t) + kvh_off;
                float dot = 0.f;
                for (int i = 0; i < hd; ++i) dot += qh[i] * kt[i];
                scores_[t] = dot * scale;
            }
            softmax(scores_.data(), pos + 1);
            float* out = attn_out_.data() + h * hd;
            std::memset(out, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                const float* vt = kv.v_at(l, t) + kvh_off;
                const float s = scores_[t];
                for (int i = 0; i < hd; ++i) out[i] += s * vt[i];
            }
        }

        matmul(xb_.data(), w.wo, attn_out_.data(), nullptr);
        for (int i = 0; i < c.hidden_size; ++i) x_[i] += xb_[i];

        // --- MLP block ---
        rmsnorm(xb_.data(), x_.data(), w.post_attn_norm.data(), c.hidden_size, c.rms_norm_eps);
        matmul(gate_.data(), w.w_gate, xb_.data(), nullptr);
        matmul(up_.data(), w.w_up, xb_.data(), nullptr);
        silu_mul(gate_.data(), up_.data(), c.intermediate_size);
        matmul(xb_.data(), w.w_down, gate_.data(), nullptr);
        for (int i = 0; i < c.hidden_size; ++i) x_[i] += xb_[i];

        if (trace) trace("layer_out", l, x_.data(), c.hidden_size);
    }

    kv.set_seq_len(pos + 1);

    if (logits) {
        rmsnorm(xb_.data(), x_.data(), final_norm_.data(), c.hidden_size, c.rms_norm_eps);
        if (trace) trace("final_norm", -1, xb_.data(), c.hidden_size);
        matmul(logits, lm_head_, xb_.data(), nullptr);
    }
}

} // namespace mk
