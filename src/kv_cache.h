#pragma once
#include <vector>

namespace mk {

// Contiguous per-layer KV cache: [n_layers][max_seq][n_kv_heads * head_dim]
// for K and V separately. Paged variant replaces this in Phase 6.
class KVCache {
public:
    KVCache(int n_layers, int n_kv_heads, int head_dim, int max_seq);

    // Pointer to the K (or V) slot for (layer, pos): n_kv_heads*head_dim floats.
    float* k_at(int layer, int pos) { return k_[layer].data() + static_cast<size_t>(pos) * kv_dim_; }
    float* v_at(int layer, int pos) { return v_[layer].data() + static_cast<size_t>(pos) * kv_dim_; }

    int seq_len() const { return seq_len_; }
    void set_seq_len(int n) { seq_len_ = n; }
    int kv_dim() const { return kv_dim_; }
    int max_seq() const { return max_seq_; }

private:
    int kv_dim_, max_seq_, seq_len_ = 0;
    std::vector<std::vector<float>> k_, v_;
};

} // namespace mk
