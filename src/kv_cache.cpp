#include "kv_cache.h"

namespace mk {

KVCache::KVCache(int n_layers, int n_kv_heads, int head_dim, int max_seq)
    : kv_dim_(n_kv_heads * head_dim), max_seq_(max_seq) {
    k_.resize(n_layers);
    v_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        k_[l].assign(static_cast<size_t>(max_seq) * kv_dim_, 0.f);
        v_[l].assign(static_cast<size_t>(max_seq) * kv_dim_, 0.f);
    }
}

} // namespace mk
