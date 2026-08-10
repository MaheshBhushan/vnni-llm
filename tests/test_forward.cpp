// Golden-logit test vs HF reference dumps (tools/dump_reference.py).
// Usage: test_forward <model_dir> <reference_dir>
#include "../src/model.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Reference files are raw little-endian f32 (no npy header):
//   tokens.txt          one token id per line
//   layer_<i>.bin       hidden state after layer i, for the LAST prompt token
//   final_norm.bin      after final rmsnorm, last token
//   logits.bin          final logits, last token
static std::vector<float> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    return v;
}

static float max_abs_diff(const std::vector<float>& a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: test_forward <model_dir> <ref_dir>\n"); return 2; }
    std::string model_dir = argv[1], ref_dir = argv[2];

    std::vector<int> tokens;
    {
        std::ifstream f(ref_dir + "/tokens.txt");
        int t;
        while (f >> t) tokens.push_back(t);
    }
    std::fprintf(stderr, "prompt: %zu tokens\n", tokens.size());

    mk::Model model(model_dir);
    const auto& cfg = model.config();
    mk::KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 512);

    bool last = false;
    int worst_layer = -1;
    float worst = 0.f;
    model.trace = [&](const std::string& what, int layer, const float* data, int n) {
        if (!last) return;
        std::string fname = what == "layer_out"
            ? ref_dir + "/layer_" + std::to_string(layer) + ".bin"
            : ref_dir + "/final_norm.bin";
        auto ref = load_bin(fname);
        float d = max_abs_diff(ref, data, n);
        std::printf("%-12s %3d  max_abs_diff %.6f\n", what.c_str(), layer, d);
        if (d > worst) { worst = d; worst_layer = layer; }
    };

    std::vector<float> logits(cfg.vocab_size);
    for (size_t i = 0; i < tokens.size(); ++i) {
        last = (i + 1 == tokens.size());
        model.forward(tokens[i], kv, last ? logits.data() : nullptr);
    }

    auto ref_logits = load_bin(ref_dir + "/logits.bin");
    float dl = max_abs_diff(ref_logits, logits.data(), cfg.vocab_size);
    int our_argmax = 0, ref_argmax = 0;
    for (int i = 1; i < cfg.vocab_size; ++i) {
        if (logits[i] > logits[our_argmax]) our_argmax = i;
        if (ref_logits[i] > ref_logits[ref_argmax]) ref_argmax = i;
    }
    std::printf("logits: max_abs_diff %.6f | argmax ours=%d ref=%d %s\n",
                dl, our_argmax, ref_argmax, our_argmax == ref_argmax ? "MATCH" : "MISMATCH");
    std::printf("worst layer: %d (%.6f)\n", worst_layer, worst);

    // BF16 weights + f32 accumulation vs HF f32: expect small drift, growing
    // with depth. Threshold is deliberately loose; argmax must match exactly.
    bool ok = (our_argmax == ref_argmax) && dl < 0.05f;
    std::printf(ok ? "FORWARD PASS OK\n" : "FORWARD PASS FAILED\n");
    return ok ? 0 : 1;
}
