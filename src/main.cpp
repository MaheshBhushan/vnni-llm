#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace mk;

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  mk-infer dump     -m <model_dir>\n"
        "  mk-infer tokenize -m <model_dir> -p <text>\n"
        "  mk-infer generate -m <model_dir> -p <prompt> [-n max_tokens] [-t temp] [--raw]\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) usage();
    std::string mode = argv[1];
    std::string model_dir, prompt;
    int max_tokens = 128;
    float temperature = 0.f;
    bool raw = false;
    bool quantize = false;
    int threads = 1;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (a == "-j" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "-q") quantize = true;
        else if (a == "--raw") raw = true;
        else usage();
    }
    if (model_dir.empty()) usage();

    if (mode == "dump") {
        SafetensorsFile st(model_dir);
        size_t total_params = 0, total_bytes = 0;
        for (auto& [name, t] : st.tensors()) {
            std::printf("%-64s %-5s [", name.c_str(), dtype_name(t.dtype));
            for (size_t i = 0; i < t.shape.size(); ++i)
                std::printf("%s%lld", i ? ", " : "", static_cast<long long>(t.shape[i]));
            std::printf("]\n");
            total_params += t.numel();
            total_bytes += dtype_bytes(t.dtype, t.numel());
        }
        std::printf("\n%zu tensors, %.2fB params, %.2f GiB\n", st.tensors().size(),
                    total_params / 1e9, total_bytes / (1024.0 * 1024 * 1024));
        return 0;
    }

    if (mode == "tokenize") {
        Tokenizer tok(model_dir + "/tokenizer.json");
        auto ids = tok.encode(prompt);
        for (size_t i = 0; i < ids.size(); ++i)
            std::printf("%s%d", i ? " " : "", ids[i]);
        std::printf("\n");
        std::string round = tok.decode(ids);
        std::printf("roundtrip %s\n", round == prompt ? "OK" : "MISMATCH");
        return 0;
    }

    // teacher-forcing diff harness: feed an exact token id sequence, print
    // our argmax prediction at every position.
    if (mode == "teacher") {
        Model model(model_dir, quantize, threads);
        const auto& cfg = model.config();
        std::vector<int> ids;
        {
            const char* s = prompt.c_str();
            char* end;
            for (long v = std::strtol(s, &end, 10); s != end; v = std::strtol(s, &end, 10)) {
                ids.push_back(static_cast<int>(v));
                s = end;
            }
        }
        KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 4096);
        std::vector<float> logits(cfg.vocab_size);
        for (size_t i = 0; i < ids.size(); ++i) {
            model.forward(ids[i], kv, logits.data());
            int best = 0;
            for (int v = 1; v < cfg.vocab_size; ++v)
                if (logits[v] > logits[best]) best = v;
            // logit our engine assigns to the token that actually comes next
            // in the fed sequence (i.e. HF's greedy choice when teacher-forcing).
            int nxt = (i + 1 < ids.size()) ? ids[i + 1] : -1;
            float nxt_logit = nxt >= 0 ? logits[nxt] : 0.f;
            std::printf("%zu argmax=%d %.6f next=%d %.6f margin=%.6f\n",
                        i, best, logits[best], nxt, nxt_logit, logits[best] - nxt_logit);
        }
        return 0;
    }

    // Mean negative log-likelihood / perplexity over a text file (contiguous,
    // teacher-forced). Compares quant quality against llama-perplexity.
    if (mode == "perplexity") {
        Tokenizer tok(model_dir + "/tokenizer.json");
        Model model(model_dir, quantize, threads);
        const auto& cfg = model.config();
        std::ifstream in(prompt);  // -p is the text file path here
        std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        std::vector<int> ids = tok.encode(text);
        if (max_tokens > 0 && static_cast<int>(ids.size()) > max_tokens)
            ids.resize(max_tokens);
        std::fprintf(stderr, "[perplexity over %zu tokens]\n", ids.size());

        KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
                   static_cast<int>(ids.size()) + 8);
        std::vector<float> logits(cfg.vocab_size);
        double nll = 0.0;
        int counted = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            model.forward(ids[i], kv, logits.data());
            float mx = logits[0];
            for (int v = 1; v < cfg.vocab_size; ++v) mx = std::max(mx, logits[v]);
            double se = 0.0;
            for (int v = 0; v < cfg.vocab_size; ++v) se += std::exp(logits[v] - mx);
            double logp = (logits[ids[i + 1]] - mx) - std::log(se);
            nll += -logp;
            ++counted;
        }
        double mean_nll = nll / counted;
        std::printf("mean NLL %.4f  perplexity %.4f  (%d tokens)\n",
                    mean_nll, std::exp(mean_nll), counted);
        return 0;
    }

    // Prefill/decode microbenchmark mirroring llama-bench's pp/tg split.
    // -n sets decode tokens; prompt length is fixed at 512 (pp512).
    if (mode == "bench") {
        Model model(model_dir, quantize, threads);
        const auto& cfg = model.config();
        const int pp = 512, tg = max_tokens > 0 ? max_tokens : 128;
        KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, pp + tg + 8);
        std::vector<float> logits(cfg.vocab_size);

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < pp; ++i)
            model.forward(1000 + (i % 100), kv, i + 1 == pp ? logits.data() : nullptr);
        auto t1 = std::chrono::steady_clock::now();
        int token = 1000;
        for (int i = 0; i < tg; ++i) { model.forward(token, kv, logits.data()); token = 1000 + (i % 100); }
        auto t2 = std::chrono::steady_clock::now();

        double pp_s = std::chrono::duration<double>(t1 - t0).count();
        double tg_s = std::chrono::duration<double>(t2 - t1).count();
        std::printf("pp%d: %.2f tok/s   tg%d: %.2f tok/s   (threads=%d, quant=%s)\n",
                    pp, pp / pp_s, tg, tg / tg_s, threads, quantize ? "Q4_0" : "BF16");
        return 0;
    }

    if (mode == "generate") {
        Tokenizer tok(model_dir + "/tokenizer.json");
        Model model(model_dir, quantize, threads);
        const auto& cfg = model.config();

        std::string text = raw ? prompt
            : "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> ids = tok.encode_with_specials(text);
        std::fprintf(stderr, "[prompt: %zu tokens]\n", ids.size());

        KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 4096);
        std::vector<float> logits(cfg.vocab_size);
        Sampler sampler({temperature, 0.8f, 42});

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i + 1 < ids.size(); ++i)
            model.forward(ids[i], kv, nullptr);
        auto t1 = std::chrono::steady_clock::now();

        int token = ids.back();
        int produced = 0;
        for (; produced < max_tokens; ++produced) {
            model.forward(token, kv, logits.data());
            token = sampler.sample(logits.data(), cfg.vocab_size);
            if (token == cfg.eos_token_id || token == tok.special_id("<|im_end|>")) break;
            std::fputs(tok.decode_one(token).c_str(), stdout);
            std::fflush(stdout);
        }
        auto t2 = std::chrono::steady_clock::now();

        double prefill_s = std::chrono::duration<double>(t1 - t0).count();
        double decode_s = std::chrono::duration<double>(t2 - t1).count();
        std::fprintf(stderr, "\n[prefill: %.2f tok/s | decode: %.2f tok/s]\n",
                     (ids.size() - 1) / prefill_s, (produced + 1) / decode_s);
        return 0;
    }

    if (mode == "chat") {
        Tokenizer tok(model_dir + "/tokenizer.json");
        Model model(model_dir, quantize, threads);
        const auto& cfg = model.config();
        KVCache kv(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 8192);
        std::vector<float> logits(cfg.vocab_size);
        Sampler sampler({temperature, 0.8f, 42});
        int im_end = tok.special_id("<|im_end|>");

        // The KV cache is the single source of truth for conversation state:
        // every token fed stays in it, so each turn only processes its own new
        // tokens. feed() leaves `logits` holding the model's prediction for the
        // token that follows the segment's last token.
        auto feed = [&](const std::vector<int>& seg) {
            for (size_t i = 0; i < seg.size(); ++i)
                model.forward(seg[i], kv, i + 1 == seg.size() ? logits.data() : nullptr);
        };

        feed(tok.encode_with_specials(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"));

        std::fprintf(stderr, "chat ready (blank line or Ctrl-D to quit)\n");
        for (;;) {
            std::fputs("\nuser> ", stderr);
            std::string line;
            if (!std::getline(std::cin, line) || line.empty()) break;

            // The header ends with "assistant\n", so `logits` now predicts the
            // first token of the reply.
            feed(tok.encode_with_specials(
                "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n"));

            std::fputs("assistant> ", stdout);
            for (int n = 0; n < max_tokens; ++n) {
                int token = sampler.sample(logits.data(), cfg.vocab_size);
                if (token == cfg.eos_token_id || token == im_end) break;
                std::string piece = tok.decode_one(token);
                std::fputs(piece.c_str(), stdout);
                std::fflush(stdout);
                model.forward(token, kv, logits.data());  // advance, next logits
            }
            std::fputc('\n', stdout);
            // Close the assistant turn so the next turn is well-formed.
            feed({im_end, tok.encode("\n").at(0)});
        }
        return 0;
    }

    usage();
}
