<h1 align="center">vnni-llm</h1>
<p align="center">An LLM inference engine written from scratch in C++20 — no PyTorch, no ONNX, no BLAS — running Qwen2.5-3B on a laptop CPU with hand-written AVX-512 VNNI INT4 kernels.</p>

<p align="center">
  <img alt="top language" src="https://img.shields.io/github/languages/top/MaheshBhushan/vnni-llm">
  <img alt="code size" src="https://img.shields.io/github/languages/code-size/MaheshBhushan/vnni-llm">
  <img alt="last commit" src="https://img.shields.io/github/last-commit/MaheshBhushan/vnni-llm">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#benchmark-vs-llamacpp">Benchmark</a> ·
  <a href="#correctness">Correctness</a> ·
  <a href="#whats-slow-and-why">What's slow</a>
</p>

<!-- HERO: record a terminal GIF of `./build/mk-infer chat` generating a reply and drop it here as assets/demo.gif. See the punch-list at the bottom of this file. -->

## Overview

Running a language model comes down to reading a few billion numbers off RAM and multiplying them by a vector, once per token. Every serious engine is a set of answers to two questions: how small can you make those numbers, and how fast can you stream them through the CPU.

This repo answers both from zero. Weights load straight from HuggingFace `safetensors`, get quantized to 4 bits, and are multiplied against 8-bit activations using `vpdpbusd` — a single AVX-512 VNNI instruction that does four int8 multiply-accumulates at once. The transformer, the BPE tokenizer, the KV cache, the sampler, and the threadpool are all hand-written. The only dependency is a header-only JSON parser.

It is not faster than llama.cpp. It is roughly half the decode speed, and the [benchmark section](#benchmark-vs-llamacpp) shows the gap with the profiling that explains it.

## Quickstart

Needs a C++20 compiler, CMake ≥ 3.20, and a Linux x86-64 machine.

```bash
git clone https://github.com/MaheshBhushan/vnni-llm.git
cd vnni-llm
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure --timeout 20
```

Weights are not in the repo. Fetch Qwen2.5-3B-Instruct (~6 GB) into `models/`:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen2.5-3b-instruct
```

Then generate:

```bash
./build/mk-infer generate -m models/qwen2.5-3b-instruct -q -j 4 -p "Explain attention."
./build/mk-infer chat     -m models/qwen2.5-3b-instruct -q -j 4
./build/mk-infer bench    -m models/qwen2.5-3b-instruct -q -j 4    # pp512 / tg128
```

| Flag | Meaning |
|---|---|
| `-q` | Q4_0 quantized weights (omit for BF16 — 4× the RAM, slower) |
| `-j N` | worker threads |
| `-t` | sampling temperature (`0` = greedy) |
| `-n` | max tokens to generate |
| `--raw` | skip the chat template |

> [!IMPORTANT]
> The fast path needs **AVX-512 VNNI + VL** (Intel Ice Lake / Tiger Lake or newer, or Zen 4+). On other CPUs the kernels compile down to a scalar fallback that is correct but many times slower. `-march=native` is set in `CMakeLists.txt`, so binaries are not portable between machines.

> [!NOTE]
> BF16 mode memory-maps ~6 GB of weights. On a 16 GB laptop, run with `-q`; quantized mode releases the mapping after conversion (`src/model.cpp:139`) specifically because the two working sets were thrashing each other.

### Regression tests and installed CPU smoke

CTest runs the kernel tests and the fixed C++ tokenizer/quantization references
in [`tests/fixtures`](tests/fixtures/README.md). Python 3 enables the `cpu_smoke`
test, which generates a tiny one-layer BF16 model with the standard library.
These checks require no downloaded weights, network access, or Python packages.
Release builds keep the reference checks active.

Install the native executable into a local prefix and check that executable:

```bash
cmake --install build --prefix "$PWD/build/install"
python3 tests/smoke_cpu.py --binary build/install/bin/mk-infer
```

The smoke scenario feeds the UTF-8 prompt `hé` through stdin with raw prompting,
greedy sampling (`-t 0`), and three requested output tokens. It checks BF16 and
Q4 inference with one and two threads, twice per configuration. Every run must
return `text=AAA`, `prompt_tokens=3`, `output_tokens=3`, and
`finish_reason=length`. Timing fields vary and do not enter reference comparisons.
The script returns nonzero on a timeout, engine failure, or reference mismatch.
This synthetic model verifies installation and basic inference wiring. It does
not measure production-model quality or performance.

The default build selects CPU instructions with `-march=native`. To exercise
the scalar fallback on x86-64, build a separate directory with baseline flags:

```bash
cmake -S . -B build/scalar -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g -march=x86-64"
cmake --build build/scalar -j2
ctest --test-dir build/scalar --output-on-failure --timeout 20
```

The HuggingFace forward comparison remains a separate optional check because it
requires a model and reference dumps:

```bash
./build/test_forward /path/to/model /path/to/reference-dumps
```

## Python MCP service

The stdio service exposes one `generate` tool through the [official MCP Python
SDK](https://py.sdk.modelcontextprotocol.io/v1/server/). Build `mk-infer` and
download a supported model as above, then install the Python service:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/vnni-mcp --binary build/mk-infer --model models/qwen2.5-3b-instruct --quantize --threads 4
```

Configure an MCP client with absolute paths:

```json
{
  "mcpServers": {
    "vnni-llm": {
      "command": "/absolute/path/vnni-llm/.venv/bin/vnni-mcp",
      "args": ["--binary", "/absolute/path/vnni-llm/build/mk-infer",
               "--model", "/absolute/path/vnni-llm/models/qwen2.5-3b-instruct",
               "--quantize", "--threads", "4", "--timeout", "120"]
    }
  }
}
```

Call `generate` with `{"prompt":"Explain attention.","max_tokens":128,
"temperature":0,"raw":false}`. The service uses the existing Qwen chat template
unless `raw` is true. It sends prompts over stdin and reserves stdout for MCP.
Each call launches a fresh engine process and loads the configured model.

| Bound | Value |
|---|---|
| Prompt | Nonempty, at most 16,384 UTF-8 bytes and 2,048 encoded tokens |
| Output | 1–1,024 requested tokens; default 128 |
| Context | Prompt plus requested output must fit `min(4096, max_position_embeddings)` |
| Sampling | Temperature 0–2; default greedy; fixed seed 42 and top-p 0.8 |
| Execution | One active call; concurrent calls return a busy error |
| Time | Default 120 seconds including startup/loading; `--timeout` accepts up to 300 |

Timeouts and client cancellation kill and reap the engine. Invalid requests,
engine failures, and timeouts produce MCP tool errors without success metrics.

Successful calls return structured JSON with generated `text`, `finish_reason`
(`stop` or `length`), exact `prompt_tokens` and `output_tokens`, `generation_ms`,
and `latency_ms`. Prompt tokens include the chat template and special tokens.
Output tokens count emitted tokens and exclude the terminating EOS token.
`generation_ms` measures native prefill and decoding. `latency_ms` measures the
Python subprocess round trip, including weight loading, using a monotonic clock.
The engine replaces incomplete UTF-8 output fragments with replacement characters.
Counts describe generated token IDs before that replacement.

Results also record the prompt SHA-256, engine binary SHA-256, model directory,
token limit, temperature, seed, template mode, thread count, and quantization mode.
Comparison harnesses can persist each structured result as a JSONL record. Keep
model files immutable during a run and record their revision or checksums alongside
results. Keep the engine binary immutable too. The directory path does not identify model content. Record CPU/hardware
and distinguish cold model-loading latency from native generation time when
comparing runs. The service provides no production-model benchmark claims.

The native structured interface is also available directly:

```bash
printf '%s' 'Explain attention.' | ./build/mk-infer generate -m models/qwen2.5-3b-instruct -q --json --prompt-stdin -n 128
.venv/bin/python -m unittest discover -s tests -p test_mcp.py -v
```

The Python tests generate a tiny one-layer model in a temporary directory and
exercise real BF16/Q4 inference, MCP initialization/tool schemas/results, token
bounds, error propagation, timeout cleanup, and cancellation. They require a
built engine and the installed Python dependencies. They do not download weights.

## How it works

One token's round trip:

**Load.** `src/safetensors.cpp` mmaps the shards and reads the header. Each of the 7 weight matrices per layer is converted BF16 → Q4_0: 4-bit integers in blocks of 32, one f32 scale per block. 3.09 B parameters go from ~6 GB to ~1.7 GB.

**Tokenize.** `src/tokenizer.cpp` is a byte-level BPE with a GPT-4-style pre-tokenizer written as a hand-rolled Unicode-category scanner, plus NFC normalization and merge ranks. Generated tables live in `src/unicode_tables.h` and `src/nfc_tables.h`.

**Forward.** `src/model.cpp:154`, 36 layers of:

```
rmsnorm → Q,K,V → RoPE → grouped-query attention → O → residual
rmsnorm → gate,up → SwiGLU → down → residual
```

K and V are written directly into their KV-cache slot for the current position, so step *n* computes one new key/value and reuses the previous *n−1* instead of recomputing the whole prefix.

**Multiply.** Every projection is a matrix–vector product. The activation vector is quantized to Q8_0 on the fly, then `matvec_q4_0_q8_0` (`src/kernels/quant.cpp:96`) runs the `vpdpbusd` kernel four weight rows at a time, so four independent accumulator chains keep the port busy. Rows are split across a spin-wait threadpool (`src/threadpool.cpp`) — spinning rather than blocking because at ~1 ms per token, condition-variable wakeup latency is a measurable fraction of the budget.

**Sample.** Greedy argmax, or temperature + top-p (`src/sampler.cpp`). The chosen token is fed back in.

## Correctness

The engine is checked against HuggingFace at every layer, not just at the output.

| Check | Result | How |
|---|---|---|
| Per-layer activations | agree to ~1e-4 across all 36 layers | `tools/dump_reference.py` dumps HF activations, `tests/test_forward.cpp` asserts agreement |
| Greedy decode | token-exact for 50 tokens | teacher-forcing harness (`mk-infer teacher`) |
| Tokenizer | exact token-ID match vs HF `tokenizers` | `tools/verify_tokenizer.py`, adversarial + 200-case fuzz set |
| VNNI kernel | bit-close to a scalar reference | `tests/test_kernels.cpp` |
| Quantization quality | `mk-infer perplexity -p wiki.test.raw` | mean NLL over a contiguous corpus |

`-ffast-math` is deliberately off, so reductions stay comparable against the reference dumps.

## Benchmark vs llama.cpp

Qwen2.5-3B Q4_0, 4 threads, RAM-resident, same machine (Intel i5-1135G7, Tiger Lake), back to back. **Not at parity.**

| Engine | Prefill pp512 | Decode tg128 |
|---|---:|---:|
| llama.cpp (`mmap=0`) | 46.74 ± 8.14 tok/s | 12.00 ± 0.27 tok/s |
| vnni-llm | ~8 tok/s | ~6.1 tok/s |

llama.cpp figures are the committed run in [`bench/llama_bench2.txt`](bench/llama_bench2.txt) (build `2da6686`). [`bench/llama_bench.txt`](bench/llama_bench.txt) is the same binary with mmap on, where decode collapses to 3.10 tok/s from page thrashing — which is what motivated dropping the mapping in quantized mode here too.

## What's slow, and why

**Decode is ~2× off** (it was 3× before the 4-row kernel). llama.cpp reaches roughly the memory-bandwidth ceiling of this machine; this engine sits around half of it, meaning it is still compute-bound — the weight bytes arrive faster than the kernel consumes them. The measured next steps:

1. Process two Q4 blocks per `vpdpbusd`, halving per-MAC instruction overhead.
2. Cut threading overhead. There are currently 252 fork–join syncs per token; a persistent per-layer work model removes most of that latency.
3. Prefetch the next weight rows during the current block's compute.

**Prefill is ~6× off** for a structural reason, not a tuning one: this engine feeds the prompt one token at a time (GEMV), so every weight byte is read once per token. llama.cpp batches the prompt into a GEMM and amortizes each weight read across the whole batch. Batched prefill is the fix, and it is not implemented.

Further out: paged attention, continuous batching (which turns decode into a GEMM too), and an OpenAI-compatible HTTP server.

## Repository structure

```
src/
  main.cpp          CLI: generate, chat, bench, tokenize, dump, teacher, perplexity
  model.cpp/.h      config parsing, weight loading, the transformer forward pass
  safetensors.cpp   mmap + multi-shard reader, BF16→F32
  tokenizer.cpp     byte-level BPE, Unicode pre-tokenizer, NFC
  kv_cache.cpp      per-layer K/V storage
  sampler.cpp       greedy / temperature / top-p
  threadpool.cpp    spin-wait row-partitioned parallel_for
  kernels/
    quant.cpp       Q4_0/Q8_0 quantize + the vpdpbusd VNNI matvec
    matmul.cpp      BF16 matvec (AVX-512, scalar fallback)
    ops.cpp         rmsnorm, rope, softmax, silu_mul
  *_tables.h        generated Unicode + NFC tables
tests/              kernel unit tests, layer-by-layer forward test
tools/              HF reference dumps, tokenizer verification, table generators
bench/              llama.cpp benchmark logs (the GGUF and corpus are gitignored)
```

## Acknowledgements

Quantization block formats (Q4_0/Q8_0) follow the layout established by [llama.cpp](https://github.com/ggerganov/llama.cpp), which is also the benchmark target. JSON parsing via [nlohmann/json](https://github.com/nlohmann/json), vendored in `third_party/`. Model weights are Qwen2.5-3B-Instruct by Alibaba Cloud, under their own license.

## Contact

Mahesh Bhushan Chowdary Koduri — [@MaheshBhushan](https://github.com/MaheshBhushan)
