#include "../src/kernels/matmul.h"
#include "../src/kernels/ops.h"
#include "../src/kernels/quant.h"
#include "../src/tensor.h"

#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); ++failures; } \
} while (0)

static float frand() { return (std::rand() / (float)RAND_MAX) * 2.f - 1.f; }

static void test_bf16_conversion() {
    for (float f : {0.f, 1.f, -1.f, 0.5f, 3.14159f, 1e-20f, 1e20f}) {
        uint32_t bits;
        __builtin_memcpy(&bits, &f, 4);
        uint16_t bf = static_cast<uint16_t>(bits >> 16);  // truncation, like torch storage view
        float back = mk::bf16_to_f32(bf);
        CHECK(std::fabs(back - f) <= std::fabs(f) * 0.01f + 1e-30f,
              "bf16 roundtrip %g -> %g", f, back);
    }
}

static void test_matvec_bf16_vs_ref() {
    // include sizes with tails not divisible by 32
    for (int cols : {32, 64, 100, 2048}) {
        for (int rows : {1, 7, 64}) {
            std::vector<uint16_t> w(static_cast<size_t>(rows) * cols);
            std::vector<float> x(cols), bias(rows), a(rows), b(rows);
            for (auto& v : w) { float f = frand(); uint32_t u; __builtin_memcpy(&u, &f, 4); v = u >> 16; }
            for (auto& v : x) v = frand();
            for (auto& v : bias) v = frand();
            mk::matvec_bf16_ref(a.data(), w.data(), x.data(), bias.data(), rows, cols);
            mk::matvec_bf16(b.data(), w.data(), x.data(), bias.data(), rows, cols);
            for (int r = 0; r < rows; ++r)
                CHECK(std::fabs(a[r] - b[r]) < 1e-3f * cols / 32.f,
                      "matvec rows=%d cols=%d r=%d: %g vs %g", rows, cols, r, a[r], b[r]);
        }
    }
}

static void test_softmax() {
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    mk::softmax(x.data(), 4);
    float sum = x[0] + x[1] + x[2] + x[3];
    CHECK(std::fabs(sum - 1.f) < 1e-6f, "softmax sum %g", sum);
    CHECK(x[3] > x[2] && x[2] > x[1], "softmax monotonic");
}

static void test_rmsnorm() {
    std::vector<float> x(64), w(64, 1.f), y(64);
    for (auto& v : x) v = frand();
    mk::rmsnorm(y.data(), x.data(), w.data(), 64, 1e-6f);
    float ss = 0;
    for (float v : y) ss += v * v;
    CHECK(std::fabs(ss / 64 - 1.f) < 1e-3f, "rmsnorm unit rms, got %g", ss / 64);
}

static void test_rope_orthogonality() {
    // Rotations preserve norm.
    std::vector<float> h(128);
    for (auto& v : h) v = frand();
    float n0 = 0; for (float v : h) n0 += v * v;
    mk::rope(h.data(), 128, 17, 1000000.f);
    float n1 = 0; for (float v : h) n1 += v * v;
    CHECK(std::fabs(n0 - n1) < 1e-3f, "rope norm %g -> %g", n0, n1);
}

static void test_quant_matvec() {
    for (int cols : {32, 256, 2048}) {
        for (int rows : {1, 3, 64}) {
            std::vector<float> wf(static_cast<size_t>(rows) * cols), x(cols);
            for (auto& v : wf) v = frand();
            for (auto& v : x) v = frand();

            std::vector<mk::BlockQ4_0> wq(static_cast<size_t>(rows) * (cols / 32));
            for (int r = 0; r < rows; ++r)
                mk::quantize_q4_0(wq.data() + static_cast<size_t>(r) * (cols / 32),
                                  wf.data() + static_cast<size_t>(r) * cols, cols);
            std::vector<mk::BlockQ8_0> xq(cols / 32);
            mk::quantize_q8_0(xq.data(), x.data(), cols);

            std::vector<float> ref(rows), vnni(rows), f32(rows);
            mk::matvec_q4_0_q8_0_ref(ref.data(), wq.data(), xq.data(), nullptr, rows, cols);
            mk::matvec_q4_0_q8_0(vnni.data(), wq.data(), xq.data(), nullptr, rows, cols);
            mk::matvec_f32(f32.data(), wf.data(), x.data(), nullptr, rows, cols);

            for (int r = 0; r < rows; ++r) {
                // VNNI must match the scalar reference bit-for-bit-close.
                CHECK(std::fabs(ref[r] - vnni[r]) < 1e-3f,
                      "q4q8 vnni!=ref rows=%d cols=%d r=%d: %g vs %g", rows, cols, r, ref[r], vnni[r]);
                // Quant vs f32 on *uniform-random* data (worst case for Q4).
                // Dot-product error grows as sqrt(cols); this is a loose
                // sanity bound, not the quality gate — that's model perplexity.
                float tol = 0.12f * std::sqrt(static_cast<float>(cols)) + 0.1f;
                CHECK(std::fabs(ref[r] - f32[r]) < tol,
                      "q4q8 vs f32 rows=%d cols=%d r=%d: %g vs %g (tol %g)",
                      rows, cols, r, ref[r], f32[r], tol);
            }
        }
    }
}

static void test_q4_roundtrip() {
    std::vector<float> x(256), back(256);
    for (auto& v : x) v = frand() * 3.f;
    std::vector<mk::BlockQ4_0> q(256 / 32);
    mk::quantize_q4_0(q.data(), x.data(), 256);
    mk::dequantize_q4_0_row(back.data(), q.data(), 256);
    float maxerr = 0;
    for (int i = 0; i < 256; ++i) maxerr = std::fmax(maxerr, std::fabs(x[i] - back[i]));
    // worst case ~ half a quant level = amax/8/2
    CHECK(maxerr < 0.5f, "q4 roundtrip maxerr %g", maxerr);
}

int main() {
    std::srand(42);
    test_bf16_conversion();
    test_matvec_bf16_vs_ref();
    test_softmax();
    test_rmsnorm();
    test_rope_orthogonality();
    test_quant_matvec();
    test_q4_roundtrip();
    if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
    std::printf("all kernel tests passed\n");
    return 0;
}
