// Fixed, independently specified references; no random inputs or model downloads.
#include "tokenizer.h"
#include "kernels/quant.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static void check(bool ok, const std::string& context) {
    if (!ok) throw std::runtime_error(context);
}

static json load(const std::string& path) {
    std::ifstream input(path);
    check(input.good(), "cannot open fixture: " + path);
    return json::parse(input);
}

static void tokenizer(const std::string& directory) {
    mk::Tokenizer tok(directory + "/tokenizer.json");
    const auto fixture = load(directory + "/regression.json");
    for (const auto& test : fixture.at("tokenizer")) {
        const auto name = test.at("name").get<std::string>();
        const auto text = test.at("text").get<std::string>();
        const auto expected = test.at("ids").get<std::vector<int>>();
        const auto actual = test.value("specials", false)
            ? tok.encode_with_specials(text) : tok.encode(text);
        check(actual == expected, name + ": token IDs differ: " + json(actual).dump());
        check(tok.decode(expected) == test.at("decoded").get<std::string>(),
              name + ": decoded bytes differ");
        std::string pieces;
        for (int id : expected) pieces += tok.decode_one(id);
        check(pieces == tok.decode(expected), name + ": decode_one differs");
    }
    check(tok.special_id("<|end|>") == 300, "special token lookup");
    check(tok.special_id("missing") == -1, "unknown special token lookup");
}

static void quantization(const std::string& directory) {
    const auto fixture = load(directory + "/regression.json");
    std::vector<float> input, expected_dequant;
    for (const auto& test : fixture.at("quantization")) {
        auto block = test.at("input").get<std::vector<float>>();
        auto decoded = test.at("q4_dequant").get<std::vector<float>>();
        check(block.size() == mk::QK && decoded.size() == mk::QK, "fixture block size");
        input.insert(input.end(), block.begin(), block.end());
        expected_dequant.insert(expected_dequant.end(), decoded.begin(), decoded.end());
    }
    const int n = static_cast<int>(input.size());
    std::vector<mk::BlockQ4_0> q4(n / mk::QK);
    std::vector<mk::BlockQ8_0> q8(n / mk::QK);
    mk::quantize_q4_0(q4.data(), input.data(), n);
    mk::quantize_q8_0(q8.data(), input.data(), n);
    for (size_t b = 0; b < q4.size(); ++b) {
        const auto& test = fixture.at("quantization").at(b);
        const auto name = test.at("name").get<std::string>();
        check(q4[b].d == test.at("q4_scale_bits").get<uint16_t>(), name + ": Q4 scale");
        check(q8[b].d == test.at("q8_scale_bits").get<uint16_t>(), name + ": Q8 scale");
        check(q8[b].sum == test.at("q8_sum").get<int>(), name + ": Q8 sum");
        for (int i = 0; i < mk::QK / 2; ++i)
            check(q4[b].qs[i] == test.at("q4_packed").at(i).get<int>(),
                  name + ": Q4 packed byte " + std::to_string(i));
        for (int i = 0; i < mk::QK; ++i)
            check(q8[b].qs[i] == test.at("q8_values").at(i).get<int>(),
                  name + ": Q8 value " + std::to_string(i));
    }
    std::vector<float> decoded(n);
    mk::dequantize_q4_0_row(decoded.data(), q4.data(), n);
    check(decoded == expected_dequant, "Q4 dequantized values");

    // Five 64-column rows exercise both the four-row SIMD loop and its tail.
    std::vector<mk::BlockQ4_0> weights;
    for (int index : fixture.at("matvec").at("weight_blocks").get<std::vector<int>>())
        weights.push_back(q4.at(index));
    std::vector<mk::BlockQ8_0> activations;
    for (int index : fixture.at("matvec").at("activation_blocks").get<std::vector<int>>())
        activations.push_back(q8.at(index));
    const auto golden = fixture.at("matvec").at("output").get<std::vector<float>>();
    const std::array<float, 5> bias = {0.25f, -0.5f, 1.f, 2.f, -2.f};
    check(weights.size() == 10 && activations.size() == 2 && golden.size() == 5,
          "matvec fixture dimensions");
    for (int rows : {1, 2, 3, 4, 5}) {
        for (bool with_bias : {false, true}) {
            for (auto matvec : {mk::matvec_q4_0_q8_0_ref, mk::matvec_q4_0_q8_0}) {
                std::array<float, 5> output{};
                matvec(output.data(), weights.data(), activations.data(),
                       with_bias ? bias.data() : nullptr, rows, 64);
                for (int r = 0; r < rows; ++r) {
                    float expected = golden[r] + (with_bias ? bias[r] : 0.f);
                    // All fixture scales are binary fractions, so these sums are exact.
                    check(std::isfinite(output[r]) && output[r] == expected,
                          "matvec row " + std::to_string(r) + ": expected " +
                          std::to_string(expected) + ", got " + std::to_string(output[r]));
                }
            }
        }
    }
}

int main(int argc, char** argv) try {
    check(argc == 3, "usage: test_regression <tokenizer|quantization> <fixture_dir>");
    const std::string suite = argv[1];
    if (suite == "tokenizer") tokenizer(argv[2]);
    else if (suite == "quantization") quantization(argv[2]);
    else throw std::runtime_error("unknown suite: " + suite);
    std::cout << suite << " fixed references passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
