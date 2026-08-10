#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mk {

// Byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json.
// Implements the GPT-4-style pre-tokenizer split by hand (see tokenizer.cpp).
class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path);

    // Encodes text. Special tokens in the text are NOT parsed; use
    // encode_with_specials for template-formatted strings.
    std::vector<int> encode(const std::string& text) const;

    // Splits on registered special-token strings first, then BPE-encodes the
    // pieces in between.
    std::vector<int> encode_with_specials(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;
    // Decodes one token id to raw bytes (may be an incomplete UTF-8 fragment).
    std::string decode_one(int id) const;

    int special_id(const std::string& content) const;  // -1 if absent
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    void bpe(const std::string& piece, std::vector<int>& out) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;   // "left right" -> rank
    std::unordered_map<std::string, int> special_to_id_;
    std::vector<std::string> special_tokens_;           // longest-first
    std::string byte_to_char_[256];                     // GPT-2 byte->unicode (UTF-8)
    std::unordered_map<uint32_t, uint8_t> char_to_byte_; // codepoint -> byte
};

} // namespace mk
