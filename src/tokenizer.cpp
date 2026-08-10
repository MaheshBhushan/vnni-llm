#include "tokenizer.h"
#include "nfc_tables.h"
#include "unicode_tables.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mk {

// ---------- unicode helpers ----------

static bool in_ranges(const CpRange* rs, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp > rs[mid].hi) lo = mid + 1;
        else if (cp < rs[mid].lo) hi = mid;
        else return true;
    }
    return false;
}

static bool is_letter(uint32_t cp) {
    return in_ranges(kLetterRanges, std::size(kLetterRanges), cp);
}
static bool is_number(uint32_t cp) {
    return in_ranges(kNumberRanges, std::size(kNumberRanges), cp);
}
static bool is_ws(uint32_t cp) {
    return in_ranges(kWhitespaceRanges, std::size(kWhitespaceRanges), cp);
}

// Decodes the UTF-8 codepoint at s[i]; advances len. Invalid bytes are
// treated as Latin-1 (they still round-trip through the byte mapping).
static uint32_t next_cp(const std::string& s, size_t i, int& len) {
    uint8_t b0 = s[i];
    if (b0 < 0x80) { len = 1; return b0; }
    if ((b0 >> 5) == 0x6 && i + 1 < s.size()) {
        len = 2;
        return ((b0 & 0x1F) << 6) | (s[i + 1] & 0x3F);
    }
    if ((b0 >> 4) == 0xE && i + 2 < s.size()) {
        len = 3;
        return ((b0 & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
    }
    if ((b0 >> 3) == 0x1E && i + 3 < s.size()) {
        len = 4;
        return ((b0 & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
               ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    }
    len = 1;
    return b0;
}

static void append_cp_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ---------- NFC normalization ----------
// Table-driven canonical decompose -> reorder -> compose. Hangul algorithmic
// (de)composition is omitted: Korean input is virtually always precomposed,
// and NFC of precomposed Hangul is the identity.

static const NfcDecomp* find_decomp(uint32_t cp) {
    size_t lo = 0, hi = std::size(kNfcDecomp);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kNfcDecomp[mid].cp < cp) lo = mid + 1;
        else if (kNfcDecomp[mid].cp > cp) hi = mid;
        else return &kNfcDecomp[mid];
    }
    return nullptr;
}

static uint8_t combining_class(uint32_t cp) {
    size_t lo = 0, hi = std::size(kNfcCcc);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kNfcCcc[mid].cp < cp) lo = mid + 1;
        else if (kNfcCcc[mid].cp > cp) hi = mid;
        else return kNfcCcc[mid].ccc;
    }
    return 0;
}

static uint32_t find_composite(uint32_t a, uint32_t b) {
    size_t lo = 0, hi = std::size(kNfcComp);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kNfcComp[mid].a < a || (kNfcComp[mid].a == a && kNfcComp[mid].b < b)) lo = mid + 1;
        else if (kNfcComp[mid].a > a || kNfcComp[mid].b > b) hi = mid;
        else return kNfcComp[mid].composed;
    }
    return 0;
}

static void decompose_into(uint32_t cp, std::vector<uint32_t>& out) {
    if (const NfcDecomp* d = find_decomp(cp)) {
        decompose_into(d->a, out);
        if (d->b) decompose_into(d->b, out);
    } else {
        out.push_back(cp);
    }
}

static std::string normalize_nfc(const std::string& text) {
    // Fast path: ASCII (and most Latin-1) needs no work.
    bool ascii = true;
    for (unsigned char c : text)
        if (c >= 0x80) { ascii = false; break; }
    if (ascii) return text;

    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        int len;
        decompose_into(next_cp(text, i, len), cps);
        i += len;
    }

    // Canonical ordering: bubble adjacent marks with descending nonzero ccc.
    for (size_t k = 1; k < cps.size(); ++k) {
        uint8_t c = combining_class(cps[k]);
        if (c == 0) continue;
        size_t j = k;
        while (j > 0 && combining_class(cps[j - 1]) > c) {
            std::swap(cps[j - 1], cps[j]);
            --j;
        }
    }

    // Canonical composition.
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    int last_starter = -1;
    uint8_t last_ccc = 0;
    for (uint32_t cp : cps) {
        uint8_t c = combining_class(cp);
        if (last_starter >= 0 && (last_ccc < c || (last_ccc == 0 && c == 0))) {
            // not blocked (starter directly precedes, or ccc strictly increases)
            if (last_ccc < c || out.size() == static_cast<size_t>(last_starter) + 1) {
                if (uint32_t comp = find_composite(out[last_starter], cp)) {
                    out[last_starter] = comp;
                    continue;
                }
            }
        }
        if (c == 0) { last_starter = static_cast<int>(out.size()); last_ccc = 0; }
        else last_ccc = c;
        out.push_back(cp);
    }

    std::string res;
    res.reserve(text.size());
    for (uint32_t cp : out) append_cp_utf8(res, cp);
    return res;
}

// ---------- construction ----------

Tokenizer::Tokenizer(const std::string& path) {
    // GPT-2 byte <-> unicode table: printable bytes map to themselves,
    // the rest to 256, 257, ... in order.
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        uint32_t cp = printable ? static_cast<uint32_t>(b)
                                : static_cast<uint32_t>(256 + extra++);
        append_cp_utf8(byte_to_char_[b], cp);
        char_to_byte_[cp] = static_cast<uint8_t>(b);
    }

    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);

    const auto& vocab = j.at("model").at("vocab");
    id_to_token_.resize(vocab.size());
    token_to_id_.reserve(vocab.size());
    for (auto& [tok, id] : vocab.items()) {
        int i = id.get<int>();
        if (i >= static_cast<int>(id_to_token_.size())) id_to_token_.resize(i + 1);
        id_to_token_[i] = tok;
        token_to_id_[tok] = i;
    }

    const auto& merges = j.at("model").at("merges");
    merge_rank_.reserve(merges.size());
    int rank = 0;
    for (auto& m : merges) {
        if (m.is_string()) {
            merge_rank_[m.get<std::string>()] = rank++;
        } else {  // newer format: ["a", "b"]
            merge_rank_[m[0].get<std::string>() + " " + m[1].get<std::string>()] = rank++;
        }
    }

    for (auto& a : j.at("added_tokens")) {
        int id = a.at("id").get<int>();
        std::string content = a.at("content").get<std::string>();
        if (id >= static_cast<int>(id_to_token_.size())) id_to_token_.resize(id + 1);
        id_to_token_[id] = content;
        special_to_id_[content] = id;
        special_tokens_.push_back(content);
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

int Tokenizer::special_id(const std::string& content) const {
    auto it = special_to_id_.find(content);
    return it == special_to_id_.end() ? -1 : it->second;
}

// ---------- pre-tokenizer ----------
// Hand-written scanner for:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}
// |  ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
// Alternatives are tried in order at each position (leftmost-first), each
// greedy — matching what the rust regex crate does for this pattern.

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = text.size();

    auto cp_at = [&](size_t pos, int& len) { return next_cp(text, pos, len); };

    while (i < n) {
        int len;
        uint32_t c0 = cp_at(i, len);
        size_t end = i;

        // (a) contractions
        if (c0 == '\'' && i + 1 < n) {
            static const char* alts[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char* alt : alts) {
                size_t alen = alt[1] ? 2 : 1;
                if (i + 1 + alen > n) continue;
                bool ok = true;
                for (size_t k = 0; k < alen; ++k) {
                    char ch = text[i + 1 + k];
                    if (ch >= 'A' && ch <= 'Z') ch += 32;
                    if (ch != alt[k]) { ok = false; break; }
                }
                if (ok) { end = i + 1 + alen; break; }
            }
        }

        // (b) [^\r\n\p{L}\p{N}]?\p{L}+
        if (end == i) {
            size_t p = i;
            if (!is_letter(c0) && !is_number(c0) && c0 != '\r' && c0 != '\n')
                p = i + len;  // tentative optional prefix
            size_t letters_start = p;
            while (p < n) {
                int l2;
                uint32_t c = cp_at(p, l2);
                if (!is_letter(c)) break;
                p += l2;
            }
            if (p > letters_start)
                end = p;
            // If the prefix was consumed but no letters followed, (b) fails
            // even when c0 itself is a letter?  c0 can't be: the prefix class
            // excludes letters, and if c0 were a letter we never took the prefix.
        }

        // (c) \p{N}
        if (end == i && is_number(c0)) end = i + len;

        // (d)  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (end == i) {
            size_t p = i;
            if (c0 == ' ') p = i + 1;
            size_t punct_start = p;
            while (p < n) {
                int l2;
                uint32_t c = cp_at(p, l2);
                if (is_ws(c) || is_letter(c) || is_number(c)) break;
                p += l2;
            }
            if (p > punct_start) {
                while (p < n && (text[p] == '\r' || text[p] == '\n')) ++p;
                end = p;
            }
        }

        // whitespace run for (e)/(f)/(g)
        if (end == i && is_ws(c0)) {
            size_t p = i;
            size_t last_nl = std::string::npos;  // end offset of last \r or \n
            while (p < n) {
                int l2;
                uint32_t c = cp_at(p, l2);
                if (!is_ws(c)) break;
                if (c == '\r' || c == '\n') last_nl = p + l2;
                p += l2;
            }
            if (last_nl != std::string::npos) {
                end = last_nl;                    // (e) \s*[\r\n]+
            } else if (p == n) {
                end = p;                          // (f) trailing whitespace
            } else if (p - i >= 2) {
                // (f) with lookahead: leave the final ws char for the next token
                int l2;
                cp_at(p - 1, l2);  // p-1 may be mid-codepoint; recompute properly
                // step back one codepoint from p
                size_t q = i, prev = i;
                while (q < p) { int l3; cp_at(q, l3); prev = q; q += l3; }
                end = prev > i ? prev : p;        // fall back to (g) if single cp
            } else {
                end = p;                          // (g) single whitespace cp
            }
        }

        if (end == i) end = i + len;  // safety: consume one cp (unreachable)
        out.emplace_back(text.substr(i, end - i));
        i = end;
    }
    return out;
}

// ---------- BPE ----------

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    // bytes -> mapped unicode symbols
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (unsigned char b : piece) sym.push_back(byte_to_char_[b]);

    while (sym.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best = 0;
        for (size_t k = 0; k + 1 < sym.size(); ++k) {
            auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = k;
            }
        }
        if (best_rank == INT32_MAX) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }

    for (auto& s : sym) {
        auto it = token_to_id_.find(s);
        if (it == token_to_id_.end())
            throw std::runtime_error("BPE symbol not in vocab: " + s);
        out.push_back(it->second);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    for (const auto& piece : pre_tokenize(normalize_nfc(text))) bpe(piece, out);
    return out;
}

std::vector<int> Tokenizer::encode_with_specials(const std::string& text) const {
    std::vector<int> out;
    size_t i = 0;
    size_t plain_start = 0;
    while (i < text.size()) {
        const std::string* hit = nullptr;
        for (const auto& sp : special_tokens_) {
            if (text.compare(i, sp.size(), sp) == 0) { hit = &sp; break; }
        }
        if (hit) {
            if (i > plain_start) {
                auto ids = encode(text.substr(plain_start, i - plain_start));
                out.insert(out.end(), ids.begin(), ids.end());
            }
            out.push_back(special_to_id_.at(*hit));
            i += hit->size();
            plain_start = i;
        } else {
            ++i;
        }
    }
    if (plain_start < text.size()) {
        auto ids = encode(text.substr(plain_start));
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return out;
}

std::string Tokenizer::decode_one(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return "";
    const std::string& tok = id_to_token_[id];
    if (special_to_id_.count(tok)) return tok;
    std::string bytes;
    size_t i = 0;
    while (i < tok.size()) {
        int len;
        uint32_t cp = next_cp(tok, i, len);
        auto it = char_to_byte_.find(cp);
        if (it != char_to_byte_.end()) bytes += static_cast<char>(it->second);
        i += len;
    }
    return bytes;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode_one(id);
    return out;
}

} // namespace mk
