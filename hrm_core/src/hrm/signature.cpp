#include "hrm/signature.h"
#include <cctype>
#include <string_view>

namespace hrm {

static inline uint64_t fnv1a_64(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < n; i++) { h ^= static_cast<uint64_t>(data[i]); h *= prime; }
    return h;
}

static inline bool is_word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '\'';
}

static std::vector<std::string> tokenize_words(const std::string& text) {
    std::vector<std::string> out;
    out.reserve(256);
    std::string cur;
    cur.reserve(32);
    for (char ch : text) {
        if (is_word_char(ch)) cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        else { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static std::string normalize_ws_lower(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_ws = false;
    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (std::isspace(c)) {
            if (!in_ws) { out.push_back(' '); in_ws = true; }
        } else {
            out.push_back(static_cast<char>(std::tolower(c)));
            in_ws = false;
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<uint8_t> qbins_from_text(const std::string& text) {
    constexpr int word_k = 3;
    constexpr int char_k = 5;
    constexpr int max_features = 20000;
    constexpr uint64_t mask = static_cast<uint64_t>(BINS - 1);

    auto toks = tokenize_words(text);
    const std::string norm = normalize_ws_lower(text);

    std::vector<uint32_t> counts(BINS, 0);
    uint32_t total = 0;

    // word 3-grams
    if (toks.size() >= (size_t)word_k) {
        for (size_t i = 0; i + word_k <= toks.size(); i++) {
            if (total >= (uint32_t)max_features) break;
            std::string sh;
            sh.reserve(toks[i].size() + toks[i+1].size() + toks[i+2].size() + 2);
            sh += toks[i]; sh.push_back('\x1f');
            sh += toks[i+1]; sh.push_back('\x1f');
            sh += toks[i+2];
            const uint64_t h = fnv1a_64((const uint8_t*)sh.data(), sh.size());
            const uint32_t b = static_cast<uint32_t>(h & mask);
            counts[b]++; total++;
        }
    }

    // char 5-grams
    if (norm.size() >= (size_t)char_k) {
        for (size_t i = 0; i + char_k <= norm.size(); i++) {
            if (total >= (uint32_t)max_features) break;
            std::string_view sv(norm.data() + i, char_k);
            const uint64_t h = fnv1a_64((const uint8_t*)sv.data(), sv.size());
            const uint32_t b = static_cast<uint32_t>(h & mask);
            counts[b]++; total++;
        }
    }

    if (total == 0) { counts[0] = 1; total = 1; }

    std::vector<uint8_t> q(BINS, 0);
    for (int i = 0; i < BINS; i++) {
        uint32_t qi = (counts[i] * BINS) / total;
        if (qi >= (uint32_t)LEVELS) qi = LEVELS - 1;
        q[i] = (uint8_t)qi;
    }
    return q;
}

std::vector<uint8_t> pack_qbins_nibbles(const std::vector<uint8_t>& qbins) {
    if (qbins.size() != (size_t)BINS) throw std::runtime_error("qbins size must be 2048");
    std::vector<uint8_t> out(BINS/2);
    size_t j = 0;
    for (int i = 0; i < BINS; i += 2) out[j++] = (uint8_t)(((qbins[i]&0x0F)<<4) | (qbins[i+1]&0x0F));
    return out;
}

std::vector<uint8_t> unpack_qbins_nibbles(const std::vector<uint8_t>& blob) {
    if (blob.size() != (size_t)BINS/2) throw std::runtime_error("blob size must be 1024");
    std::vector<uint8_t> q(BINS, 0);
    size_t j = 0;
    for (int i = 0; i < BINS; i += 2) {
        uint8_t v = blob[j++];
        q[i] = (v >> 4) & 0x0F;
        q[i+1] = v & 0x0F;
    }
    return q;
}

} // namespace hrm
