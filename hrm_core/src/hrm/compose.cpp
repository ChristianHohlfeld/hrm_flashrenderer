// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/compose.h"
#include <regex>
#include <cctype>

namespace hrm {

static std::string collapse_ws(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_ws = false;
    for (char ch : s) {
        unsigned char c = (unsigned char)ch;
        if (std::isspace(c)) {
            if (!in_ws) { out.push_back(' '); in_ws = true; }
        } else {
            out.push_back(ch);
            in_ws = false;
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string compress_line(const std::string& txt, size_t max_len) {
    static const std::regex speaker(R"(^[A-Z][A-Z\s']{1,30}:\s*)");
    static const std::regex brackets(R"(\[[^\]]*\]|\([^\)]*\))");

    std::string s = std::regex_replace(txt, speaker, "");
    s = std::regex_replace(s, brackets, "");
    s = collapse_ws(s);

    if (s.size() <= max_len) return s;
    size_t cut = s.rfind('.', max_len);
    if (cut == std::string::npos) cut = s.rfind(';', max_len);
    if (cut == std::string::npos) cut = s.rfind(',', max_len);
    if (cut == std::string::npos || cut < 20) cut = max_len;
    s = collapse_ws(s.substr(0, cut));
    s += "...";
    return s;
}

static std::string lower_ascii(std::string s) {
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

std::string render_answer(const std::string& prompt, const std::vector<Candidate>& chosen) {
    std::string p = lower_ascii(collapse_ws(prompt));
    std::string head = "Relevant lines:";
    if (p.rfind("who", 0) == 0) head = "Most relevant lines:";
    else if (p.rfind("what", 0) == 0) head = "Most relevant lines:";
    else if (p.rfind("why", 0) == 0) head = "Relevant evidence:";

    std::string out = head + "\n";
    for (const auto& c : chosen) {
        out += "- [" + c.sid + "] " + compress_line(c.txt) + "\n";
    }
    return out;
}

} // namespace hrm

