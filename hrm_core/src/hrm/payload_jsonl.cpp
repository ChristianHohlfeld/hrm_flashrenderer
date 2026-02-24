// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/payload_jsonl.h"
#include <fstream>

namespace hrm {

static std::string unescape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (c != '\\') { out.push_back(c); continue; }
        if (i + 1 >= s.size()) break;
        char n = s[++i];
        switch (n) {
            case '\\': out.push_back('\\'); break;
            case '"':  out.push_back('"'); break;
            case 'n':  out.push_back('\n'); break;
            case 't':  out.push_back('\t'); break;
            case 'r':  out.push_back('\r'); break;
            default:   out.push_back(n); break;
        }
    }
    return out;
}

static bool extract_cid(const std::string& line, std::string& cid_out) {
    auto pos = line.find("\"cid\"");
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos);
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos);
    if (pos == std::string::npos) return false;
    auto end = line.find('"', pos+1);
    if (end == std::string::npos) return false;
    cid_out = line.substr(pos+1, end-(pos+1));
    return true;
}

static bool extract_snippets(const std::string& line, std::vector<std::string>& out) {
    auto pos = line.find("\"snippets\"");
    if (pos == std::string::npos) return false;
    pos = line.find('[', pos);
    if (pos == std::string::npos) return false;
    auto end = line.rfind(']');
    if (end == std::string::npos || end <= pos) return false;

    size_t i = pos + 1;
    while (i < end) {
        while (i < end && line[i] != '"') i++;
        if (i >= end) break;
        i++;
        std::string buf;
        bool esc = false;
        while (i < end) {
            char c = line[i++];
            if (esc) { buf.push_back('\\'); buf.push_back(c); esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') break;
            buf.push_back(c);
        }
        out.push_back(unescape_json_string(buf));
        while (i < end && line[i] != '"') i++;
    }
    return true;
}

std::vector<Payload> read_payloads_jsonl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open payloads.jsonl");
    std::vector<Payload> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string cid_s;
        std::vector<std::string> snips;
        if (!extract_cid(line, cid_s)) continue;
        if (!extract_snippets(line, snips)) continue;
        Payload p;
        p.cid = parse_cid_u32(cid_s);
        p.snippets = std::move(snips);
        out.push_back(std::move(p));
    }
    std::sort(out.begin(), out.end(), [](const Payload& a, const Payload& b){ return a.cid < b.cid; });
    return out;
}

static std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:   out.push_back(c); break;
        }
    }
    return out;
}

void write_payloads_from_text(const std::string& input_txt,
                             const std::string& out_jsonl,
                             size_t cluster_size) {
    std::ifstream in(input_txt);
    if (!in) throw std::runtime_error("cannot open input text");
    std::vector<std::string> lines;
    std::string ln;
    while (std::getline(in, ln)) {
        if (!ln.empty() && ln.back() == '\r') ln.pop_back();
        if (ln.find_first_not_of(" \t") == std::string::npos) continue;
        lines.push_back(ln);
    }

    std::ofstream out(out_jsonl);
    if (!out) throw std::runtime_error("cannot open output jsonl");

    size_t clusters = (lines.size() + cluster_size - 1) / cluster_size;
    for (size_t ci = 0; ci < clusters; ci++) {
        uint32_t cid = (uint32_t)ci;
        out << "{\"cid\":\"" << format_cid(cid) << "\",\"title\":\"cluster\",\"snippets\":[";
        size_t start = ci * cluster_size;
        size_t stop = std::min(lines.size(), start + cluster_size);
        for (size_t i = start; i < stop; i++) {
            if (i != start) out << ",";
            out << "\"" << escape_json_string(lines[i]) << "\"";
        }
        out << "]}\n";
    }
}

} // namespace hrm

