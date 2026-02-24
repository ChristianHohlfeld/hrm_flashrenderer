// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/engine.h"
#include "hrm/payload_jsonl.h"
#include "hrm/sqlite_store.h"
#include "hrm/signature.h"
#include "hrm/router_index.h"
#include "hrm/compose.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <stdexcept>

using namespace hrm;

static void usage() {
    std::cerr <<
    "hrm <command> [args]\n"
    "Commands:\n"
    "  prep   --input <txt> --out <payloads.jsonl> [--cluster-size N]\n"
    "  build  --payloads <payloads.jsonl> --outdir <modeldir>\n"
    "  query  --model <modeldir> [--top-k K] [--top-m M] [--k N] [--format text|json] [--prompt \"...\"]\n"
    "\nNotes:\n"
    "  - If --prompt is provided, query runs once and exits.\n"
    "  - Without --prompt, query starts a deterministic REPL.\n";
}

static std::string argval(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error("missing arg value");
    return std::string(argv[++i]);
}

static std::string read_all_stdin() {
    std::string s, line;
    while (std::getline(std::cin, line)) {
        s += line;
        s.push_back('\n');
    }
    return s;
}

static std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static int cmd_prep(int argc, char** argv) {
    std::string input, out;
    size_t cluster_size = 200;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input") input = argval(i, argc, argv);
        else if (a == "--out") out = argval(i, argc, argv);
        else if (a == "--cluster-size") cluster_size = (size_t)std::stoul(argval(i, argc, argv));
        else throw std::runtime_error("unknown arg: " + a);
    }
    if (input.empty() || out.empty()) throw std::runtime_error("prep requires --input and --out");
    write_payloads_from_text(input, out, cluster_size);
    std::cout << "OK wrote " << out << "\n";
    return 0;
}

static RouterIndex build_router(const std::vector<Payload>& payloads) {
    RouterIndex idx;
    idx.bins = BINS;
    idx.levels = LEVELS;
    idx.max_cid = 0;

    const size_t slots = idx.slots();
    std::vector<std::vector<uint32_t>> tmp(slots);

    for (const auto& p : payloads) {
        idx.max_cid = std::max(idx.max_cid, p.cid);
        std::string joined;
        joined.reserve(4096);
        for (const auto& s : p.snippets) { joined += s; joined.push_back('\n'); }

        const auto q = qbins_from_text(joined);
        for (uint32_t b = 0; b < idx.bins; b++) {
            const uint8_t qb = q[b];
            for (uint32_t lvl = 1; lvl <= qb; lvl++) {
                tmp[(size_t)lvl * idx.bins + b].push_back(p.cid);
            }
        }
    }

    for (auto& v : tmp) {
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    }

    idx.offsets.resize(slots + 1);
    uint64_t cur = 0;
    for (size_t i = 0; i < slots; i++) {
        idx.offsets[i] = cur;
        cur += tmp[i].size();
    }
    idx.offsets[slots] = cur;
    idx.cids.resize((size_t)cur);

    uint64_t pos = 0;
    for (size_t i = 0; i < slots; i++) for (uint32_t cid : tmp[i]) idx.cids[pos++] = cid;

    return idx;
}

static int cmd_build(int argc, char** argv) {
    std::string payloads_path, outdir;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--payloads") payloads_path = argval(i, argc, argv);
        else if (a == "--outdir") outdir = argval(i, argc, argv);
        else throw std::runtime_error("unknown arg: " + a);
    }
    if (payloads_path.empty() || outdir.empty()) throw std::runtime_error("build requires --payloads --outdir");
    std::filesystem::create_directories(outdir);

    const auto payloads = read_payloads_jsonl(payloads_path);

    SqliteStore store;
    store.open(outdir + "/index.sqlite");
    store.init_schema();
    store.begin_tx();

    uint64_t inserted = 0;
    for (const auto& p : payloads) {
        for (uint32_t sidx = 0; sidx < (uint32_t)p.snippets.size(); sidx++) {
            SnipRow row;
            row.cid = p.cid;
            row.sidx = sidx;
            row.sid = format_sid(p.cid, sidx);
            row.txt = p.snippets[sidx];
            row.qblob = pack_qbins_nibbles(qbins_from_text(row.txt));
            store.insert_snip(row);
            inserted++;
            if (inserted % 20000 == 0) { store.commit_tx(); store.begin_tx(); }
        }
    }
    store.commit_tx();
    store.close();

    const RouterIndex idx = build_router(payloads);
    save_router_index(idx, outdir + "/router_index.bin");

    {
        std::ofstream m(outdir + "/meta.json");
        m << "{";
        m << "\"bins\":" << BINS << ",";
        m << "\"levels\":" << LEVELS << ",";
        m << "\"max_cid\":" << idx.max_cid << ",";
        m << "\"snips\":" << inserted;
        m << "}\n";
    }

    std::cout << "OK built model in " << outdir << " (snips=" << inserted << ")\n";
    return 0;
}

static void print_json_once(const std::string& prompt, const QueryResult& r) {
    std::cout << "{\n";
    std::cout << "  \"prompt\": \"" << json_escape(prompt) << "\",\n";

    std::cout << "  \"cids\": [";
    for (size_t i = 0; i < r.cids.size(); i++) {
        if (i) std::cout << ",";
        std::cout << "\"" << format_cid(r.cids[i]) << "\"";
    }
    std::cout << "],\n";

    std::cout << "  \"chosen\": [\n";
    for (size_t i = 0; i < r.chosen.size(); i++) {
        const auto& c = r.chosen[i];
        std::cout << "    {";
        std::cout << "\"sid\":\"" << json_escape(c.sid) << "\",";
        std::cout << "\"cid\":\"" << format_cid(c.cid) << "\",";
        std::cout << "\"rel\":" << c.rel << ",";
        std::cout << "\"txt\":\"" << json_escape(c.txt) << "\",";
        std::cout << "\"txt_c\":\"" << json_escape(compress_line(c.txt)) << "\"";
        std::cout << "}";
        if (i + 1 < r.chosen.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

static int cmd_query(int argc, char** argv) {
    std::string modeldir;
    QueryParams qp;
    std::string format = "text";
    std::string prompt_once;
    bool prompt_from_stdin = false;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") modeldir = argval(i, argc, argv);
        else if (a == "--top-k") qp.top_k = std::stoi(argval(i, argc, argv));
        else if (a == "--top-m") qp.top_m = std::stoi(argval(i, argc, argv));
        else if (a == "--k") qp.k = std::stoi(argval(i, argc, argv));
        else if (a == "--lam-num") qp.lam_num = std::stoi(argval(i, argc, argv));
        else if (a == "--lam-den") qp.lam_den = std::stoi(argval(i, argc, argv));
        else if (a == "--format") format = argval(i, argc, argv);
        else if (a == "--prompt") prompt_once = argval(i, argc, argv);
        else if (a == "--prompt-stdin") prompt_from_stdin = true;
        else throw std::runtime_error("unknown arg: " + a);
    }
    if (modeldir.empty()) throw std::runtime_error("query requires --model");

    Engine e;
    e.open(modeldir);

    if (!prompt_once.empty() || prompt_from_stdin) {
        std::string p = !prompt_once.empty() ? prompt_once : read_all_stdin();
        if (p.empty()) return 0;
        auto r = e.query(p, qp);
        if (format == "json") print_json_once(p, r);
        else {
            std::cout << "[cids] ";
            for (auto cid : r.cids) std::cout << format_cid(cid) << " ";
            std::cout << "\n";
            std::cout << render_answer(p, r.chosen) << "\n";
        }
        e.close();
        return 0;
    }

    std::cout << "HRM query REPL (deterministic). Ctrl+D to exit.\n";
    std::string prompt;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, prompt)) break;
        if (prompt.empty()) continue;

        auto r = e.query(prompt, qp);
        if (format == "json") print_json_once(prompt, r);
        else {
            std::cout << "\n[cids] ";
            for (auto cid : r.cids) std::cout << format_cid(cid) << " ";
            std::cout << "\n";
            std::cout << render_answer(prompt, r.chosen) << "\n";
        }
    }
    e.close();
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 1; }
        std::string cmd = argv[1];
        if (cmd == "prep") return cmd_prep(argc, argv);
        if (cmd == "build") return cmd_build(argc, argv);
        if (cmd == "query") return cmd_query(argc, argv);
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERR: " << e.what() << "\n";
        return 2;
    }
}

