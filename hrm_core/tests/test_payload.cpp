// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/payload_jsonl.h"
#include <fstream>
#include <stdexcept>

static void assert_true(bool v, const char* msg) {
    if (!v) throw std::runtime_error(msg);
}

void test_payload() {
    const std::string path = "tmp_payloads.jsonl";
    {
        std::ofstream f(path);
        f << "{\"cid\":\"0001\",\"title\":\"x\",\"snippets\":[\"a\",\"b\\nline\",\"c\\\"q\"]}\n";
    }
    auto v = hrm::read_payloads_jsonl(path);
    assert_true(v.size() == 1, "payload count");
    assert_true(v[0].cid == 1, "cid parse");
    assert_true(v[0].snippets.size() == 3, "snips size");
    assert_true(v[0].snippets[1].find("line") != std::string::npos, "newline unescape");
}

