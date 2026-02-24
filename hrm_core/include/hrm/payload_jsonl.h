// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once
#include "hrm/common.h"

namespace hrm {

struct Payload {
    uint32_t cid = 0;
    std::vector<std::string> snippets;
};

std::vector<Payload> read_payloads_jsonl(const std::string& path);
void write_payloads_from_text(const std::string& input_txt,
                             const std::string& out_jsonl,
                             size_t cluster_size);

} // namespace hrm

