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
