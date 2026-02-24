#pragma once
#include "hrm/common.h"

namespace hrm {

int overlap_query_blob(const std::vector<uint8_t>& q_query, const std::vector<uint8_t>& qblob);
int overlap_blob_blob(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);

struct Candidate {
    std::string sid;
    uint32_t cid = 0;
    std::string txt;
    std::vector<uint8_t> qblob;
    int rel = 0;
};

std::vector<Candidate> mmr_select(const std::vector<Candidate>& candidates,
                                 int k,
                                 int lam_num = 7,
                                 int lam_den = 10);

} // namespace hrm
