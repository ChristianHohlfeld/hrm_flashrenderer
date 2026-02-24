#include "hrm/rank.h"

namespace hrm {

int overlap_query_blob(const std::vector<uint8_t>& q_query, const std::vector<uint8_t>& qblob) {
    if (q_query.size() != (size_t)BINS) throw std::runtime_error("q_query size");
    if (qblob.size() != (size_t)BINS/2) throw std::runtime_error("qblob size");
    int s = 0;
    for (int i = 0; i < BINS/2; i++) {
        uint8_t v = qblob[i];
        int a = (v >> 4) & 0x0F;
        int b = v & 0x0F;
        s += imin(a, (int)q_query[2*i]);
        s += imin(b, (int)q_query[2*i + 1]);
    }
    return s;
}

int overlap_blob_blob(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != (size_t)BINS/2 || b.size() != (size_t)BINS/2) throw std::runtime_error("blob size");
    int s = 0;
    for (int i = 0; i < BINS/2; i++) {
        uint8_t va = a[i], vb = b[i];
        int a1 = (va >> 4) & 0x0F, a2 = va & 0x0F;
        int b1 = (vb >> 4) & 0x0F, b2 = vb & 0x0F;
        s += imin(a1, b1);
        s += imin(a2, b2);
    }
    return s;
}

std::vector<Candidate> mmr_select(const std::vector<Candidate>& candidates, int k, int lam_num, int lam_den) {
    std::vector<Candidate> chosen;
    chosen.reserve(k);
    std::vector<bool> used(candidates.size(), false);

    for (int iter = 0; iter < k && iter < (int)candidates.size(); iter++) {
        int best_idx = -1;
        long long best_score = 0;

        for (int i = 0; i < (int)candidates.size(); i++) {
            if (used[i]) continue;
            int red = 0;
            for (const auto& c : chosen) red = std::max(red, overlap_blob_blob(candidates[i].qblob, c.qblob));
            long long score = (long long)candidates[i].rel * lam_den - (long long)lam_num * red;

            if (best_idx < 0 || score > best_score ||
                (score == best_score && candidates[i].sid < candidates[best_idx].sid)) {
                best_idx = i;
                best_score = score;
            }
        }
        if (best_idx < 0) break;
        used[best_idx] = true;
        chosen.push_back(candidates[best_idx]);
    }
    return chosen;
}

} // namespace hrm
