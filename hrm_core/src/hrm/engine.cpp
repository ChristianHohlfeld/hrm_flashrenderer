// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/engine.h"
#include "hrm/signature.h"
#include "hrm/rank.h"

namespace hrm {

Engine::~Engine() { close(); }

void Engine::open(const std::string& model_dir) {
    close();
    router_ = load_router_index(model_dir + "/router_index.bin");
    store_.open(model_dir + "/index.sqlite");
}

void Engine::close() {
    store_.close();
}

QueryResult Engine::query(const std::string& prompt, const QueryParams& qp) const {
    QueryResult out;
    const auto q = qbins_from_text(prompt);

    // Route: scores per cid
    std::vector<int32_t> scores(router_.max_cid + 1, 0);
    for (uint32_t b = 0; b < router_.bins; b++) {
        const uint8_t qb = q[b];
        for (uint32_t lvl = 1; lvl <= qb; lvl++) {
            const int w = (int)lvl;
            auto [p, e] = router_.postings(lvl, b);
            for (auto it = p; it != e; ++it) {
                const uint32_t cid = *it;
                if (cid <= router_.max_cid) scores[cid] += w;
            }
        }
    }

    std::vector<std::pair<int32_t, uint32_t>> items;
    items.reserve(scores.size());
    for (uint32_t cid = 0; cid < scores.size(); cid++) if (scores[cid] > 0) items.emplace_back(scores[cid], cid);

    std::sort(items.begin(), items.end(), [](auto a, auto b){
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    for (int i = 0; i < (int)items.size() && i < qp.top_k; i++) out.cids.push_back(items[i].second);

    // Candidates
    std::vector<Candidate> cand;
    for (uint32_t cid : out.cids) {
        auto rows = store_.fetch_by_cid(cid);
        for (auto& r : rows) {
            Candidate c;
            c.sid = r.sid;
            c.cid = r.cid;
            c.txt = r.txt;
            c.qblob = std::move(r.qblob);
            c.rel = overlap_query_blob(q, c.qblob);
            cand.push_back(std::move(c));
        }
    }

    std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b){
        if (a.rel != b.rel) return a.rel > b.rel;
        return a.sid < b.sid;
    });
    if ((int)cand.size() > qp.top_m) cand.resize(qp.top_m);

    out.chosen = mmr_select(cand, qp.k, qp.lam_num, qp.lam_den);
    return out;
}

} // namespace hrm

