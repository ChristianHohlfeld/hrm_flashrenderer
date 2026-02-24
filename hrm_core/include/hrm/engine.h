#pragma once
#include "hrm/router_index.h"
#include "hrm/sqlite_store.h"
#include "hrm/rank.h"

namespace hrm {

struct QueryParams {
    int top_k = 5;
    int top_m = 400;
    int k = 8;
    int lam_num = 7;
    int lam_den = 10;
};

struct QueryResult {
    std::vector<uint32_t> cids;
    std::vector<Candidate> chosen;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    void open(const std::string& model_dir);   // loads router_index.bin and opens index.sqlite
    void close();

    QueryResult query(const std::string& prompt, const QueryParams& qp) const;

private:
    RouterIndex router_;
    mutable SqliteStore store_;
};

} // namespace hrm
