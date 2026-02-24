// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/router_index.h"
#include <fstream>

namespace hrm {

static constexpr const char* MAGIC = "HRMIDX2";
static constexpr size_t MAGIC_LEN = 6;

void save_router_index(const RouterIndex& idx, const std::string& path) {
    if (idx.offsets.size() != idx.slots() + 1) throw std::runtime_error("bad offsets size");
    if (!idx.offsets.empty() && idx.offsets.back() != idx.cids.size()) throw std::runtime_error("offsets/cids mismatch");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write router_index");

    out.write(MAGIC, MAGIC_LEN);
    out.write((const char*)&idx.bins, sizeof(idx.bins));
    out.write((const char*)&idx.levels, sizeof(idx.levels));
    out.write((const char*)&idx.max_cid, sizeof(idx.max_cid));

    uint64_t off_n = (uint64_t)idx.offsets.size();
    uint64_t cid_n = (uint64_t)idx.cids.size();
    out.write((const char*)&off_n, sizeof(off_n));
    out.write((const char*)&cid_n, sizeof(cid_n));

    out.write((const char*)idx.offsets.data(), idx.offsets.size() * sizeof(uint64_t));
    out.write((const char*)idx.cids.data(), idx.cids.size() * sizeof(uint32_t));
}

RouterIndex load_router_index(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read router_index");

    char magic[MAGIC_LEN];
    in.read(magic, MAGIC_LEN);
    if (std::string(magic, MAGIC_LEN) != std::string(MAGIC, MAGIC_LEN)) throw std::runtime_error("bad router index magic");

    RouterIndex idx;
    in.read((char*)&idx.bins, sizeof(idx.bins));
    in.read((char*)&idx.levels, sizeof(idx.levels));
    in.read((char*)&idx.max_cid, sizeof(idx.max_cid));

    uint64_t off_n = 0, cid_n = 0;
    in.read((char*)&off_n, sizeof(off_n));
    in.read((char*)&cid_n, sizeof(cid_n));

    idx.offsets.resize((size_t)off_n);
    idx.cids.resize((size_t)cid_n);
    in.read((char*)idx.offsets.data(), idx.offsets.size() * sizeof(uint64_t));
    in.read((char*)idx.cids.data(), idx.cids.size() * sizeof(uint32_t));

    if (idx.offsets.size() != idx.slots() + 1) throw std::runtime_error("offset size mismatch");
    if (idx.offsets.back() != idx.cids.size()) throw std::runtime_error("offsets/cids mismatch");
    return idx;
}

} // namespace hrm

