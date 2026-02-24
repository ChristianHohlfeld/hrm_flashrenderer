#pragma once
#include "hrm/common.h"
#include <string>

namespace hrm {

// Flat router postings with offsets for cache-friendly access.
// postings_slot = lvl*bins + bin, lvl in [0..levels-1]. lvl=0 is unused.
struct RouterIndex {
    uint32_t bins = BINS;
    uint32_t levels = LEVELS;
    uint32_t max_cid = 0;

    std::vector<uint64_t> offsets; // size = levels*bins + 1
    std::vector<uint32_t> cids;    // flattened cid lists

    inline size_t slots() const { return static_cast<size_t>(levels) * static_cast<size_t>(bins); }

    inline std::pair<const uint32_t*, const uint32_t*> postings(uint32_t lvl, uint32_t bin) const {
        const size_t idx = static_cast<size_t>(lvl) * bins + bin;
        const uint64_t a = offsets[idx];
        const uint64_t b = offsets[idx + 1];
        return { cids.data() + a, cids.data() + b };
    }
};

RouterIndex load_router_index(const std::string& path);
void save_router_index(const RouterIndex& idx, const std::string& path);

} // namespace hrm
