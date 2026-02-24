// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/router_index.h"
#include <fstream>
#include <stdexcept>

static void assert_true(bool v, const char* msg) {
    if (!v) throw std::runtime_error(msg);
}

void test_router_index() {
    hrm::RouterIndex idx;
    idx.bins = hrm::BINS;
    idx.levels = hrm::LEVELS;
    idx.max_cid = 9;

    const size_t slots = idx.slots();
    idx.offsets.assign(slots + 1, 0);

    // Put cid 3 into slot (lvl=1, bin=7)
    const size_t slot = 1 * idx.bins + 7;

    // offsets: all 0 up to slot, then 1 after slot
    for (size_t i = slot + 1; i < idx.offsets.size(); i++) idx.offsets[i] = 1;
    idx.cids = {3};

    const std::string path = "tmp_router.bin";
    hrm::save_router_index(idx, path);
    auto r = hrm::load_router_index(path);

    auto [p, e] = r.postings(1, 7);
    assert_true((e - p) == 1, "posting len");
    assert_true(*p == 3, "posting val");
}

