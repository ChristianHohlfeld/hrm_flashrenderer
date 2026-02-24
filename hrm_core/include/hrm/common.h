// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace hrm {

constexpr int BINS   = 2048;
constexpr int LEVELS = 16;

inline int imin(int a, int b) { return (a < b) ? a : b; }

inline uint32_t parse_cid_u32(const std::string& cid) {
    // deterministic: accept leading zeros
    return static_cast<uint32_t>(std::stoul(cid));
}

inline std::string format_cid(uint32_t cid) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04u", cid);
    return std::string(buf);
}

inline std::string format_sid(uint32_t cid, uint32_t sidx) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04u#s%04u", cid, sidx);
    return std::string(buf);
}

} // namespace hrm

