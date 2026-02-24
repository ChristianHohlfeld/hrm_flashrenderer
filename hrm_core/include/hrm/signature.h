// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once
#include "hrm/common.h"

namespace hrm {

// Quantized signature (2048 bins, values 0..15)
std::vector<uint8_t> qbins_from_text(const std::string& text);

// Pack/unpack qbins as 4-bit nibbles (1024 bytes)
std::vector<uint8_t> pack_qbins_nibbles(const std::vector<uint8_t>& qbins);
std::vector<uint8_t> unpack_qbins_nibbles(const std::vector<uint8_t>& blob);

} // namespace hrm

