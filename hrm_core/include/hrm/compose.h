// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once
#include "hrm/rank.h"
#include <string>

namespace hrm {

std::string compress_line(const std::string& txt, size_t max_len = 240);
std::string render_answer(const std::string& prompt, const std::vector<Candidate>& chosen);

} // namespace hrm

