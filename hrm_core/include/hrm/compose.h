#pragma once
#include "hrm/rank.h"
#include <string>

namespace hrm {

std::string compress_line(const std::string& txt, size_t max_len = 240);
std::string render_answer(const std::string& prompt, const std::vector<Candidate>& chosen);

} // namespace hrm
