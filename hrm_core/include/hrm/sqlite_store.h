// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once
#include "hrm/common.h"
#include <sqlite3.h>

namespace hrm {

struct SnipRow {
    std::string sid;
    uint32_t cid = 0;
    uint32_t sidx = 0;
    std::string txt;
    std::vector<uint8_t> qblob; // 1024 bytes
};

class SqliteStore {
public:
    SqliteStore() = default;
    ~SqliteStore();

    void open(const std::string& path);
    void close();

    void init_schema();
    void begin_tx();
    void commit_tx();

    void insert_snip(const SnipRow& row);
    std::vector<SnipRow> fetch_by_cid(uint32_t cid) const;

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* ins_ = nullptr;
};

} // namespace hrm

