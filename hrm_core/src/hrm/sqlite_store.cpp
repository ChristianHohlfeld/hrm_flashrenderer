// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include "hrm/sqlite_store.h"
#include <stdexcept>

namespace hrm {

SqliteStore::~SqliteStore() { close(); }

void SqliteStore::open(const std::string& path) {
    close();
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) throw std::runtime_error("sqlite open failed");
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=FULL;", nullptr, nullptr, nullptr);
}

void SqliteStore::close() {
    if (ins_) { sqlite3_finalize(ins_); ins_ = nullptr; }
    if (db_) { sqlite3_close(db_); db_ = nullptr; }
}

void SqliteStore::init_schema() {
    const char* sql =
        "CREATE TABLE IF NOT EXISTS snips ("
        " sid TEXT PRIMARY KEY,"
        " cid INTEGER,"
        " sidx INTEGER,"
        " txt TEXT,"
        " q BLOB"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_snips_cid ON snips(cid);";
    char* err = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string e = err ? err : "";
        sqlite3_free(err);
        throw std::runtime_error("sqlite schema failed: " + e);
    }
}

void SqliteStore::begin_tx() {
    char* err = nullptr;
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, &err) != SQLITE_OK) {
        std::string e = err ? err : "";
        sqlite3_free(err);
        throw std::runtime_error("sqlite begin failed: " + e);
    }
}

void SqliteStore::commit_tx() {
    char* err = nullptr;
    if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
        std::string e = err ? err : "";
        sqlite3_free(err);
        throw std::runtime_error("sqlite commit failed: " + e);
    }
}

void SqliteStore::insert_snip(const SnipRow& row) {
    if (!ins_) {
        const char* sql = "INSERT OR REPLACE INTO snips(sid,cid,sidx,txt,q) VALUES(?,?,?,?,?);";
        if (sqlite3_prepare_v2(db_, sql, -1, &ins_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("sqlite prepare insert failed");
        }
    }

    sqlite3_reset(ins_);
    sqlite3_clear_bindings(ins_);
    sqlite3_bind_text(ins_, 1, row.sid.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(ins_, 2, (int)row.cid);
    sqlite3_bind_int(ins_, 3, (int)row.sidx);
    sqlite3_bind_text(ins_, 4, row.txt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(ins_, 5, row.qblob.data(), (int)row.qblob.size(), SQLITE_TRANSIENT);
    if (sqlite3_step(ins_) != SQLITE_DONE) {
        throw std::runtime_error("sqlite insert failed");
    }
}

std::vector<SnipRow> SqliteStore::fetch_by_cid(uint32_t cid) const {
    const char* sql = "SELECT sid,cid,sidx,txt,q FROM snips WHERE cid=? ORDER BY sidx ASC;";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK) throw std::runtime_error("sqlite prepare select failed");
    sqlite3_bind_int(st, 1, (int)cid);

    std::vector<SnipRow> out;
    while (true) {
        int rc = sqlite3_step(st);
        if (rc == SQLITE_DONE) break;
        if (rc != SQLITE_ROW) { sqlite3_finalize(st); throw std::runtime_error("sqlite step failed"); }
        SnipRow r;
        r.sid  = (const char*)sqlite3_column_text(st, 0);
        r.cid  = (uint32_t)sqlite3_column_int(st, 1);
        r.sidx = (uint32_t)sqlite3_column_int(st, 2);
        r.txt  = (const char*)sqlite3_column_text(st, 3);
        const void* blob = sqlite3_column_blob(st, 4);
        int blen = sqlite3_column_bytes(st, 4);
        r.qblob.assign((const uint8_t*)blob, (const uint8_t*)blob + blen);
        out.push_back(std::move(r));
    }
    sqlite3_finalize(st);
    return out;
}

} // namespace hrm

