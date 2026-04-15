/*
 * Lightweight SQLite3 compatibility header for build environments where
 * libsqlite3 runtime is present but sqlite3 development headers are missing.
 * This declares only the symbols used by hrm_core.
 */
#ifndef HRM_SQLITE3_COMPAT_H
#define HRM_SQLITE3_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;

typedef void (*sqlite3_destructor_type)(void *);

#ifndef SQLITE_OK
#define SQLITE_OK 0
#endif

#ifndef SQLITE_ROW
#define SQLITE_ROW 100
#endif

#ifndef SQLITE_DONE
#define SQLITE_DONE 101
#endif

#ifndef SQLITE_STATIC
#define SQLITE_STATIC ((sqlite3_destructor_type)0)
#endif

#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((sqlite3_destructor_type)-1)
#endif

int sqlite3_open(const char *filename, sqlite3 **ppDb);
int sqlite3_close(sqlite3 *);
int sqlite3_exec(
    sqlite3 *,
    const char *sql,
    int (*callback)(void *, int, char **, char **),
    void *,
    char **errmsg
);
void sqlite3_free(void *);

int sqlite3_prepare_v2(
    sqlite3 *db,
    const char *zSql,
    int nByte,
    sqlite3_stmt **ppStmt,
    const char **pzTail
);
int sqlite3_finalize(sqlite3_stmt *pStmt);
int sqlite3_reset(sqlite3_stmt *pStmt);
int sqlite3_clear_bindings(sqlite3_stmt *);

int sqlite3_bind_text(sqlite3_stmt *, int, const char *, int, sqlite3_destructor_type);
int sqlite3_bind_int(sqlite3_stmt *, int, int);
int sqlite3_bind_blob(sqlite3_stmt *, int, const void *, int, sqlite3_destructor_type);

int sqlite3_step(sqlite3_stmt *);

const unsigned char *sqlite3_column_text(sqlite3_stmt *, int iCol);
int sqlite3_column_int(sqlite3_stmt *, int iCol);
const void *sqlite3_column_blob(sqlite3_stmt *, int iCol);
int sqlite3_column_bytes(sqlite3_stmt *, int iCol);

#ifdef __cplusplus
}
#endif

#endif
