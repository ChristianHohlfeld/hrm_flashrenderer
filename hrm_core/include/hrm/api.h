// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#pragma once

// Production integration ABI (C)
// - Deterministic, no subprocess
// - Returns JSON compatible with `hrm query --format json`

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrm_handle_t hrm_handle_t;

// Open HRM model directory containing router_index.bin and index.sqlite.
// Returns NULL on failure.
hrm_handle_t* hrm_open(const char* model_dir);

// Close and free handle.
void hrm_close(hrm_handle_t* h);

// Query JSON.
// Returns heap-allocated, NUL-terminated UTF-8 string (must free with hrm_free).
// Returns NULL on failure.
char* hrm_query_json(
    hrm_handle_t* h,
    const char* prompt_utf8,
    int top_k,
    int top_m,
    int k,
    int lam_num,
    int lam_den
);

// Free pointer allocated by hrm_query_json.
void hrm_free(void* p);

// On failure, returns pointer to internal static error string (do not free).
// Note: not thread-safe; intended for single-threaded server use.
const char* hrm_last_error(void);

#ifdef __cplusplus
}
#endif

