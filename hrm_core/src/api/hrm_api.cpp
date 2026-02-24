#include "hrm/api.h"

#include "hrm/engine.h"
#include "hrm/compose.h"

#include <cstdlib>
#include <cstring>
#include <string>

using hrm::Engine;
using hrm::QueryParams;
using hrm::QueryResult;
using hrm::format_cid;
using hrm::compress_line;

struct hrm_handle_t {
    Engine e;
    bool ok = false;
};

static std::string g_last_err;

static std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static char* strdup_heap(const std::string& s) {
    void* p = std::malloc(s.size() + 1);
    if (!p) return nullptr;
    std::memcpy(p, s.data(), s.size());
    ((char*)p)[s.size()] = '\0';
    return (char*)p;
}

extern "C" {

hrm_handle_t* hrm_open(const char* model_dir) {
    try {
        if (!model_dir) {
            g_last_err = "hrm_open: model_dir is null";
            return nullptr;
        }
        auto* h = new hrm_handle_t();
        h->e.open(std::string(model_dir));
        h->ok = true;
        return h;
    } catch (const std::exception& e) {
        g_last_err = std::string("hrm_open failed: ") + e.what();
        return nullptr;
    }
}

void hrm_close(hrm_handle_t* h) {
    if (!h) return;
    try {
        if (h->ok) h->e.close();
    } catch (...) {
        // swallow
    }
    delete h;
}

char* hrm_query_json(
    hrm_handle_t* h,
    const char* prompt_utf8,
    int top_k,
    int top_m,
    int k,
    int lam_num,
    int lam_den
) {
    try {
        if (!h || !h->ok) {
            g_last_err = "hrm_query_json: handle not open";
            return nullptr;
        }
        if (!prompt_utf8) {
            g_last_err = "hrm_query_json: prompt is null";
            return nullptr;
        }

        QueryParams qp;
        qp.top_k = top_k;
        qp.top_m = top_m;
        qp.k = k;
        qp.lam_num = lam_num;
        qp.lam_den = lam_den;

        const std::string prompt(prompt_utf8);
        QueryResult r = h->e.query(prompt, qp);

        std::string out;
        out.reserve(8192);
        out += "{\n";
        out += "  \"prompt\": \"" + json_escape(prompt) + "\",\n";

        out += "  \"cids\": [";
        for (size_t i = 0; i < r.cids.size(); i++) {
            if (i) out += ",";
            out += "\"" + json_escape(format_cid(r.cids[i])) + "\"";
        }
        out += "],\n";

        out += "  \"chosen\": [\n";
        for (size_t i = 0; i < r.chosen.size(); i++) {
            const auto& c = r.chosen[i];
            out += "    {";
            out += "\"sid\":\"" + json_escape(c.sid) + "\",";
            out += "\"cid\":\"" + json_escape(format_cid(c.cid)) + "\",";
            out += "\"rel\":" + std::to_string(c.rel) + ",";
            out += "\"txt\":\"" + json_escape(c.txt) + "\",";
            out += "\"txt_c\":\"" + json_escape(compress_line(c.txt)) + "\"";
            out += "}";
            if (i + 1 < r.chosen.size()) out += ",";
            out += "\n";
        }
        out += "  ]\n";
        out += "}\n";

        char* ret = strdup_heap(out);
        if (!ret) {
            g_last_err = "hrm_query_json: OOM";
            return nullptr;
        }
        return ret;
    } catch (const std::exception& e) {
        g_last_err = std::string("hrm_query_json failed: ") + e.what();
        return nullptr;
    }
}

void hrm_free(void* p) {
    if (p) std::free(p);
}

const char* hrm_last_error(void) {
    return g_last_err.c_str();
}

} // extern "C"
