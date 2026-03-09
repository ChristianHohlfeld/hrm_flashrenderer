#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

static constexpr uint32_t MAGIC = 0x50484F58u; // PHOX
static constexpr int MAX_Q = 32;
static constexpr int MAX_PHO = 8;
static constexpr int MAX_CANDS = 4096;

struct IndexHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t record_count;
  uint32_t reserved;
};

struct Record {
  uint32_t id;
  uint32_t pho_len;
  uint32_t pho[MAX_PHO];
  uint32_t flags;
};

static inline uint32_t fnv1a(std::string_view s) {
  uint32_t h = 2166136261u;
  for (unsigned char c : s) { h ^= c; h *= 16777619u; }
  return h;
}

static inline std::string lower_ascii(std::string s) {
  for (char &c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : s) {
    unsigned char c = (unsigned char)ch;
    if (std::isalnum(c)) cur.push_back((char)std::tolower(c));
    else if (!cur.empty()) { out.push_back(cur); cur.clear(); }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

// Tiny PHO encoder prototype: deterministic, bounded, no reverse resolve here.
// It maps word prefixes/suffixes/shapes to compact integer IDs.
static std::vector<uint32_t> encode_pho(const std::string& text) {
  auto words = split_words(text);
  std::vector<uint32_t> ids;
  ids.reserve(std::min<int>(MAX_Q, (int)words.size() * 2));
  for (const auto& w0 : words) {
    std::string w = lower_ascii(w0);
    if (w.empty()) continue;
    std::string a = w.substr(0, std::min<size_t>(3, w.size()));
    std::string b = w.size() <= 3 ? w : w.substr(w.size() - std::min<size_t>(3, w.size()));
    uint32_t id1 = 1u + (fnv1a(a) % 8191u);
    uint32_t id2 = 8192u + (fnv1a(b) % 8191u);
    ids.push_back(id1);
    if ((int)ids.size() < MAX_Q) ids.push_back(id2);
    if ((int)ids.size() >= MAX_Q) break;
  }
  if (ids.empty()) ids.push_back(1u + (fnv1a("empty") % 8191u));
  return ids;
}

static void build_demo_index(const char* path) {
  std::vector<std::pair<uint32_t, std::string>> items = {
    {1001, "hello world greeting"},
    {1002, "how are you status"},
    {1003, "deepseek distilled reasoning"},
    {1004, "gpu vram bounded inference"},
    {1005, "pho index retrieval server"},
    {1006, "client only resolve output"},
    {1007, "cuda mmap lookup path"},
    {1008, "deterministic compressed ids"},
    {1009, "pipeline kv issue debug"},
    {1010, "single gpu fallback"},
    {1011, "asymmetric vram routing"},
    {1012, "ssd backed external memory"},
    {1013, "top speed low memory"},
    {1014, "native converter runtime"},
    {1015, "chat query response ids"},
    {1016, "semantic retrieval bounded"}
  };

  int fd = ::open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) { perror("open"); exit(1); }

  IndexHeader hdr{};
  hdr.magic = MAGIC;
  hdr.version = 1;
  hdr.record_count = (uint32_t)items.size();
  if (::write(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) { perror("write hdr"); exit(1); }

  for (auto& kv : items) {
    Record r{};
    r.id = kv.first;
    auto pho = encode_pho(kv.second);
    r.pho_len = (uint32_t)std::min<size_t>(MAX_PHO, pho.size());
    for (uint32_t i = 0; i < r.pho_len; ++i) r.pho[i] = pho[i];
    if (::write(fd, &r, sizeof(r)) != (ssize_t)sizeof(r)) { perror("write rec"); exit(1); }
  }
  ::close(fd);
  std::cout << "[*] Built demo index: " << path << " (" << items.size() << " records)\n";
}

__global__ void score_kernel(const uint32_t* rec_pho, const uint32_t* rec_len,
                             const uint32_t* q_pho, int q_len,
                             float* scores, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  uint32_t base = (uint32_t)i * MAX_PHO;
  float score = 0.0f;
  for (uint32_t a = 0; a < rec_len[i]; ++a) {
    uint32_t x = rec_pho[base + a];
    for (int b = 0; b < q_len; ++b) {
      if (x == q_pho[b]) score += 1.0f;
    }
  }
  // Mild length prior
  score -= 0.05f * abs((int)rec_len[i] - q_len);
  scores[i] = score;
}

static void server_query(const char* index_path, int topk, int cand_limit) {
  int fd = ::open(index_path, O_RDONLY);
  if (fd < 0) { perror("open"); exit(1); }
  struct stat st{};
  if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); exit(1); }

  auto* hdr = reinterpret_cast<const IndexHeader*>(map);
  if (hdr->magic != MAGIC) {
    std::cerr << "bad index magic\n";
    exit(1);
  }
  auto* recs = reinterpret_cast<const Record*>((const char*)map + sizeof(IndexHeader));
  int n_all = (int)hdr->record_count;
  int n = std::min(n_all, cand_limit);

  std::vector<uint32_t> h_ids(n), h_len(n), h_pho((size_t)n * MAX_PHO);
  for (int i = 0; i < n; ++i) {
    h_ids[i] = recs[i].id;
    h_len[i] = recs[i].pho_len;
    for (int j = 0; j < MAX_PHO; ++j) h_pho[(size_t)i * MAX_PHO + j] = recs[i].pho[j];
  }

  uint32_t *d_pho = nullptr, *d_len = nullptr, *d_q = nullptr;
  float *d_scores = nullptr;
  CUDA_OK(cudaMalloc(&d_pho, h_pho.size() * sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&d_len, h_len.size() * sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&d_q, MAX_Q * sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&d_scores, n * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_pho, h_pho.data(), h_pho.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_len, h_len.data(), h_len.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  std::cout << "[server] PHO-only mode. Type text, get PHO/index IDs only. /quit to exit.\n";
  for (;;) {
    std::cout << "> " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) break;
    if (line == "/quit") break;

    auto q = encode_pho(line);
    int q_len = (int)std::min<size_t>(q.size(), MAX_Q);
    CUDA_OK(cudaMemcpy(d_q, q.data(), q_len * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;
    score_kernel<<<grid, block>>>(d_pho, d_len, d_q, q_len, d_scores, n);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> h_scores(n);
    CUDA_OK(cudaMemcpy(h_scores.data(), d_scores, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(topk, n), idx.end(),
      [&](int a, int b) { return h_scores[a] > h_scores[b]; });

    std::cout << "{ \"q_pho\": [";
    for (int i = 0; i < q_len; ++i) {
      if (i) std::cout << ", ";
      std::cout << q[i];
    }
    std::cout << "], \"top\": [";
    for (int i = 0; i < std::min(topk, n); ++i) {
      int k = idx[i];
      if (i) std::cout << ", ";
      std::cout << "{ \"index_id\": " << h_ids[k] << ", \"score\": " << h_scores[k] << ", \"pho\": [";
      for (uint32_t j = 0; j < h_len[k]; ++j) {
        if (j) std::cout << ", ";
        std::cout << h_pho[(size_t)k * MAX_PHO + j];
      }
      std::cout << "] }";
    }
    std::cout << "] }\n";
  }

  cudaFree(d_scores);
  cudaFree(d_q);
  cudaFree(d_len);
  cudaFree(d_pho);
  munmap(map, st.st_size);
  close(fd);
}

// Optional local/client-only resolve; server never uses this.
static std::string local_resolve(uint32_t id) {
  switch (id) {
    case 1001: return "hello world greeting";
    case 1002: return "how are you status";
    case 1003: return "deepseek distilled reasoning";
    case 1004: return "gpu vram bounded inference";
    case 1005: return "pho index retrieval server";
    case 1006: return "client only resolve output";
    case 1007: return "cuda mmap lookup path";
    case 1008: return "deterministic compressed ids";
    case 1009: return "pipeline kv issue debug";
    case 1010: return "single gpu fallback";
    case 1011: return "asymmetric vram routing";
    case 1012: return "ssd backed external memory";
    case 1013: return "top speed low memory";
    case 1014: return "native converter runtime";
    case 1015: return "chat query response ids";
    case 1016: return "semantic retrieval bounded";
    default: return "<unknown>";
  }
}

static void client_resolve_mode() {
  std::cout << "[client] Enter numeric index IDs to resolve locally. /quit to exit.\n";
  for (;;) {
    std::cout << "id> " << std::flush;
    std::string s;
    if (!std::getline(std::cin, s)) break;
    if (s == "/quit") break;
    uint32_t id = (uint32_t)std::strtoul(s.c_str(), nullptr, 10);
    std::cout << local_resolve(id) << "\n";
  }
}

int main(int argc, char** argv) {
  std::string mode = "--server";
  std::string index = "pho_demo.index";
  int topk = 8;
  int cand_limit = 256;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--build") mode = a;
    else if (a == "--server") mode = a;
    else if (a == "--client") mode = a;
    else if (a == "--index" && i + 1 < argc) index = argv[++i];
    else if (a == "--topk" && i + 1 < argc) topk = std::atoi(argv[++i]);
    else if (a == "--cand-limit" && i + 1 < argc) cand_limit = std::atoi(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0] << " [--build|--server|--client] [--index PATH] [--topk N] [--cand-limit N]\n";
      return 1;
    }
  }

  if (mode == "--build") {
    build_demo_index(index.c_str());
    return 0;
  }
  if (mode == "--client") {
    client_resolve_mode();
    return 0;
  }
  server_query(index.c_str(), topk, cand_limit);
  return 0;
}
