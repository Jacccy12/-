// p3_weighted_core.cpp — fast weighted-sum Dijkstra candidate generator for problem 3
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <vector>

using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;

struct SItem {
    long double d;
    i32 v;
};
struct SGreater {
    bool operator()(const SItem& a, const SItem& b) const {
        return a.d != b.d ? a.d > b.d : a.v > b.v;
    }
};

struct Workspace {
    std::vector<long double> dist;
    std::vector<i32> pred;
    std::vector<u32> seen;
    u32 epoch = 0;
    explicit Workspace(i32 n) : dist(n), pred(n, -1), seen(n, 0) {}
    void next() {
        if (++epoch == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            epoch = 1;
        }
    }
};

static long double radical_inverse(u32 n, u32 base) {
    long double f = 1, res = 0;
    while (n) {
        f /= base;
        res += f * (n % base);
        n /= base;
    }
    return res;
}

static void sample_weight(i32 k, i32 m, long double* w) {
    static constexpr u32 primes[5] = {2, 3, 5, 7, 11};
    long double sum = 0;
    for (i32 i = 0; i < m; ++i) {
        long double u = std::max(radical_inverse(static_cast<u32>(k + 1), primes[i]), 1e-12L);
        w[i] = -std::log(u);
        sum += w[i];
    }
    for (i32 i = 0; i < m; ++i) w[i] /= sum;
    for (i32 i = m; i < 5; ++i) w[i] = 0;
}

static bool run_scalar(
    i32 n,
    const i64* offset,
    const i32* to,
    const i64* w5,  // n_edges * 5
    i32 s,
    i32 t,
    const long double* coef,  // 5
    Workspace& ws,
    std::array<i64, 5>& cost,
    std::vector<i32>& path
) {
    ws.next();
    std::priority_queue<SItem, std::vector<SItem>, SGreater> pq;
    ws.seen[s] = ws.epoch;
    ws.dist[s] = 0;
    ws.pred[s] = -1;
    pq.push({0, s});
    while (!pq.empty()) {
        auto x = pq.top();
        pq.pop();
        if (ws.seen[x.v] != ws.epoch || x.d != ws.dist[x.v]) continue;
        if (x.v == t) break;
        for (i64 e = offset[x.v]; e < offset[x.v + 1]; ++e) {
            long double ew = 0;
            const i64* we = w5 + e * 5;
            for (i32 j = 0; j < 5; ++j) ew += coef[j] * static_cast<long double>(we[j]);
            i32 v = to[e];
            long double nd = x.d + ew;
            if (ws.seen[v] != ws.epoch || nd < ws.dist[v]) {
                ws.seen[v] = ws.epoch;
                ws.dist[v] = nd;
                ws.pred[v] = x.v;
                pq.push({nd, v});
            }
        }
    }
    if (ws.seen[t] != ws.epoch || !std::isfinite(ws.dist[t])) return false;
    path.clear();
    for (i32 v = t;; v = ws.pred[v]) {
        path.push_back(v);
        if (v == s) break;
        if (ws.pred[v] < 0) return false;
    }
    std::reverse(path.begin(), path.end());
    cost = {0, 0, 0, 0, 0};
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        i32 u = path[i], v = path[i + 1];
        bool found = false;
        for (i64 e = offset[u]; e < offset[u + 1]; ++e) {
            if (to[e] == v) {
                const i64* we = w5 + e * 5;
                for (i32 j = 0; j < 5; ++j) cost[j] += we[j];
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
int p3_weighted_generate(
    i32 n_nodes,
    const i64* offset,
    const i32* to,
    const i64* w5,
    i32 source0,
    i32 target0,
    i32 m,
    i32 n_samples,
    i64* out_costs,      // max_out * 5
    i32* out_path_len,   // max_out
    i32* out_paths_flat, // path_cap
    i32 max_out,
    i32 path_cap,
    i32* path_used_out
) {
    if (m < 1 || m > 5 || n_samples < 0 || max_out <= 0) return -1;
    Workspace ws(n_nodes);
    struct Pack {
        std::array<i64, 5> cost{};
        std::vector<i32> path;
    };
    std::map<std::array<i64, 5>, Pack> best;

    auto insert = [&](std::array<i64, 5> c, std::vector<i32> path) {
        std::array<i64, 5> key{};
        for (i32 i = 0; i < m; ++i) key[i] = c[i];
        auto it = best.find(key);
        if (it == best.end() || path.size() < it->second.path.size()) {
            best[key] = Pack{c, std::move(path)};
        }
    };

    long double coef[5] = {0, 0, 0, 0, 0};
    std::array<i64, 5> cost{};
    std::vector<i32> path;

    // anchors
    for (i32 j = 0; j < m; ++j) {
        for (i32 i = 0; i < 5; ++i) coef[i] = 0;
        coef[j] = 1;
        if (!run_scalar(n_nodes, offset, to, w5, source0, target0, coef, ws, cost, path)) return -2;
        insert(cost, path);
    }

    // scale from anchors
    i64 lo[5], hi[5];
    bool first = true;
    for (auto& kv : best) {
        for (i32 j = 0; j < m; ++j) {
            if (first) {
                lo[j] = hi[j] = kv.second.cost[j];
            } else {
                lo[j] = std::min(lo[j], kv.second.cost[j]);
                hi[j] = std::max(hi[j], kv.second.cost[j]);
            }
        }
        first = false;
    }
    long double scale[5];
    for (i32 j = 0; j < m; ++j) scale[j] = static_cast<long double>(std::max<i64>(hi[j] - lo[j], 1));

    // equal weight
    for (i32 i = 0; i < 5; ++i) coef[i] = 0;
    for (i32 i = 0; i < m; ++i) coef[i] = (1.0L / m) / scale[i];
    if (run_scalar(n_nodes, offset, to, w5, source0, target0, coef, ws, cost, path)) insert(cost, path);

    i32 interior = std::max(0, n_samples - (m + 1));
    for (i32 k = 0; k < interior; ++k) {
        long double w[5] = {0, 0, 0, 0, 0};
        if (m == 2) {
            long double a = static_cast<long double>(k) / std::max(interior - 1, 1);
            w[0] = a;
            w[1] = 1 - a;
        } else {
            sample_weight(k, m, w);
        }
        for (i32 i = 0; i < m; ++i) coef[i] = w[i] / scale[i];
        for (i32 i = m; i < 5; ++i) coef[i] = 0;
        if (!run_scalar(n_nodes, offset, to, w5, source0, target0, coef, ws, cost, path)) continue;
        insert(cost, path);
    }

    i32 n_out = 0;
    i32 path_used = 0;
    for (auto& kv : best) {
        if (n_out >= max_out) break;
        auto& c = kv.second.cost;
        auto& p = kv.second.path;
        for (i32 j = 0; j < 5; ++j) out_costs[n_out * 5 + j] = c[j];
        if (path_used + static_cast<i32>(p.size()) > path_cap) break;
        out_path_len[n_out] = static_cast<i32>(p.size());
        for (i32 x : p) out_paths_flat[path_used++] = x;
        ++n_out;
    }
    if (path_used_out) *path_used_out = path_used;
    return n_out;
}

}  // extern "C"
