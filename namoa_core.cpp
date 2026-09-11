// namoa_core.cpp — exact 3-obj NAMOA* (+ optional profiler / top-node dump)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <string>

using namespace std;
using Steady = chrono::steady_clock;

static inline int64_t ns_since(Steady::time_point t0) {
    return chrono::duration_cast<chrono::nanoseconds>(Steady::now() - t0).count();
}

static inline bool dom_le(int64_t a0, int64_t a1, int64_t a2,
                          int64_t b0, int64_t b1, int64_t b2) {
    return a0 <= b0 && a1 <= b1 && a2 <= b2;
}

struct Cost3 { int64_t c0, c1, c2; };

struct HeapItem {
    int64_t f0, f1, f2, g0, g1, g2;
    int lid;
    bool operator<(const HeapItem& o) const {
        if (f0 != o.f0) return f0 > o.f0;
        if (f1 != o.f1) return f1 > o.f1;
        if (f2 != o.f2) return f2 > o.f2;
        if (g0 != o.g0) return g0 > o.g0;
        if (g1 != o.g1) return g1 > o.g1;
        if (g2 != o.g2) return g2 > o.g2;
        return lid > o.lid;
    }
};

struct Label {
    int64_t g0, g1, g2;
    int32_t node;
    uint8_t active;
};

struct GoalIndex {
    static const int BLOCK = 64;
    vector<Cost3> pts;
    vector<int64_t> blk_min1, blk_min2;
    bool dirty = true;

    void rebuild() {
        sort(pts.begin(), pts.end(), [](const Cost3& a, const Cost3& b) {
            if (a.c0 != b.c0) return a.c0 < b.c0;
            if (a.c1 != b.c1) return a.c1 < b.c1;
            return a.c2 < b.c2;
        });
        pts.erase(unique(pts.begin(), pts.end(),
                         [](const Cost3& a, const Cost3& b) {
                             return a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2;
                         }),
                  pts.end());
        int n = (int)pts.size();
        int nb = (n + BLOCK - 1) / BLOCK;
        blk_min1.assign(nb, 0);
        blk_min2.assign(nb, 0);
        for (int b = 0; b < nb; ++b) {
            int L = b * BLOCK, R = min(n, L + BLOCK);
            int64_t m1 = pts[L].c1, m2 = pts[L].c2;
            for (int i = L + 1; i < R; ++i) {
                if (pts[i].c1 < m1) m1 = pts[i].c1;
                if (pts[i].c2 < m2) m2 = pts[i].c2;
            }
            blk_min1[b] = m1;
            blk_min2[b] = m2;
        }
        dirty = false;
    }
    void ensure() { if (dirty) rebuild(); }
    size_t size() { ensure(); return pts.size(); }

    bool dominated(int64_t y0, int64_t y1, int64_t y2) {
        ensure();
        int n = (int)pts.size();
        if (!n) return false;
        int nb = (n + BLOCK - 1) / BLOCK;
        for (int b = 0; b < nb; ++b) {
            int L = b * BLOCK, R = min(n, L + BLOCK);
            if (pts[L].c0 > y0) break;
            if (blk_min1[b] > y1 || blk_min2[b] > y2) continue;
            for (int i = L; i < R; ++i) {
                if (pts[i].c0 > y0) break;
                if (dom_le(pts[i].c0, pts[i].c1, pts[i].c2, y0, y1, y2)) return true;
            }
        }
        return false;
    }

    bool update(int64_t y0, int64_t y1, int64_t y2) {
        ensure();
        for (const auto& z : pts)
            if (dom_le(z.c0, z.c1, z.c2, y0, y1, y2)) return false;
        vector<Cost3> kept;
        kept.reserve(pts.size() + 1);
        for (const auto& z : pts)
            if (!dom_le(y0, y1, y2, z.c0, z.c1, z.c2)) kept.push_back(z);
        kept.push_back(Cost3{y0, y1, y2});
        pts.swap(kept);
        dirty = true;
        return true;
    }
};

struct NodeProf {
    int64_t peak_front = 0;
    int64_t insert_attempts = 0;
    int64_t accepted = 0;
    int64_t dominated_reject = 0;
    int64_t old_killed = 0;
    int64_t expansions = 0;
    int64_t generated_children = 0;
    int64_t node_dom_ns = 0;
    int64_t scan_sum = 0;
    int64_t scan_max = 0;
    int64_t scan_queries = 0;
};

#include "node_front_cached.inc"
static void refresh_corridor(
    int n_nodes, const int64_t* offset, const int32_t* to,
    const int64_t* w0, const int64_t* w1, const int64_t* w2,
    const int64_t* h0, const int64_t* h1, const int64_t* h2,
    const int64_t* ds0, const int64_t* ds1, const int64_t* ds2,
    uint8_t* edge_allowed, GoalIndex& S, GoalIndex& U, int64_t& edges_killed) {
    const int64_t BIG = (1LL << 60);
    auto bad = [&](int64_t L0, int64_t L1, int64_t L2) {
        return S.dominated(L0, L1, L2) || U.dominated(L0, L1, L2);
    };
    for (int u = 0; u < n_nodes; ++u) {
        for (int64_t e = offset[u]; e < offset[u + 1]; ++e) {
            if (!edge_allowed[e]) continue;
            int v = to[e];
            if (ds0[u] >= BIG || h0[v] >= BIG) {
                edge_allowed[e] = 0; ++edges_killed; continue;
            }
            int64_t L0 = ds0[u] + w0[e] + h0[v];
            int64_t L1 = ds1[u] + w1[e] + h1[v];
            int64_t L2 = ds2[u] + w2[e] + h2[v];
            if (bad(L0, L1, L2)) { edge_allowed[e] = 0; ++edges_killed; }
        }
    }
}

static int64_t percentile_sorted(vector<int32_t>& a, double p) {
    if (a.empty()) return 0;
    size_t i = (size_t)((a.size() - 1) * p);
    return a[i];
}

// Optional target-side suffix Pareto cache (set via namoa_set_suffix_pareto).
// sp_exact[v]=1 ⇒ ∀p∈P_t(v) may be used for exact goal prune.
// sp_join=1 ⇒ also inject g+p into Sexact as feasible incumbents.
static const int64_t* g_sp_off = nullptr;
static const int64_t* g_sp_c0 = nullptr;
static const int64_t* g_sp_c1 = nullptr;
static const int64_t* g_sp_c2 = nullptr;
static const uint8_t* g_sp_exact = nullptr;
static int32_t g_sp_join = 0;
static int64_t g_sp_prune_hits = 0;
static int64_t g_sp_join_updates = 0;

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
void namoa_set_suffix_pareto(
    const int64_t* off,
    const int64_t* c0, const int64_t* c1, const int64_t* c2,
    const uint8_t* exact,
    int32_t enable_join
) {
    g_sp_off = off;
    g_sp_c0 = c0; g_sp_c1 = c1; g_sp_c2 = c2;
    g_sp_exact = exact;
    g_sp_join = enable_join;
    g_sp_prune_hits = 0;
    g_sp_join_updates = 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void namoa_clear_suffix_pareto(void) {
    g_sp_off = nullptr;
    g_sp_c0 = g_sp_c1 = g_sp_c2 = nullptr;
    g_sp_exact = nullptr;
    g_sp_join = 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int namoa_search(
    int32_t n_nodes,
    const int64_t* offset,
    const int32_t* to,
    const int64_t* w0, const int64_t* w1, const int64_t* w2,
    const int64_t* h0, const int64_t* h1, const int64_t* h2,
    int32_t source, int32_t target, int32_t use_h,
    const int64_t* seed0, const int64_t* seed1, const int64_t* seed2, int32_t n_seed,
    uint8_t* edge_allowed,
    int64_t* out_c1, int64_t* out_c2, int64_t* out_c3, int32_t max_out,
    int64_t* stats_out, int32_t max_labels,
    int64_t max_expanded,
    const int64_t* suffix_q,
    const int64_t* ds0, const int64_t* ds1, const int64_t* ds2
) {
    // Delegate to profiled entry with profiling disabled (out_dir=null).
    // Declared below.
    extern int namoa_search_profile(
        int32_t, const int64_t*, const int32_t*,
        const int64_t*, const int64_t*, const int64_t*,
        const int64_t*, const int64_t*, const int64_t*,
        int32_t, int32_t, int32_t,
        const int64_t*, const int64_t*, const int64_t*, int32_t,
        uint8_t*,
        int64_t*, int64_t*, int64_t*, int32_t,
        int64_t*, int32_t, int64_t,
        const int64_t*, const int64_t*, const int64_t*, const int64_t*,
        const char*);
    return namoa_search_profile(
        n_nodes, offset, to, w0, w1, w2, h0, h1, h2,
        source, target, use_h, seed0, seed1, seed2, n_seed,
        edge_allowed, out_c1, out_c2, out_c3, max_out,
        stats_out, max_labels, max_expanded,
        suffix_q, ds0, ds1, ds2, nullptr);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int namoa_search_profile(
    int32_t n_nodes,
    const int64_t* offset,
    const int32_t* to,
    const int64_t* w0, const int64_t* w1, const int64_t* w2,
    const int64_t* h0, const int64_t* h1, const int64_t* h2,
    int32_t source, int32_t target, int32_t use_h,
    const int64_t* seed0, const int64_t* seed1, const int64_t* seed2, int32_t n_seed,
    uint8_t* edge_allowed,
    int64_t* out_c1, int64_t* out_c2, int64_t* out_c3, int32_t max_out,
    int64_t* stats_out, int32_t max_labels,
    int64_t max_expanded,
    const int64_t* suffix_q,
    const int64_t* ds0, const int64_t* ds1, const int64_t* ds2,
    const char* out_dir
) {
    const bool prof = (out_dir != nullptr && out_dir[0] != '\0');
    g_labels_scanned = 0; g_blocks_skipped = 0;

    GoalIndex Sexact, Ubound;
    for (int i = 0; i < n_seed; ++i) Sexact.update(seed0[i], seed1[i], seed2[i]);

    vector<Label> labels; labels.reserve(1 << 20);
    vector<NodeFront> fronts(n_nodes);
    priority_queue<HeapItem> open;
    vector<NodeProf> nprof;
    if (prof) nprof.resize(n_nodes);

    int64_t gen = 0, expanded = 0, peak_live = 0, live = 0;
    int64_t node_prune = 0, goal_prune = 0, dup_prune = 0, max_at_node = 0;
    int64_t stale_skipped = 0, open_peak = 0, edges_killed = 0;
    int64_t last_S_refresh = 0, last_U_refresh = 0;
    int next_refresh_threshold = 256;
    int64_t refresh_count = 0;

    int64_t t_heap_pop = 0, t_heap_push = 0;
    int64_t t_find = 0, t_del = 0, t_ins = 0;
    int64_t t_goal = 0, t_edge = 0, t_corr = 0, t_other = 0;
    vector<int32_t> scan_samp;
    if (prof) scan_samp.reserve(250000);
    const int scan_cap = 200000;

    // Optional insert traces for wide nodes (peak>=512): store g triples
    vector<vector<int64_t>> traces; // flattened g0,g1,g2
    vector<uint8_t> tracing;
    if (prof) {
        traces.resize(n_nodes);
        tracing.assign(n_nodes, 0);
    }
    const int TRACE_PEAK = 512;
    const int TRACE_MAX_EVENTS = 80000; // triples → 240k int64 max per node

    auto maybe_refresh = [&]() {
        if (!edge_allowed || !ds0) return;
        auto tc = Steady::now();
        int64_t ssz = (int64_t)Sexact.size();
        int64_t usz = (int64_t)Ubound.size();
        bool hit_S = ssz >= next_refresh_threshold && ssz > last_S_refresh;
        bool hit_U = usz >= next_refresh_threshold && usz > last_U_refresh;
        if (!hit_S && !hit_U) return;
        refresh_corridor(n_nodes, offset, to, w0, w1, w2, h0, h1, h2,
                         ds0, ds1, ds2, edge_allowed, Sexact, Ubound, edges_killed);
        last_S_refresh = ssz;
        last_U_refresh = usz;
        while (next_refresh_threshold <= ssz || next_refresh_threshold <= usz)
            next_refresh_threshold *= 2;
        ++refresh_count;
        t_corr += ns_since(tc);
    };

    int ret_nsol = 0;

    auto fill_stats = [&](int64_t status) {
        stats_out[0] = (int64_t)Sexact.size();
        stats_out[1] = gen; stats_out[2] = expanded; stats_out[3] = peak_live;
        stats_out[4] = node_prune; stats_out[5] = goal_prune; stats_out[6] = dup_prune;
        stats_out[7] = max_at_node; stats_out[8] = (int64_t)labels.size();
        stats_out[9] = stale_skipped; stats_out[10] = open_peak;
        stats_out[11] = status;
        stats_out[12] = (int64_t)Ubound.size();
        stats_out[13] = edges_killed;
        stats_out[14] = refresh_count;
        stats_out[15] = live;
        stats_out[16] = t_heap_pop / 1000000;
        stats_out[17] = t_heap_push / 1000000;
        stats_out[18] = t_find / 1000000;
        stats_out[19] = t_del / 1000000;
        stats_out[20] = t_ins / 1000000;
        stats_out[21] = t_goal / 1000000;
        stats_out[22] = t_edge / 1000000;
        stats_out[23] = t_corr / 1000000;
        stats_out[24] = (t_find + t_del + t_ins) / 1000000;
        stats_out[25] = t_other / 1000000;
        stats_out[26] = g_sp_prune_hits;
        stats_out[27] = g_sp_join_updates;
    };

    auto finish_solutions = [&](int64_t status) {
        Sexact.ensure();
        int nsol = (int)Sexact.pts.size();
        if (nsol > max_out) nsol = max_out;
        for (int i = 0; i < nsol; ++i) {
            out_c1[i] = Sexact.pts[i].c0;
            out_c2[i] = Sexact.pts[i].c1;
            out_c3[i] = Sexact.pts[i].c2;
        }
        fill_stats(status);
        ret_nsol = nsol;
    };

    auto push_open = [&](HeapItem it) {
        auto tp = Steady::now();
        open.push(it);
        if (prof) t_heap_push += ns_since(tp);
        if ((int64_t)open.size() > open_peak) open_peak = (int64_t)open.size();
    };

    // Target-side suffix Pareto: ∀p ∃y y≤g+p (only if sp_exact[v]).
    auto suffix_all_covered = [&](int u, int64_t a, int64_t b, int64_t c) -> bool {
        if (!g_sp_off || !g_sp_exact || !g_sp_exact[u]) return false;
        int64_t L = g_sp_off[u], R = g_sp_off[u + 1];
        if (L >= R) return false;
        for (int64_t i = L; i < R; ++i) {
            if (!Sexact.dominated(a + g_sp_c0[i], b + g_sp_c1[i], c + g_sp_c2[i]))
                return false;
        }
        ++g_sp_prune_hits;
        return true;
    };

    auto maybe_join_suffix = [&](int u, int64_t a, int64_t b, int64_t c) {
        if (!g_sp_join || !g_sp_off) return;
        int64_t L = g_sp_off[u], R = g_sp_off[u + 1];
        for (int64_t i = L; i < R; ++i) {
            if (Sexact.update(a + g_sp_c0[i], b + g_sp_c1[i], c + g_sp_c2[i])) {
                ++g_sp_join_updates;
                maybe_refresh();
            }
        }
    };

    auto add_label = [&](int u, int64_t na, int64_t nb, int64_t nc) -> int {
        NodeProf* np = prof ? &nprof[u] : nullptr;
        int id = try_add_front(labels, fronts[u], u, na, nb, nc, max_labels,
                              live, gen, node_prune, dup_prune, max_at_node, peak_live,
                              np, prof ? &t_find : nullptr, prof ? &t_del : nullptr,
                              prof ? &t_ins : nullptr,
                              prof ? &scan_samp : nullptr, scan_cap,
                              prof ? &t_other : nullptr);
        if (prof && id >= 0 && nprof[u].peak_front >= TRACE_PEAK) {
            tracing[u] = 1;
        }
        if (prof && tracing[u] && (int)traces[u].size() < TRACE_MAX_EVENTS * 3) {
            traces[u].push_back(na);
            traces[u].push_back(nb);
            traces[u].push_back(nc);
        }
        return id;
    };

    int lid0 = add_label(source, 0, 0, 0);
    if (lid0 < 0) {
        fill_stats(1);
        return (int)Sexact.size();
    }
    {
        HeapItem it;
        if (use_h) { it.f0 = h0[source]; it.f1 = h1[source]; it.f2 = h2[source]; }
        else { it.f0 = it.f1 = it.f2 = 0; }
        it.g0 = it.g1 = it.g2 = 0; it.lid = lid0;
        push_open(it);
    }

    while (!open.empty()) {
        auto tp = Steady::now();
        HeapItem cur = open.top(); open.pop();
        if (prof) t_heap_pop += ns_since(tp);

        int lid = cur.lid;
        if (!labels[lid].active) { ++stale_skipped; continue; }
        int v = labels[lid].node;
        int64_t a = labels[lid].g0, b = labels[lid].g1, c = labels[lid].g2;
        ++expanded;
        if (prof) nprof[v].expansions++;

        if (max_expanded > 0 && expanded >= max_expanded) {
            finish_solutions(2);
            goto dump_profile;
        }

        if ((expanded % 1000000) == 0) {
            std::fprintf(stderr,
                "[namoa] exp=%lld gen=%lld live=%lld open=%zu |S|=%zu |U|=%zu "
                "rg=%.3f max_node=%lld t_node_ms=%lld t_heap_ms=%lld\n",
                (long long)expanded, (long long)gen, (long long)live, open.size(),
                Sexact.size(), Ubound.size(),
                (double)gen / (double)expanded, (long long)max_at_node,
                (long long)((t_find + t_del + t_ins) / 1000000),
                (long long)((t_heap_pop + t_heap_push) / 1000000));
            std::fflush(stderr);
        }

        if (suffix_q) {
            for (int k = 0; k < 3; ++k) {
                int64_t base = (int64_t)v * 9 + (int64_t)k * 3;
                int64_t q0 = suffix_q[base], q1 = suffix_q[base + 1], q2 = suffix_q[base + 2];
                if (q0 < 0 || q1 < 0 || q2 < 0 || q0 > (1LL << 60)) continue;
                if (Ubound.update(a + q0, b + q1, c + q2)) maybe_refresh();
            }
        }

        maybe_join_suffix(v, a, b, c);

        if (v == target) {
            auto tg = Steady::now();
            bool dom = Sexact.dominated(a, b, c);
            if (prof) t_goal += ns_since(tg);
            if (dom) { ++goal_prune; continue; }
            Sexact.update(a, b, c);
            maybe_refresh();
            continue;
        }

        {
            auto tg = Steady::now();
            int64_t ff0 = use_h ? a + h0[v] : a;
            int64_t ff1 = use_h ? b + h1[v] : b;
            int64_t ff2 = use_h ? c + h2[v] : c;
            bool dom = Sexact.dominated(ff0, ff1, ff2) || Ubound.dominated(ff0, ff1, ff2)
                       || suffix_all_covered(v, a, b, c);
            if (prof) t_goal += ns_since(tg);
            if (dom) { ++goal_prune; continue; }
        }

        auto te = Steady::now();
        for (int64_t e = offset[v]; e < offset[v + 1]; ++e) {
            if (edge_allowed && edge_allowed[e] == 0) continue;
            int u = to[e];
            int64_t na = a + w0[e], nb = b + w1[e], nc = c + w2[e];
            if (prof) nprof[v].generated_children++;

            if (u == target) {
                auto tg = Steady::now();
                bool dom = Sexact.dominated(na, nb, nc);
                if (prof) t_goal += ns_since(tg);
                if (dom) { ++goal_prune; continue; }
                Sexact.update(na, nb, nc);
                maybe_refresh();
                add_label(u, na, nb, nc);
                continue;
            }

            {
                auto tg = Steady::now();
                int64_t nf0 = use_h ? na + h0[u] : na;
                int64_t nf1 = use_h ? nb + h1[u] : nb;
                int64_t nf2 = use_h ? nc + h2[u] : nc;
                bool gdom = Sexact.dominated(nf0, nf1, nf2) || Ubound.dominated(nf0, nf1, nf2)
                            || suffix_all_covered(u, na, nb, nc);
                if (prof) t_goal += ns_since(tg);
                if (gdom) { ++goal_prune; continue; }

                int new_id = add_label(u, na, nb, nc);
                if (new_id == -1) continue;
                if (new_id == -2) {
                    fill_stats(0);
                    stats_out[0] = -1;
                    goto dump_profile_fail;
                }
                push_open(HeapItem{nf0, nf1, nf2, na, nb, nc, new_id});
            }
        }
        if (prof) t_edge += ns_since(te);
    }

    finish_solutions(1);

dump_profile:
    if (prof) {
        // write summary + top100 + traces
        string dir(out_dir);
        auto path = [&](const char* name) { return dir + "/" + name; };

        FILE* fs = std::fopen(path("profiler_summary.csv").c_str(), "w");
        if (fs) {
            int64_t node_ms = (t_find + t_del + t_ins) / 1000000;
            int64_t heap_ms = (t_heap_pop + t_heap_push) / 1000000;
            int64_t total_ms = node_ms + heap_ms + t_goal / 1000000 + t_edge / 1000000 + t_corr / 1000000;
            if (total_ms <= 0) total_ms = 1;
            std::fprintf(fs,
                "metric,value\n"
                "expanded,%lld\ngenerated,%lld\nlive,%lld\nmax_node,%lld\n"
                "heap_pop_ms,%lld\nheap_push_ms,%lld\n"
                "node_dom_query_ms,%lld\nnode_dom_delete_ms,%lld\nnode_insert_ms,%lld\n"
                "node_skyline_ms,%lld\n"
                "goal_dom_query_ms,%lld\nedge_expand_ms,%lld\ncorridor_ms,%lld\n"
                "total_accounted_ms,%lld\n"
                "node_frac,%.4f\nheap_frac,%.4f\ngoal_frac,%.4f\nedge_frac,%.4f\n"
                "labels_scanned,%lld\nblocks_skipped,%lld\nsky_B,%d\nsky_switch,%d\n",
                (long long)expanded, (long long)gen, (long long)live, (long long)max_at_node,
                (long long)(t_heap_pop / 1000000), (long long)(t_heap_push / 1000000),
                (long long)(t_find / 1000000), (long long)(t_del / 1000000), (long long)(t_ins / 1000000),
                (long long)node_ms,
                (long long)(t_goal / 1000000), (long long)(t_edge / 1000000), (long long)(t_corr / 1000000),
                (long long)total_ms,
                (double)node_ms / total_ms, (double)heap_ms / total_ms,
                (double)(t_goal / 1000000) / total_ms, (double)(t_edge / 1000000) / total_ms,
                (long long)g_labels_scanned, (long long)g_blocks_skipped, g_sky_B, g_sky_switch);
            std::fclose(fs);
        }

        sort(scan_samp.begin(), scan_samp.end());
        FILE* fp = std::fopen(path("scan_percentiles.csv").c_str(), "w");
        if (fp) {
            std::fprintf(fp, "n_samples,avg,P50,P90,P95,P99,P99_9,max\n");
            double avg = 0;
            if (!scan_samp.empty()) {
                int64_t s = 0;
                for (auto x : scan_samp) s += x;
                avg = (double)s / scan_samp.size();
            }
            int32_t mx = scan_samp.empty() ? 0 : scan_samp.back();
            std::fprintf(fp, "%zu,%.3f,%lld,%lld,%lld,%lld,%lld,%d\n",
                         scan_samp.size(), avg,
                         (long long)percentile_sorted(scan_samp, 0.50),
                         (long long)percentile_sorted(scan_samp, 0.90),
                         (long long)percentile_sorted(scan_samp, 0.95),
                         (long long)percentile_sorted(scan_samp, 0.99),
                         (long long)percentile_sorted(scan_samp, 0.999),
                         (int)mx);
            std::fclose(fp);
        }

        vector<int> order(n_nodes);
        for (int i = 0; i < n_nodes; ++i) order[i] = i;
        sort(order.begin(), order.end(), [&](int i, int j) {
            if (nprof[i].peak_front != nprof[j].peak_front)
                return nprof[i].peak_front > nprof[j].peak_front;
            return nprof[i].node_dom_ns > nprof[j].node_dom_ns;
        });

        int64_t total_node_ns = 0;
        for (int i = 0; i < n_nodes; ++i) total_node_ns += nprof[i].node_dom_ns;
        if (total_node_ns <= 0) total_node_ns = 1;
        int64_t top10_ns = 0, top100_ns = 0;
        for (int k = 0; k < n_nodes && k < 10; ++k) top10_ns += nprof[order[k]].node_dom_ns;
        for (int k = 0; k < n_nodes && k < 100; ++k) top100_ns += nprof[order[k]].node_dom_ns;

        FILE* fr = std::fopen(path("concentration.csv").c_str(), "w");
        if (fr) {
            std::fprintf(fr, "R10,R100,total_node_dom_ms,top10_ms,top100_ms\n");
            std::fprintf(fr, "%.6f,%.6f,%.3f,%.3f,%.3f\n",
                         (double)top10_ns / total_node_ns,
                         (double)top100_ns / total_node_ns,
                         total_node_ns / 1e6, top10_ns / 1e6, top100_ns / 1e6);
            std::fclose(fr);
        }

        FILE* ft = std::fopen(path("top100_wide_nodes.csv").c_str(), "w");
        if (ft) {
            std::fprintf(ft,
                "rank,node,peak_front,current_front,insert_attempts,accepted,"
                "dominated_reject,old_killed,expansions,generated_children,"
                "node_dom_ms,avg_scan,max_scan,out_degree,in_degree,"
                "d1,d2,d3,h1,h2,h3,p1,p2,p3\n");
            // in-degree from reverse scan of CSR
            vector<int32_t> indeg(n_nodes, 0), outdeg(n_nodes, 0);
            for (int u = 0; u < n_nodes; ++u) {
                outdeg[u] = (int32_t)(offset[u + 1] - offset[u]);
                for (int64_t e = offset[u]; e < offset[u + 1]; ++e)
                    indeg[to[e]]++;
            }
            int nout = min(100, n_nodes);
            for (int r = 0; r < nout; ++r) {
                int v = order[r];
                const auto& np = nprof[v];
                if (np.peak_front == 0 && np.insert_attempts == 0) break;
                int cur = 0;
                cur = fronts[v].live_size();
                double avg = np.scan_queries ? (double)np.scan_sum / np.scan_queries : 0.0;
                int64_t d1 = ds0 ? ds0[v] : -1, d2 = ds1 ? ds1[v] : -1, d3 = ds2 ? ds2[v] : -1;
                int64_t hh1 = h0[v], hh2 = h1[v], hh3 = h2[v];
                auto pi = [](int64_t d, int64_t h) -> double {
                    if (d < 0 || h < 0) return -1;
                    double den = (double)d + (double)h;
                    return den > 0 ? (double)d / den : 0;
                };
                std::fprintf(ft,
                    "%d,%d,%lld,%d,%lld,%lld,%lld,%lld,%lld,%lld,%.3f,%.3f,%lld,%d,%d,"
                    "%lld,%lld,%lld,%lld,%lld,%lld,%.4f,%.4f,%.4f\n",
                    r + 1, v, (long long)np.peak_front, cur,
                    (long long)np.insert_attempts, (long long)np.accepted,
                    (long long)np.dominated_reject, (long long)np.old_killed,
                    (long long)np.expansions, (long long)np.generated_children,
                    np.node_dom_ns / 1e6, avg, (long long)np.scan_max,
                    (int)outdeg[v], (int)indeg[v],
                    (long long)d1, (long long)d2, (long long)d3,
                    (long long)hh1, (long long)hh2, (long long)hh3,
                    pi(d1, hh1), pi(d2, hh2), pi(d3, hh3));
            }
            std::fclose(ft);
        }

        // write insert traces for top-10 peak nodes
        for (int r = 0; r < 10 && r < n_nodes; ++r) {
            int v = order[r];
            if (traces[v].empty()) continue;
            char name[128];
            std::snprintf(name, sizeof(name), "trace_node_%d.bin", v);
            FILE* tb = std::fopen(path(name).c_str(), "wb");
            if (!tb) continue;
            int64_t ntrip = (int64_t)traces[v].size() / 3;
            std::fwrite(&ntrip, sizeof(ntrip), 1, tb);
            std::fwrite(traces[v].data(), sizeof(int64_t), traces[v].size(), tb);
            std::fclose(tb);
        }
    }
    return ret_nsol;

dump_profile_fail:
    fill_stats(0);
    stats_out[0] = -1;
    if (prof) {
        // partial dump already attempted via fill_stats only
    }
    return -1;
}

} // extern C
