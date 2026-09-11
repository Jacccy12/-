// namoa_dr_lazy_core.cpp — exact 3-obj NAMOA*_dr-lazy (label-setting)
//
// Lex OPEN by f = g+π; permanent fronts store truncated (g2,g3) only (DR).
// Lazy: no Gop merge on generation; dominance vs permanent checked at pop.
// Exact with consistent component-wise π (reverse Dijkstra). Same API shape
// as namoa_search for drop-in A/B (no suffix_q / dyn corridor).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>

using namespace std;
using Steady = chrono::steady_clock;

static inline int64_t ns_since(Steady::time_point t0) {
    return chrono::duration_cast<chrono::nanoseconds>(Steady::now() - t0).count();
}

struct HeapItem {
    int64_t f0, f1, f2, g0, g1, g2;
    int32_t node;
    bool operator<(const HeapItem& o) const {
        if (f0 != o.f0) return f0 > o.f0;
        if (f1 != o.f1) return f1 > o.f1;
        if (f2 != o.f2) return f2 > o.f2;
        if (g0 != o.g0) return g0 > o.g0;
        if (g1 != o.g1) return g1 > o.g1;
        if (g2 != o.g2) return g2 > o.g2;
        return node > o.node;
    }
};

// 2D ND front of truncated costs (a=g2, b=g3), kept sorted by a asc, b desc.
struct Front2D {
    vector<int64_t> a; // g2
    vector<int64_t> b; // g3

    size_t size() const { return a.size(); }

    // true if some (a',b') ⪯ (x,y)
    bool dominated(int64_t x, int64_t y) const {
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] > x) break; // sorted by a ascending
            if (b[i] <= y) return true;
        }
        return false;
    }

    // Insert (x,y); remove points dominated by it. Assumes not already dominated.
    void insert(int64_t x, int64_t y) {
        vector<int64_t> na, nb;
        na.reserve(a.size() + 1);
        nb.reserve(b.size() + 1);
        size_t i = 0;
        while (i < a.size() && a[i] < x) {
            na.push_back(a[i]);
            nb.push_back(b[i]);
            ++i;
        }
        // skip equal-a with worse/equal b (should not happen if caller checked)
        while (i < a.size() && a[i] == x) {
            if (b[i] < y) {
                // existing better on b with same a — should have been dominated check
                na.push_back(a[i]);
                nb.push_back(b[i]);
            }
            ++i;
        }
        na.push_back(x);
        nb.push_back(y);
        while (i < a.size()) {
            if (y <= b[i]) {
                // (x,y) dominates (a[i],b[i]) since x <= a[i] (a sorted) and y <= b[i]
                ++i;
                continue;
            }
            na.push_back(a[i]);
            nb.push_back(b[i]);
            ++i;
        }
        a.swap(na);
        b.swap(nb);
    }
};

struct Sol3 {
    int64_t c0, c1, c2;
};

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
int namoa_dr_lazy_search(
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
    int64_t max_expanded
) {
    vector<Front2D> Gtr(n_nodes);
    Front2D& Gtr_goal = Gtr[target];

    // Full solution costs at target (for output); ND in 3D among sols.
    vector<Sol3> sols;
    auto sol_dom = [&](int64_t a0, int64_t a1, int64_t a2) -> bool {
        for (const auto& z : sols)
            if (z.c0 <= a0 && z.c1 <= a1 && z.c2 <= a2) return true;
        return false;
    };
    auto sol_update = [&](int64_t a0, int64_t a1, int64_t a2) {
        if (sol_dom(a0, a1, a2)) return;
        vector<Sol3> kept;
        kept.reserve(sols.size() + 1);
        for (const auto& z : sols)
            if (!(a0 <= z.c0 && a1 <= z.c1 && a2 <= z.c2)) kept.push_back(z);
        kept.push_back(Sol3{a0, a1, a2});
        sols.swap(kept);
        // Do NOT insert into Gtr_goal here: seeds / out-of-order joins
        // would break the lex DR invariant. Gtr_goal only grows when a
        // label at `target` is permanently expanded in lex-f order.
    };

    for (int i = 0; i < n_seed; ++i)
        sol_update(seed0[i], seed1[i], seed2[i]);

    priority_queue<HeapItem> open;

    int64_t gen = 0, expanded = 0;
    int64_t peak_open = 0, peak_live = 0, peak_perm = 0;
    int64_t lazy_reject = 0, goal_prune = 0, gen_goal_prune = 0;
    int64_t max_node = 0;
    int64_t perm_total = 0;

    auto push_open = [&](HeapItem it) {
        open.push(it);
        ++gen;
        if ((int64_t)open.size() > peak_open) peak_open = (int64_t)open.size();
        int64_t live = (int64_t)open.size() + perm_total;
        if (live > peak_live) peak_live = live;
        if (max_labels > 0 && gen >= max_labels) return false;
        return true;
    };

    {
        HeapItem it;
        it.g0 = it.g1 = it.g2 = 0;
        it.node = source;
        if (use_h) {
            it.f0 = h0[source]; it.f1 = h1[source]; it.f2 = h2[source];
        } else {
            it.f0 = it.f1 = it.f2 = 0;
        }
        if (!push_open(it)) {
            stats_out[0] = (int64_t)sols.size();
            stats_out[1] = gen; stats_out[2] = expanded; stats_out[3] = peak_live;
            stats_out[11] = 0; // LABEL_LIMIT
            return -1;
        }
    }

    auto f_of = [&](int u, int64_t g0, int64_t g1, int64_t g2, HeapItem& it) {
        it.g0 = g0; it.g1 = g1; it.g2 = g2; it.node = u;
        if (use_h) {
            it.f0 = g0 + h0[u]; it.f1 = g1 + h1[u]; it.f2 = g2 + h2[u];
        } else {
            it.f0 = g0; it.f1 = g1; it.f2 = g2;
        }
    };

    while (!open.empty()) {
        HeapItem cur = open.top(); open.pop();
        int v = cur.node;
        int64_t g0 = cur.g0, g1 = cur.g1, g2 = cur.g2;
        int64_t f0 = cur.f0, f1 = cur.f1, f2 = cur.f2;

        // Lazy confirm vs permanent truncated fronts
        if (Gtr[v].dominated(g1, g2)) {
            ++lazy_reject;
            continue;
        }
        if (Gtr_goal.dominated(f1, f2)) {
            ++goal_prune;
            continue;
        }
        // Full 3D goal prune for seeds that may not align with DR edge cases
        if (sol_dom(f0, f1, f2)) {
            ++goal_prune;
            continue;
        }

        // Permanent: insert truncated g
        size_t before = Gtr[v].size();
        Gtr[v].insert(g1, g2);
        perm_total += (int64_t)Gtr[v].size() - (int64_t)before;
        if ((int64_t)Gtr[v].size() > max_node) max_node = (int64_t)Gtr[v].size();
        if (perm_total > peak_perm) peak_perm = perm_total;
        {
            int64_t live = (int64_t)open.size() + perm_total;
            if (live > peak_live) peak_live = live;
        }

        ++expanded;
        if (max_expanded > 0 && expanded >= max_expanded) {
            // BENCH_LIMIT
            goto finish_bench;
        }
        if ((expanded % 1000000) == 0) {
            std::fprintf(stderr,
                "[drlazy] exp=%lld gen=%lld open=%zu perm=%lld max_node=%lld "
                "lazy_rej=%lld\n",
                (long long)expanded, (long long)gen, open.size(),
                (long long)perm_total, (long long)max_node,
                (long long)lazy_reject);
            std::fflush(stderr);
        }

        if (v == target) {
            sol_update(g0, g1, g2);
            continue;
        }

        for (int64_t e = offset[v]; e < offset[v + 1]; ++e) {
            if (edge_allowed && edge_allowed[e] == 0) continue;
            int u = to[e];
            int64_t ng0 = g0 + w0[e], ng1 = g1 + w1[e], ng2 = g2 + w2[e];
            HeapItem child;
            f_of(u, ng0, ng1, ng2, child);

            // Generation: only cheap goal prune (3D sols + DR Gtr_goal).
            // No Gop / no Gtr[u] merge (lazy).
            if (Gtr_goal.dominated(child.f1, child.f2) || sol_dom(child.f0, child.f1, child.f2)) {
                ++gen_goal_prune;
                continue;
            }
            if (!push_open(child)) {
                // LABEL_LIMIT on generated
                stats_out[0] = (int64_t)sols.size();
                stats_out[1] = gen; stats_out[2] = expanded; stats_out[3] = peak_live;
                stats_out[4] = lazy_reject; stats_out[5] = goal_prune;
                stats_out[6] = gen_goal_prune;
                stats_out[7] = max_node; stats_out[8] = peak_perm;
                stats_out[9] = lazy_reject; stats_out[10] = peak_open;
                stats_out[11] = 0;
                stats_out[15] = (int64_t)open.size() + perm_total;
                return -1;
            }
        }
    }

    // EXACT
    {
        sort(sols.begin(), sols.end(), [](const Sol3& a, const Sol3& b) {
            if (a.c0 != b.c0) return a.c0 < b.c0;
            if (a.c1 != b.c1) return a.c1 < b.c1;
            return a.c2 < b.c2;
        });
        int nsol = (int)sols.size();
        if (nsol > max_out) nsol = max_out;
        for (int i = 0; i < nsol; ++i) {
            out_c1[i] = sols[i].c0;
            out_c2[i] = sols[i].c1;
            out_c3[i] = sols[i].c2;
        }
        stats_out[0] = (int64_t)sols.size();
        stats_out[1] = gen; stats_out[2] = expanded; stats_out[3] = peak_live;
        stats_out[4] = lazy_reject; // node-side lazy reject
        stats_out[5] = goal_prune;
        stats_out[6] = gen_goal_prune;
        stats_out[7] = max_node;
        stats_out[8] = peak_perm;
        stats_out[9] = lazy_reject;
        stats_out[10] = peak_open;
        stats_out[11] = 1; // EXACT
        stats_out[12] = 0;
        stats_out[13] = 0;
        stats_out[14] = 0;
        stats_out[15] = (int64_t)open.size() + perm_total;
        return nsol;
    }

finish_bench:
    {
        sort(sols.begin(), sols.end(), [](const Sol3& a, const Sol3& b) {
            if (a.c0 != b.c0) return a.c0 < b.c0;
            if (a.c1 != b.c1) return a.c1 < b.c1;
            return a.c2 < b.c2;
        });
        int nsol = (int)sols.size();
        if (nsol > max_out) nsol = max_out;
        for (int i = 0; i < nsol; ++i) {
            out_c1[i] = sols[i].c0;
            out_c2[i] = sols[i].c1;
            out_c3[i] = sols[i].c2;
        }
        stats_out[0] = (int64_t)sols.size();
        stats_out[1] = gen; stats_out[2] = expanded; stats_out[3] = peak_live;
        stats_out[4] = lazy_reject;
        stats_out[5] = goal_prune;
        stats_out[6] = gen_goal_prune;
        stats_out[7] = max_node;
        stats_out[8] = peak_perm;
        stats_out[9] = lazy_reject;
        stats_out[10] = peak_open;
        stats_out[11] = 2; // BENCH_LIMIT
        stats_out[15] = (int64_t)open.size() + perm_total;
        return nsol;
    }
}

} // extern "C"
