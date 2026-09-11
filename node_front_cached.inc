// Cached BlockMeta node skyline (exact). Included conceptually into namoa_core.cpp.
// Insert: O(1) meta update. Delete: lazy; stale min/max are conservative.

struct Block {
    vector<int> ids;
    int64_t first_g1 = 0, last_g1 = 0;
    int64_t min_g2 = 0, min_g3 = 0, max_g2 = 0, max_g3 = 0;
    uint32_t live_count = 0, dead_count = 0;

    void rebuild_exact(const vector<Label>& labels) {
        vector<int> kept;
        kept.reserve(ids.size());
        for (int id : ids) if (labels[id].active) kept.push_back(id);
        ids.swap(kept);
        dead_count = 0;
        live_count = (uint32_t)ids.size();
        if (ids.empty()) {
            first_g1 = last_g1 = min_g2 = min_g3 = max_g2 = max_g3 = 0;
            return;
        }
        // keep g1-sorted
        sort(ids.begin(), ids.end(), [&](int i, int j) {
            const auto &A = labels[i], &B = labels[j];
            if (A.g0 != B.g0) return A.g0 < B.g0;
            if (A.g1 != B.g1) return A.g1 < B.g1;
            return A.g2 < B.g2;
        });
        first_g1 = labels[ids.front()].g0;
        last_g1 = labels[ids.back()].g0;
        min_g2 = max_g2 = labels[ids[0]].g1;
        min_g3 = max_g3 = labels[ids[0]].g2;
        for (int id : ids) {
            const auto& L = labels[id];
            if (L.g1 < min_g2) min_g2 = L.g1;
            if (L.g1 > max_g2) max_g2 = L.g1;
            if (L.g2 < min_g3) min_g3 = L.g2;
            if (L.g2 > max_g3) max_g3 = L.g2;
        }
    }

    void note_insert(const Label& L) {
        ++live_count;
        if (live_count == 1 && dead_count == 0) {
            first_g1 = last_g1 = L.g0;
            min_g2 = max_g2 = L.g1;
            min_g3 = max_g3 = L.g2;
            return;
        }
        if (L.g0 < first_g1) first_g1 = L.g0;
        if (L.g0 > last_g1) last_g1 = L.g0;
        if (L.g1 < min_g2) min_g2 = L.g1;
        if (L.g1 > max_g2) max_g2 = L.g1;
        if (L.g2 < min_g3) min_g3 = L.g2;
        if (L.g2 > max_g3) max_g3 = L.g2;
    }
};

struct NodeFront {
    vector<int> small;
    vector<Block> blocks;
    int live_size() const {
        if (!blocks.empty()) {
            int s = 0;
            for (const auto& b : blocks) s += (int)b.live_count;
            return s;
        }
        return (int)small.size(); // small keeps only actives
    }
};

static int g_sky_B = 64;
static int g_sky_switch = 256;
static double g_sky_dead_ratio = 0.35;
static int64_t g_labels_scanned = 0;
static int64_t g_blocks_skipped = 0;

static void front_promote(NodeFront& F, const vector<Label>& labels) {
    if (!F.blocks.empty()) return;
    if ((int)F.small.size() < g_sky_switch) return;
    F.blocks.clear();
    for (int i = 0; i < (int)F.small.size(); i += g_sky_B) {
        Block bl;
        int R = min((int)F.small.size(), i + g_sky_B);
        bl.ids.assign(F.small.begin() + i, F.small.begin() + R);
        bl.rebuild_exact(labels);
        F.blocks.push_back(std::move(bl));
    }
    F.small.clear();
}

static void maybe_compact_block(NodeFront& F, int bi, const vector<Label>& labels, int64_t* t_compact_ns) {
    if (bi < 0 || bi >= (int)F.blocks.size()) return;
    Block& bl = F.blocks[bi];
    uint32_t tot = bl.live_count + bl.dead_count;
    if (tot == 0 || (double)bl.dead_count / tot <= g_sky_dead_ratio) return;
    auto t0 = Steady::now();
    bl.rebuild_exact(labels);
    if (bl.ids.size() > (size_t)(2 * g_sky_B)) {
        Block left, right;
        size_t mid = bl.ids.size() / 2;
        left.ids.assign(bl.ids.begin(), bl.ids.begin() + mid);
        right.ids.assign(bl.ids.begin() + mid, bl.ids.end());
        left.rebuild_exact(labels);
        right.rebuild_exact(labels);
        F.blocks[bi] = std::move(left);
        F.blocks.insert(F.blocks.begin() + bi + 1, std::move(right));
    } else if (bl.ids.empty()) {
        F.blocks.erase(F.blocks.begin() + bi);
    }
    if (t_compact_ns) *t_compact_ns += ns_since(t0);
}

static int try_add_front(
    vector<Label>& labels, NodeFront& F,
    int v, int64_t a, int64_t b, int64_t c, int max_labels,
    int64_t& live, int64_t& gen, int64_t& node_prune, int64_t& dup_prune,
    int64_t& max_at_node, int64_t& peak_live,
    NodeProf* np, int64_t* t_find, int64_t* t_del, int64_t* t_ins,
    vector<int32_t>* scan_samp, int scan_samp_cap,
    int64_t* t_compact) {

    if (np) ++np->insert_attempts;
    front_promote(F, labels);

    auto t0 = Steady::now();
    int scanned = 0;
    bool dominated = false, is_dup = false;

    if (F.blocks.empty()) {
        for (int id : F.small) {
            ++scanned;
            const auto& L = labels[id];
            if (L.g0 > a) break;
            if (dom_le(L.g0, L.g1, L.g2, a, b, c)) {
                dominated = true;
                is_dup = (L.g0 == a && L.g1 == b && L.g2 == c);
                break;
            }
        }
    } else {
        for (const auto& bl : F.blocks) {
            if (bl.live_count == 0) continue;
            if (bl.first_g1 > a) break;
            if (bl.min_g2 > b || bl.min_g3 > c) {
                ++g_blocks_skipped;
                continue;
            }
            for (int id : bl.ids) {
                if (!labels[id].active) continue;
                ++scanned;
                const auto& L = labels[id];
                if (L.g0 > a) break;
                if (dom_le(L.g0, L.g1, L.g2, a, b, c)) {
                    dominated = true;
                    is_dup = (L.g0 == a && L.g1 == b && L.g2 == c);
                    break;
                }
            }
            if (dominated) break;
        }
    }
    g_labels_scanned += scanned;
    if (t_find) *t_find += ns_since(t0);
    if (np) {
        np->scan_sum += scanned; np->scan_queries++;
        if (scanned > np->scan_max) np->scan_max = scanned;
        np->node_dom_ns += ns_since(t0);
    }
    if (scan_samp && (int)scan_samp->size() < scan_samp_cap) scan_samp->push_back(scanned);
    if (dominated) {
        if (np) np->dominated_reject++;
        if (is_dup) ++dup_prune; else ++node_prune;
        return -1;
    }

    auto t1 = Steady::now();
    int killed = 0;
    if (F.blocks.empty()) {
        vector<int> kept;
        kept.reserve(F.small.size() + 1);
        for (int id : F.small) {
            auto& L = labels[id];
            if (L.g0 >= a && dom_le(a, b, c, L.g0, L.g1, L.g2) &&
                !(L.g0 == a && L.g1 == b && L.g2 == c)) {
                L.active = 0; --live; ++killed;
            } else kept.push_back(id);
        }
        F.small.swap(kept);
    } else {
        vector<int> to_compact;
        for (int bi = 0; bi < (int)F.blocks.size(); ++bi) {
            Block& bl = F.blocks[bi];
            if (bl.live_count == 0) continue;
            if (bl.last_g1 < a) continue;
            if (bl.max_g2 < b || bl.max_g3 < c) {
                ++g_blocks_skipped;
                continue;
            }
            if (bl.first_g1 >= a && bl.min_g2 >= b && bl.min_g3 >= c) {
                for (int id : bl.ids) {
                    if (!labels[id].active) continue;
                    labels[id].active = 0; --live; ++killed;
                    --bl.live_count; ++bl.dead_count;
                }
                to_compact.push_back(bi);
                continue;
            }
            for (int id : bl.ids) {
                if (!labels[id].active) continue;
                auto& L = labels[id];
                if (L.g0 >= a && dom_le(a, b, c, L.g0, L.g1, L.g2) &&
                    !(L.g0 == a && L.g1 == b && L.g2 == c)) {
                    L.active = 0; --live; ++killed;
                    --bl.live_count; ++bl.dead_count;
                }
            }
            to_compact.push_back(bi);
        }
        sort(to_compact.rbegin(), to_compact.rend());
        to_compact.erase(unique(to_compact.begin(), to_compact.end()), to_compact.end());
        for (int bi : to_compact) maybe_compact_block(F, bi, labels, t_compact);
        // drop empty
        F.blocks.erase(remove_if(F.blocks.begin(), F.blocks.end(),
            [](const Block& b) { return b.live_count == 0; }), F.blocks.end());
    }
    if (t_del) *t_del += ns_since(t1);
    if (np) { np->old_killed += killed; np->node_dom_ns += ns_since(t1); }

    auto t2 = Steady::now();
    if ((int)labels.size() >= max_labels) return -2;
    int lid = (int)labels.size();
    labels.push_back(Label{a, b, c, (int32_t)v, 1});

    if (F.blocks.empty()) {
        int pos = (int)F.small.size();
        for (int i = 0; i < (int)F.small.size(); ++i)
            if (labels[F.small[i]].g0 > a) { pos = i; break; }
        F.small.insert(F.small.begin() + pos, lid);
        front_promote(F, labels);
    } else {
        // find block by last_g1
        int bi = 0;
        while (bi < (int)F.blocks.size() && F.blocks[bi].last_g1 < a) ++bi;
        if (bi >= (int)F.blocks.size()) {
            if (!F.blocks.empty() &&
                (int)(F.blocks.back().live_count + F.blocks.back().dead_count) < 2 * g_sky_B) {
                bi = (int)F.blocks.size() - 1;
            } else {
                Block nb;
                nb.ids.push_back(lid);
                nb.rebuild_exact(labels);
                F.blocks.push_back(std::move(nb));
                bi = -1;
            }
        }
        if (bi >= 0) {
            Block& bl = F.blocks[bi];
            bl.ids.push_back(lid);
            // keep sorted by g1
            int j = (int)bl.ids.size() - 1;
            while (j > 0 && labels[bl.ids[j - 1]].g0 > labels[bl.ids[j]].g0) {
                swap(bl.ids[j - 1], bl.ids[j]);
                --j;
            }
            bl.note_insert(labels[lid]);
            if ((int)bl.ids.size() > 2 * g_sky_B) {
                Block left, right;
                size_t mid = bl.ids.size() / 2;
                left.ids.assign(bl.ids.begin(), bl.ids.begin() + mid);
                right.ids.assign(bl.ids.begin() + mid, bl.ids.end());
                left.rebuild_exact(labels);
                right.rebuild_exact(labels);
                F.blocks[bi] = std::move(left);
                F.blocks.insert(F.blocks.begin() + bi + 1, std::move(right));
            }
        }
    }

    ++gen; ++live;
    int fsz = F.live_size();
    if ((int64_t)fsz > max_at_node) max_at_node = fsz;
    if (live > peak_live) peak_live = live;
    if (np) {
        np->accepted++;
        if (fsz > np->peak_front) np->peak_front = fsz;
        np->node_dom_ns += ns_since(t2);
    }
    if (t_ins) *t_ins += ns_since(t2);
    return lid;
}
