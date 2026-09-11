# -*- coding: utf-8 -*-
"""Numba-accelerated exact 3-objective NAMOA* kernel."""
from __future__ import annotations

import numpy as np
from numba import njit

MAX_SOLUTIONS = 200_000


@njit(cache=True)
def _heap_push(hf0, hf1, hf2, hg0, hg1, hg2, hid, size, f0, f1, f2, g0, g1, g2, lid):
    i = size
    size += 1
    while i > 0:
        p = (i - 1) // 2
        # if parent better or equal than new, stop
        if hf0[p] < f0 or (
            hf0[p] == f0
            and (
                hf1[p] < f1
                or (
                    hf1[p] == f1
                    and (
                        hf2[p] < f2
                        or (
                            hf2[p] == f2
                            and (
                                hg0[p] < g0
                                or (
                                    hg0[p] == g0
                                    and (
                                        hg1[p] < g1
                                        or (
                                            hg1[p] == g1
                                            and (hg2[p] < g2 or (hg2[p] == g2 and hid[p] <= lid))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ):
            break
        hf0[i], hf1[i], hf2[i] = hf0[p], hf1[p], hf2[p]
        hg0[i], hg1[i], hg2[i], hid[i] = hg0[p], hg1[p], hg2[p], hid[p]
        i = p
    hf0[i], hf1[i], hf2[i] = f0, f1, f2
    hg0[i], hg1[i], hg2[i], hid[i] = g0, g1, g2, lid
    return size


@njit(cache=True)
def _heap_pop(hf0, hf1, hf2, hg0, hg1, hg2, hid, size):
    f0, f1, f2 = hf0[0], hf1[0], hf2[0]
    g0, g1, g2, lid = hg0[0], hg1[0], hg2[0], hid[0]
    size -= 1
    if size == 0:
        return f0, f1, f2, g0, g1, g2, lid, size
    lf0, lf1, lf2 = hf0[size], hf1[size], hf2[size]
    lg0, lg1, lg2, llid = hg0[size], hg1[size], hg2[size], hid[size]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= size:
            break
        r = l + 1
        best = l
        if r < size:
            # r < l ?
            if hf0[r] < hf0[l] or (
                hf0[r] == hf0[l]
                and (
                    hf1[r] < hf1[l]
                    or (
                        hf1[r] == hf1[l]
                        and (
                            hf2[r] < hf2[l]
                            or (
                                hf2[r] == hf2[l]
                                and (
                                    hg0[r] < hg0[l]
                                    or (
                                        hg0[r] == hg0[l]
                                        and (
                                            hg1[r] < hg1[l]
                                            or (
                                                hg1[r] == hg1[l]
                                                and (
                                                    hg2[r] < hg2[l]
                                                    or (hg2[r] == hg2[l] and hid[r] < hid[l])
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ):
                best = r
        # last <= best?
        if lf0 < hf0[best] or (
            lf0 == hf0[best]
            and (
                lf1 < hf1[best]
                or (
                    lf1 == hf1[best]
                    and (
                        lf2 < hf2[best]
                        or (
                            lf2 == hf2[best]
                            and (
                                lg0 < hg0[best]
                                or (
                                    lg0 == hg0[best]
                                    and (
                                        lg1 < hg1[best]
                                        or (
                                            lg1 == hg1[best]
                                            and (
                                                lg2 < hg2[best]
                                                or (lg2 == hg2[best] and llid <= hid[best])
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ):
            break
        hf0[i], hf1[i], hf2[i] = hf0[best], hf1[best], hf2[best]
        hg0[i], hg1[i], hg2[i], hid[i] = hg0[best], hg1[best], hg2[best], hid[best]
        i = best
    hf0[i], hf1[i], hf2[i] = lf0, lf1, lf2
    hg0[i], hg1[i], hg2[i], hid[i] = lg0, lg1, lg2, llid
    return f0, f1, f2, g0, g1, g2, lid, size


@njit(cache=True)
def _dom_le(a0, a1, a2, b0, b1, b2):
    return a0 <= b0 and a1 <= b1 and a2 <= b2


@njit(cache=True)
def _sol_dominated(s0, s1, s2, nsol, y0, y1, y2):
    for i in range(nsol):
        if _dom_le(s0[i], s1[i], s2[i], y0, y1, y2):
            return True
    return False


@njit(cache=True)
def _sol_update(s0, s1, s2, nsol, y0, y1, y2):
    for i in range(nsol):
        if _dom_le(s0[i], s1[i], s2[i], y0, y1, y2):
            return nsol
    w = 0
    for i in range(nsol):
        if not _dom_le(y0, y1, y2, s0[i], s1[i], s2[i]):
            s0[w] = s0[i]
            s1[w] = s1[i]
            s2[w] = s2[i]
            w += 1
    if w >= MAX_SOLUTIONS:
        return -1
    s0[w] = y0
    s1[w] = y1
    s2[w] = y2
    return w + 1


@njit(cache=True)
def _try_add_label(
    head,
    next_in,
    active,
    g0,
    g1,
    g2,
    node_of,
    nlab,
    live,
    v,
    a,
    b,
    c,
    max_labels,
):
    """
    Returns: (new_nlab, new_live, lid, flag)
    flag: 0=ok, 1=dominated/dup, 2=overflow
    also returns node_prune_inc, dup_inc, max_cnt
    """
    # dominated?
    p = head[v]
    while p != -1:
        if active[p] != 0 and _dom_le(g0[p], g1[p], g2[p], a, b, c):
            if g0[p] == a and g1[p] == b and g2[p] == c:
                return nlab, live, -1, 1, 0, 1, 0  # dup
            return nlab, live, -1, 1, 1, 0, 0  # node prune

    # remove labels dominated by new
    p = head[v]
    new_head = np.int32(-1)
    cnt = 0
    while p != -1:
        nxt = next_in[p]
        if active[p] != 0:
            if _dom_le(a, b, c, g0[p], g1[p], g2[p]) and (a != g0[p] or b != g1[p] or c != g2[p]):
                active[p] = 0
                live -= 1
            else:
                next_in[p] = new_head
                new_head = p
                cnt += 1
        p = nxt
    head[v] = new_head

    if nlab >= max_labels:
        return nlab, live, -1, 2, 0, 0, cnt

    lid = nlab
    nlab += 1
    g0[lid] = a
    g1[lid] = b
    g2[lid] = c
    node_of[lid] = v
    active[lid] = 1
    next_in[lid] = head[v]
    head[v] = lid
    live += 1
    cnt += 1
    return nlab, live, lid, 0, 0, 0, cnt


@njit(cache=True)
def namoa_kernel(
    offset,
    to,
    w0,
    w1,
    w2,
    h0,
    h1,
    h2,
    source,
    target,
    use_h,
    seed0,
    seed1,
    seed2,
    n_seed,
    max_labels,
):
    n_nodes = offset.shape[0] - 1

    g0a = np.empty(max_labels, dtype=np.int64)
    g1a = np.empty(max_labels, dtype=np.int64)
    g2a = np.empty(max_labels, dtype=np.int64)
    node_of = np.empty(max_labels, dtype=np.int32)
    active = np.zeros(max_labels, dtype=np.uint8)
    next_in = np.full(max_labels, -1, dtype=np.int32)
    head = np.full(n_nodes, -1, dtype=np.int32)

    hf0 = np.empty(max_labels, dtype=np.int64)
    hf1 = np.empty(max_labels, dtype=np.int64)
    hf2 = np.empty(max_labels, dtype=np.int64)
    hg0 = np.empty(max_labels, dtype=np.int64)
    hg1 = np.empty(max_labels, dtype=np.int64)
    hg2 = np.empty(max_labels, dtype=np.int64)
    hid = np.empty(max_labels, dtype=np.int32)
    hsize = 0

    s0 = np.empty(MAX_SOLUTIONS, dtype=np.int64)
    s1 = np.empty(MAX_SOLUTIONS, dtype=np.int64)
    s2 = np.empty(MAX_SOLUTIONS, dtype=np.int64)
    nsol = 0
    for i in range(n_seed):
        nsol = _sol_update(s0, s1, s2, nsol, seed0[i], seed1[i], seed2[i])
        if nsol < 0:
            break

    nlab = 0
    expanded = 0
    gen = 0
    node_prune = 0
    goal_prune = 0
    dup_prune = 0
    peak_live = 0
    live = 0
    max_at_node = 0

    nlab, live, lid0, flag, np_inc, dup_inc, cnt = _try_add_label(
        head, next_in, active, g0a, g1a, g2a, node_of, nlab, live, source, 0, 0, 0, max_labels
    )
    if flag != 0:
        out = np.zeros((max(nsol, 0), 3), dtype=np.int64)
        for i in range(nsol):
            out[i, 0], out[i, 1], out[i, 2] = s0[i], s1[i], s2[i]
        stats = np.array(
            [nsol, gen, expanded, peak_live, node_prune, goal_prune, dup_prune, max_at_node, nlab],
            dtype=np.int64,
        )
        return out, stats

    gen += 1
    if cnt > max_at_node:
        max_at_node = cnt
    if live > peak_live:
        peak_live = live

    if use_h == 1:
        f0, f1, f2 = h0[source], h1[source], h2[source]
    else:
        f0, f1, f2 = np.int64(0), np.int64(0), np.int64(0)
    hsize = _heap_push(hf0, hf1, hf2, hg0, hg1, hg2, hid, hsize, f0, f1, f2, 0, 0, 0, lid0)

    while hsize > 0:
        _f0, _f1, _f2, a, b, c, lid, hsize = _heap_pop(
            hf0, hf1, hf2, hg0, hg1, hg2, hid, hsize
        )
        if active[lid] == 0:
            continue
        v = node_of[lid]
        expanded += 1

        if use_h == 1:
            ff0, ff1, ff2 = a + h0[v], b + h1[v], c + h2[v]
        else:
            ff0, ff1, ff2 = a, b, c
        if _sol_dominated(s0, s1, s2, nsol, ff0, ff1, ff2):
            goal_prune += 1
            continue

        if v == target:
            nsol = _sol_update(s0, s1, s2, nsol, a, b, c)
            continue

        for e in range(offset[v], offset[v + 1]):
            u = to[e]
            na = a + w0[e]
            nb = b + w1[e]
            nc = c + w2[e]

            if u == target:
                if _sol_dominated(s0, s1, s2, nsol, na, nb, nc):
                    goal_prune += 1
                    continue
                nsol = _sol_update(s0, s1, s2, nsol, na, nb, nc)
                nlab, live, _lid, flag, np_inc, dup_inc, cnt = _try_add_label(
                    head, next_in, active, g0a, g1a, g2a, node_of, nlab, live, u, na, nb, nc, max_labels
                )
                node_prune += np_inc
                dup_prune += dup_inc
                if flag == 0:
                    gen += 1
                    if cnt > max_at_node:
                        max_at_node = cnt
                    if live > peak_live:
                        peak_live = live
                continue

            # quick dominate check
            p = head[u]
            dominated = False
            while p != -1:
                if active[p] != 0 and _dom_le(g0a[p], g1a[p], g2a[p], na, nb, nc):
                    dominated = True
                    if g0a[p] == na and g1a[p] == nb and g2a[p] == nc:
                        dup_prune += 1
                    else:
                        node_prune += 1
                    break
                p = next_in[p]
            if dominated:
                continue

            if use_h == 1:
                nf0, nf1, nf2 = na + h0[u], nb + h1[u], nc + h2[u]
            else:
                nf0, nf1, nf2 = na, nb, nc
            if _sol_dominated(s0, s1, s2, nsol, nf0, nf1, nf2):
                goal_prune += 1
                continue

            nlab, live, new_id, flag, np_inc, dup_inc, cnt = _try_add_label(
                head, next_in, active, g0a, g1a, g2a, node_of, nlab, live, u, na, nb, nc, max_labels
            )
            node_prune += np_inc
            dup_prune += dup_inc
            if flag == 1:
                continue
            if flag == 2:
                out = np.zeros((0, 3), dtype=np.int64)
                stats = np.array(
                    [-1, gen, expanded, peak_live, node_prune, goal_prune, dup_prune, max_at_node, nlab],
                    dtype=np.int64,
                )
                return out, stats
            gen += 1
            if cnt > max_at_node:
                max_at_node = cnt
            if live > peak_live:
                peak_live = live
            hsize = _heap_push(
                hf0, hf1, hf2, hg0, hg1, hg2, hid, hsize, nf0, nf1, nf2, na, nb, nc, new_id
            )

    out = np.empty((nsol, 3), dtype=np.int64)
    for i in range(nsol):
        out[i, 0], out[i, 1], out[i, 2] = s0[i], s1[i], s2[i]
    # insertion sort by (c1,c2,c3)
    for i in range(1, nsol):
        k0, k1, k2 = out[i, 0], out[i, 1], out[i, 2]
        j = i - 1
        while j >= 0 and (
            out[j, 0] > k0
            or (out[j, 0] == k0 and (out[j, 1] > k1 or (out[j, 1] == k1 and out[j, 2] > k2)))
        ):
            out[j + 1, 0] = out[j, 0]
            out[j + 1, 1] = out[j, 1]
            out[j + 1, 2] = out[j, 2]
            j -= 1
        out[j + 1, 0] = k0
        out[j + 1, 1] = k1
        out[j + 1, 2] = k2

    stats = np.array(
        [nsol, gen, expanded, peak_live, node_prune, goal_prune, dup_prune, max_at_node, nlab],
        dtype=np.int64,
    )
    return out, stats
