# -*- coding: utf-8 -*-
"""
Trace replay for CachedBlockSkyline: pick B and switch threshold.
Generates synthetic traces in size tiers + optional .bin traces from profiler.
"""
from __future__ import annotations

import csv
import random
import struct
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skyline_blocked import CachedBlockSkyline, NaiveSkyline, Cost3
from config import ANALYSIS_DIR


def load_bin_trace(path: Path):
    data = path.read_bytes()
    ntrip = struct.unpack_from("<q", data, 0)[0]
    vals = struct.unpack_from(f"<{3 * ntrip}q", data, 8)
    return [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals), 3)]


def synth_trace(target_live: int, seed: int, max_ops: int = 200000) -> list:
    """Grow a skyline toward target_live; return all insert attempts (g triples)."""
    random.seed(seed)
    naive = NaiveSkyline()
    ops = []
    guard = 0
    while naive.stats.inserts - naive.stats.rejects < target_live and guard < max_ops:
        guard += 1
        # bias toward incomparable points to widen front
        if random.random() < 0.5:
            # spread on simplex-ish
            x = random.randint(0, 10000)
            y = random.randint(0, 10000)
            z = random.randint(0, 10000)
        else:
            x = random.randint(0, 5000) + guard
            y = random.randint(0, 5000)
            z = 15000 - (x % 10000) // 2 + random.randint(0, 500)
            if z < 0:
                z = random.randint(0, 10000)
        p = (x, y, z)
        ops.append(p)
        naive.try_add(p)
        if len(naive.frontier()) >= target_live and guard > target_live:
            # keep going a bit to create deletes
            if guard > target_live * 3:
                break
    return ops


def replay(ops, B: int, switch_n: int):
    naive = NaiveSkyline()
    blocked = CachedBlockSkyline(block_size=B, switch_n=switch_n)
    t0 = time.perf_counter()
    for p in ops:
        a1, k1 = naive.try_add(p)
        a2, k2 = blocked.try_add(p)
        if a1 != a2 or sorted(k1) != sorted(k2):
            return {"mismatch": 1, "wall_sec": 0}
        if sorted(naive.frontier()) != sorted(blocked.frontier()):
            return {"mismatch": 1, "wall_sec": 0}
    # also time blocked-only
    t_both = time.perf_counter() - t0

    b2 = CachedBlockSkyline(block_size=B, switch_n=switch_n)
    t1 = time.perf_counter()
    for p in ops:
        b2.try_add(p)
    t_b = time.perf_counter() - t1

    n2 = NaiveSkyline()
    t2 = time.perf_counter()
    for p in ops:
        n2.try_add(p)
    t_n = time.perf_counter() - t2

    return {
        "mismatch": 0,
        "ops": len(ops),
        "final_live": len(n2.frontier()),
        "wall_sec_both_check": round(t_both, 4),
        "wall_naive": round(t_n, 4),
        "wall_blocked": round(t_b, 4),
        "labels_scanned_naive": n2.stats.labels_scanned,
        "labels_scanned_blocked": b2.stats.labels_scanned,
        "blocks_skipped": b2.stats.blocks_skipped,
        "comparisons_blocked": b2.stats.comparisons,
        "compactions": b2.stats.compactions,
        "scan_ratio": round(b2.stats.labels_scanned / max(n2.stats.labels_scanned, 1), 4),
        "speedup": round(t_n / max(t_b, 1e-9), 3),
    }


def main():
    out_dir = ANALYSIS_DIR / "problem2" / "skyline_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    tiers = [
        ("medium_256_512", 400),
        ("wide_512_2048", 1200),
        ("very_wide_2048p", 3500),
    ]
    Bs = [32, 64, 128]
    switches = [64, 128, 256, 512]

    for tier, target in tiers:
        ops = synth_trace(target, seed=hash(tier) % 100000)
        print(f"[trace] {tier} ops={len(ops)} final~{target}", flush=True)
        # linear baseline row
        n = NaiveSkyline()
        t0 = time.perf_counter()
        for p in ops:
            n.try_add(p)
        t_n = time.perf_counter() - t0
        base_scan = n.stats.labels_scanned
        for B in Bs:
            for sw in switches:
                if sw < B:
                    continue
                r = replay(ops, B, sw)
                r.update({"tier": tier, "B": B, "switch": sw, "wall_naive_ref": round(t_n, 4),
                          "labels_scanned_naive_ref": base_scan})
                rows.append(r)
                print(
                    f"  B={B} sw={sw} mismatch={r['mismatch']} "
                    f"scan_ratio={r.get('scan_ratio')} speedup={r.get('speedup')} "
                    f"wall_b={r.get('wall_blocked')}",
                    flush=True,
                )

    # real profiler traces if any
    prof = ANALYSIS_DIR / "problem2" / "profiler_B_3M"
    for tp in sorted(prof.glob("trace_node_*.bin"))[:10]:
        ops = load_bin_trace(tp)
        tier = f"real_{tp.stem}"
        print(f"[trace] {tier} ops={len(ops)}", flush=True)
        for B in Bs:
            for sw in [128, 256, 512]:
                r = replay(ops, B, sw)
                r.update({"tier": tier, "B": B, "switch": sw})
                rows.append(r)

    out = out_dir / "replay_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    # pick best: among mismatch=0, maximize speedup on very_wide, require scan_ratio<0.6
    cands = [r for r in rows if r.get("mismatch") == 0 and r.get("tier") == "very_wide_2048p"]
    cands.sort(key=lambda r: (-r.get("speedup", 0), r.get("scan_ratio", 9)))
    best = cands[0] if cands else None
    md = ["# Skyline trace replay", "", f"wrote `{out}`", ""]
    if best:
        md += [
            f"**Suggested:** B={best['B']} switch={best['switch']} "
            f"(very_wide speedup={best['speedup']}, scan_ratio={best['scan_ratio']})",
            "",
        ]
    md += ["## very_wide top configs", ""]
    for r in cands[:8]:
        md.append(
            f"- B={r['B']} sw={r['switch']}: speedup={r['speedup']} "
            f"scan_ratio={r['scan_ratio']} wall_b={r['wall_blocked']} skip={r['blocks_skipped']}"
        )
    (out_dir / "replay_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[wrote] {out}", flush=True)
    if best:
        print(f"[suggest] B={best['B']} switch={best['switch']}", flush=True)


if __name__ == "__main__":
    main()
