#!/usr/bin/env python3
"""
Aggregate CounterFact-style metrics + PPL stability metrics (Butterfly Effect proxy) from JSONL logs.

This script does NOT run editing. It only reads an existing results file and aggregates metrics.

CORE METRICS (ROME/CounterFact-style):
- ES / PS / NS and harmonic mean S
- Forward ES/PS: post success iff logP(target_new) > logP(ground_truth)
- Rollback ES/PS: post success iff logP(ground_truth) > logP(target_new) (as logged; GT/NEW not swapped)
- NS: success iff logP(ground_truth) > logP(target_new) on neighborhood prompts

PPL STABILITY METRICS (Butterfly Effect proxy):
- Uses per-record PPL on a fixed external text set (already computed in your logs):
    ppl.M0 = before forward
    ppl.M1 = after forward
    ppl.M2 = after rollback
- Filters non-finite PPL values (inf/nan) before aggregation.
- Reports both:
    * Unpaired aggregates (each mean on its own valid subset)
    * Paired aggregates (only records where M0,M1,M2 are all finite), recommended for BE-style evaluation

Collapse rate:
- Prefer be_report.M1/M2.is_collapse when present; else compute from thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional


# ----------------------------- math helpers ----------------------------- #

def harmonic_mean(values: List[float], eps: float = 1e-12) -> float:
    vals = [max(eps, v) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)


def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci_mean(xs: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """
    95% bootstrap CI for the mean of xs (works for 0/1 proportions too).
    Deterministic LCG RNG (no global state).
    """
    if not xs:
        return (float("nan"), float("nan"))

    m = 2**32
    a = 1664525
    c = 1013904223
    state = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF

    def rnd() -> int:
        nonlocal state
        state = (a * state + c) % m
        return state

    n = len(xs)
    means: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _j in range(n):
            idx = rnd() % n
            s += xs[idx]
        means.append(s / n)

    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{100.0 * x:6.2f}%"


# ----------------------------- IO helpers ----------------------------- #

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    Generator yielding dict records from:
    - JSONL (one JSON per line), or
    - JSON array file (fallback).
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array at top-level.")
            for x in data:
                if isinstance(x, dict):
                    yield x
            return

        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def as_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    try:
        return [float(x)]
    except Exception:
        return []


# ----------------------------- aggregation model ----------------------------- #

@dataclass
class AggBucket:
    es_post: List[float]
    ps_post: List[float]
    ns_post: List[float]

    em_post: List[float]
    pm_post: List[float]
    nm_post: List[float]

    forward_pre_gt_pref_rewrite: List[float]
    forward_pre_gt_pref_rephrase: List[float]

    n_cases_seen: int
    n_neigh_prompts_seen: int

    def __init__(self) -> None:
        self.es_post = []
        self.ps_post = []
        self.ns_post = []
        self.em_post = []
        self.pm_post = []
        self.nm_post = []
        self.forward_pre_gt_pref_rewrite = []
        self.forward_pre_gt_pref_rephrase = []
        self.n_cases_seen = 0
        self.n_neigh_prompts_seen = 0


@dataclass
class PPLBucket:
    # Unpaired raw PPLs per record (finite only)
    m0: List[float]
    m1: List[float]
    m2: List[float]

    # Unpaired deltas (finite only)
    f_abs: List[float]
    f_rel: List[float]
    r_abs: List[float]
    r_rel: List[float]
    rvf_abs: List[float]
    rvf_rel: List[float]

    # Paired views: records where M0, M1, M2 are all finite
    paired_m0: List[float]
    paired_m1: List[float]
    paired_m2: List[float]
    paired_f_abs: List[float]
    paired_f_rel: List[float]
    paired_r_abs: List[float]
    paired_r_rel: List[float]
    paired_rvf_abs: List[float]
    paired_rvf_rel: List[float]

    # Collapse flags
    collapse_m1: List[float]
    collapse_m2: List[float]

    # Thresholds observed (for reference)
    thr_rel: List[float]
    thr_abs: List[float]

    # Diagnostics
    dropped_m1_nonfinite: int
    dropped_m2_nonfinite: int
    invalid_rel_div0: int

    def __init__(self) -> None:
        self.m0, self.m1, self.m2 = [], [], []
        self.f_abs, self.f_rel = [], []
        self.r_abs, self.r_rel = [], []
        self.rvf_abs, self.rvf_rel = [], []

        self.paired_m0, self.paired_m1, self.paired_m2 = [], [], []
        self.paired_f_abs, self.paired_f_rel = [], []
        self.paired_r_abs, self.paired_r_rel = [], []
        self.paired_rvf_abs, self.paired_rvf_rel = [], []

        self.collapse_m1, self.collapse_m2 = [], []
        self.thr_rel, self.thr_abs = [], []

        self.dropped_m1_nonfinite = 0
        self.dropped_m2_nonfinite = 0
        self.invalid_rel_div0 = 0


# ----------------------------- extraction: metrics_core_per_case ----------------------------- #

def extract_from_metrics_core_per_case(rec: Dict[str, Any], mode: str, bucket: AggBucket) -> int:
    mcpp = rec.get("metrics_core_per_case")
    if not isinstance(mcpp, dict):
        return 0

    cases = mcpp.get(mode)
    if not isinstance(cases, list):
        return 0

    used = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        pre = case.get("pre", {})
        post = case.get("post", {})

        rw_post = as_float_list(post.get("rewrite_success"))
        rp_post = as_float_list(post.get("paraphrase_success"))
        nb_post = as_float_list(post.get("neighborhood_preservation"))

        if rw_post:
            bucket.es_post.append(float(rw_post[0]))
        if rp_post:
            bucket.ps_post.append(float(rp_post[0]))
        if nb_post:
            bucket.ns_post.extend([float(x) for x in nb_post])
            bucket.n_neigh_prompts_seen += len(nb_post)

        if mode == "forward":
            rw_pre = as_float_list(pre.get("rewrite_success"))
            rp_pre = as_float_list(pre.get("paraphrase_success"))
            if rw_pre:
                bucket.forward_pre_gt_pref_rewrite.append(float(rw_pre[0]))
            if rp_pre:
                bucket.forward_pre_gt_pref_rephrase.append(float(rp_pre[0]))

        bucket.n_cases_seen += 1
        used += 1

    return used


# ----------------------------- extraction: fallback from metrics_per_case ----------------------------- #

def extract_from_metrics_per_case(rec: Dict[str, Any], mode: str, bucket: AggBucket) -> int:
    mpc = rec.get("metrics_per_case")
    if not isinstance(mpc, dict):
        return 0

    cases = mpc.get(mode)
    if not isinstance(cases, list):
        return 0

    used = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        pre = case.get("pre", {})
        post = case.get("post", {})

        if mode == "forward":
            pre_gt = as_float_list(pre.get("rewrite_ground_truth_logprob"))
            pre_new = as_float_list(pre.get("rewrite_target_new_logprob"))
            if pre_gt and pre_new:
                bucket.forward_pre_gt_pref_rewrite.append(1.0 if pre_gt[0] > pre_new[0] else 0.0)

            pre_r_gt = as_float_list(pre.get("rephrase_ground_truth_logprob"))
            pre_r_new = as_float_list(pre.get("rephrase_target_new_logprob"))
            if pre_r_gt and pre_r_new:
                bucket.forward_pre_gt_pref_rephrase.append(1.0 if pre_r_gt[0] > pre_r_new[0] else 0.0)

        post_gt = as_float_list(post.get("rewrite_ground_truth_logprob"))
        post_new = as_float_list(post.get("rewrite_target_new_logprob"))
        if post_gt and post_new:
            delta = (post_new[0] - post_gt[0]) if mode == "forward" else (post_gt[0] - post_new[0])
            bucket.em_post.append(delta)
            bucket.es_post.append(1.0 if delta > 0 else 0.0)

        post_r_gt = as_float_list(post.get("rephrase_ground_truth_logprob"))
        post_r_new = as_float_list(post.get("rephrase_target_new_logprob"))
        if post_r_gt and post_r_new:
            delta = (post_r_new[0] - post_r_gt[0]) if mode == "forward" else (post_r_gt[0] - post_r_new[0])
            bucket.pm_post.append(delta)
            bucket.ps_post.append(1.0 if delta > 0 else 0.0)

        post_loc = post.get("locality", {})
        if isinstance(post_loc, dict):
            n_gt = as_float_list(post_loc.get("neighborhood_ground_truth_logprob"))
            n_new = as_float_list(post_loc.get("neighborhood_target_new_logprob"))
            if n_gt and n_new and len(n_gt) == len(n_new):
                for g, n in zip(n_gt, n_new):
                    d = g - n
                    bucket.nm_post.append(d)
                    bucket.ns_post.append(1.0 if d > 0 else 0.0)
                bucket.n_neigh_prompts_seen += len(n_gt)

        bucket.n_cases_seen += 1
        used += 1

    return used


# ----------------------------- extraction: PPL (Butterfly Effect proxy) ----------------------------- #

def _as_float_finite(x: Any) -> Optional[float]:
    """Convert to float and filter NaN/Inf. Returns None if conversion fails or value is not finite."""
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _decide_collapse(
    ppl_before: Optional[float],
    ppl_after: Optional[float],
    delta_abs: Optional[float],
    delta_rel: Optional[float],
    thr_abs: Optional[float],
    thr_rel: Optional[float],
) -> Optional[bool]:
    if delta_rel is None and (ppl_before is not None and ppl_after is not None and ppl_before > 0):
        delta_rel = ppl_after / ppl_before
    if delta_abs is None and (ppl_before is not None and ppl_after is not None):
        delta_abs = ppl_after - ppl_before

    if thr_rel is not None and delta_rel is not None:
        return delta_rel > thr_rel
    if thr_abs is not None and delta_abs is not None:
        return delta_abs > thr_abs
    return None


def extract_ppl(rec: Dict[str, Any], bucket: PPLBucket) -> bool:
    ppl = rec.get("ppl")
    if not isinstance(ppl, dict):
        return False

    m0_raw = ppl.get("M0")
    m1_raw = ppl.get("M1")
    m2_raw = ppl.get("M2")

    m0 = _as_float_finite(m0_raw)
    m1 = _as_float_finite(m1_raw)
    m2 = _as_float_finite(m2_raw)

    if m0 is None:
        return False

    if m1 is None and m1_raw is not None:
        bucket.dropped_m1_nonfinite += 1
    if m2 is None and m2_raw is not None:
        bucket.dropped_m2_nonfinite += 1

    # Unpaired aggregates
    bucket.m0.append(m0)

    if m1 is not None:
        bucket.m1.append(m1)
        bucket.f_abs.append(m1 - m0)
        if m0 > 0:
            bucket.f_rel.append(m1 / m0)
        else:
            bucket.invalid_rel_div0 += 1

    if m2 is not None:
        bucket.m2.append(m2)
        bucket.r_abs.append(m2 - m0)
        if m0 > 0:
            bucket.r_rel.append(m2 / m0)
        else:
            bucket.invalid_rel_div0 += 1

    if m1 is not None and m2 is not None:
        bucket.rvf_abs.append(m2 - m1)
        if m1 > 0:
            bucket.rvf_rel.append(m2 / m1)
        else:
            bucket.invalid_rel_div0 += 1

    # Paired aggregates (recommended): require all three finite
    if m1 is not None and m2 is not None:
        bucket.paired_m0.append(m0)
        bucket.paired_m1.append(m1)
        bucket.paired_m2.append(m2)

        bucket.paired_f_abs.append(m1 - m0)
        bucket.paired_r_abs.append(m2 - m0)
        bucket.paired_rvf_abs.append(m2 - m1)

        if m0 > 0:
            bucket.paired_f_rel.append(m1 / m0)
            bucket.paired_r_rel.append(m2 / m0)
        else:
            bucket.invalid_rel_div0 += 1

        if m1 > 0:
            bucket.paired_rvf_rel.append(m2 / m1)
        else:
            bucket.invalid_rel_div0 += 1

    # Collapse info: prefer be_report flags if present; else compute from thresholds.
    be = rec.get("be_report")
    if isinstance(be, dict):
        m1r = be.get("M1") if isinstance(be.get("M1"), dict) else None
        m2r = be.get("M2") if isinstance(be.get("M2"), dict) else None

        if m1r is not None:
            tr = _as_float_finite(m1r.get("collapse_rel_threshold"))
            ta = _as_float_finite(m1r.get("collapse_abs_threshold"))
            if tr is not None:
                bucket.thr_rel.append(tr)
            if ta is not None:
                bucket.thr_abs.append(ta)

        if m1r is not None and m1 is not None:
            flag = m1r.get("is_collapse")
            if isinstance(flag, bool):
                bucket.collapse_m1.append(1.0 if flag else 0.0)
            else:
                computed = _decide_collapse(
                    ppl_before=_as_float_finite(m1r.get("ppl_before")),
                    ppl_after=_as_float_finite(m1r.get("ppl_after")),
                    delta_abs=_as_float_finite(m1r.get("ppl_delta_abs")),
                    delta_rel=_as_float_finite(m1r.get("ppl_delta_rel")),
                    thr_abs=_as_float_finite(m1r.get("collapse_abs_threshold")),
                    thr_rel=_as_float_finite(m1r.get("collapse_rel_threshold")),
                )
                if computed is not None:
                    bucket.collapse_m1.append(1.0 if computed else 0.0)

        if m2r is not None and m2 is not None:
            flag = m2r.get("is_collapse")
            if isinstance(flag, bool):
                bucket.collapse_m2.append(1.0 if flag else 0.0)
            else:
                computed = _decide_collapse(
                    ppl_before=_as_float_finite(m2r.get("ppl_before")),
                    ppl_after=_as_float_finite(m2r.get("ppl_after")),
                    delta_abs=_as_float_finite(m2r.get("ppl_delta_abs")),
                    delta_rel=_as_float_finite(m2r.get("ppl_delta_rel")),
                    thr_abs=_as_float_finite(m2r.get("collapse_abs_threshold")),
                    thr_rel=_as_float_finite(m2r.get("collapse_rel_threshold")),
                )
                if computed is not None:
                    bucket.collapse_m2.append(1.0 if computed else 0.0)

    return True


# ----------------------------- summarization ----------------------------- #

def summarize_metrics(bucket: AggBucket, mode: str, n_boot: int, seed: int) -> Dict[str, Any]:
    ES = safe_mean(bucket.es_post)
    PS = safe_mean(bucket.ps_post)
    NS = safe_mean(bucket.ns_post)
    S = harmonic_mean([ES, PS, NS]) if all(not math.isnan(x) for x in [ES, PS, NS]) else float("nan")

    ES_ci = bootstrap_ci_mean(bucket.es_post, n_boot=n_boot, seed=seed ^ 0x1111)
    PS_ci = bootstrap_ci_mean(bucket.ps_post, n_boot=n_boot, seed=seed ^ 0x2222)
    NS_ci = bootstrap_ci_mean(bucket.ns_post, n_boot=n_boot, seed=seed ^ 0x3333)

    EM = safe_mean(bucket.em_post)
    PM = safe_mean(bucket.pm_post)
    NM = safe_mean(bucket.nm_post)

    out = {
        "mode": mode,
        "counts": {
            "cases_seen": bucket.n_cases_seen,
            "rewrite_posts": len(bucket.es_post),
            "rephrase_posts": len(bucket.ps_post),
            "neighborhood_prompts": len(bucket.ns_post),
        },
        "forward_pre_sanity": None,
        "post_metrics": {
            "ES": ES, "ES_ci95": ES_ci, "EM": EM,
            "PS": PS, "PS_ci95": PS_ci, "PM": PM,
            "NS": NS, "NS_ci95": NS_ci, "NM": NM,
            "S": S,
        },
    }

    if mode == "forward":
        out["forward_pre_sanity"] = {
            "pre_rewrite_gt_preferred_rate": safe_mean(bucket.forward_pre_gt_pref_rewrite),
            "pre_rephrase_gt_preferred_rate": safe_mean(bucket.forward_pre_gt_pref_rephrase),
        }

    return out


def summarize_ppl(bucket: PPLBucket, n_boot: int, seed: int) -> Dict[str, Any]:
    def ci(xs: List[float], s: int) -> Tuple[float, float]:
        return bootstrap_ci_mean(xs, n_boot=n_boot, seed=s) if xs else (float("nan"), float("nan"))

    unpaired = {
        "counts": {
            "records_with_M0": len(bucket.m0),
            "records_with_M1_finite": len(bucket.m1),
            "records_with_M2_finite": len(bucket.m2),
            "dropped_m1_nonfinite": bucket.dropped_m1_nonfinite,
            "dropped_m2_nonfinite": bucket.dropped_m2_nonfinite,
            "invalid_rel_div0": bucket.invalid_rel_div0,
            "valid_rate_M1": (len(bucket.m1) / len(bucket.m0)) if bucket.m0 else float("nan"),
            "valid_rate_M2": (len(bucket.m2) / len(bucket.m0)) if bucket.m0 else float("nan"),
        },
        "ppl": {
            "M0_mean": safe_mean(bucket.m0),
            "M0_ci95": ci(bucket.m0, seed ^ 0x9001),
            "M1_mean": safe_mean(bucket.m1),
            "M1_ci95": ci(bucket.m1, seed ^ 0x9002),
            "M2_mean": safe_mean(bucket.m2),
            "M2_ci95": ci(bucket.m2, seed ^ 0x9003),
        },
        "deltas": {
            "forward_abs_mean": safe_mean(bucket.f_abs),
            "forward_abs_ci95": ci(bucket.f_abs, seed ^ 0x9011),
            "forward_rel_mean": safe_mean(bucket.f_rel),
            "forward_rel_ci95": ci(bucket.f_rel, seed ^ 0x9012),

            "rollback_abs_mean": safe_mean(bucket.r_abs),
            "rollback_abs_ci95": ci(bucket.r_abs, seed ^ 0x9021),
            "rollback_rel_mean": safe_mean(bucket.r_rel),
            "rollback_rel_ci95": ci(bucket.r_rel, seed ^ 0x9022),

            "rollback_vs_forward_abs_mean": safe_mean(bucket.rvf_abs),
            "rollback_vs_forward_abs_ci95": ci(bucket.rvf_abs, seed ^ 0x9031),
            "rollback_vs_forward_rel_mean": safe_mean(bucket.rvf_rel),
            "rollback_vs_forward_rel_ci95": ci(bucket.rvf_rel, seed ^ 0x9032),
        },
    }

    paired = {
        "counts": {
            "records_with_M0_M1_M2_finite": len(bucket.paired_m0),
            "paired_rate": (len(bucket.paired_m0) / len(bucket.m0)) if bucket.m0 else float("nan"),
        },
        "ppl": {
            "M0_mean": safe_mean(bucket.paired_m0),
            "M0_ci95": ci(bucket.paired_m0, seed ^ 0xA001),
            "M1_mean": safe_mean(bucket.paired_m1),
            "M1_ci95": ci(bucket.paired_m1, seed ^ 0xA002),
            "M2_mean": safe_mean(bucket.paired_m2),
            "M2_ci95": ci(bucket.paired_m2, seed ^ 0xA003),
        },
        "deltas": {
            "forward_abs_mean": safe_mean(bucket.paired_f_abs),
            "forward_abs_ci95": ci(bucket.paired_f_abs, seed ^ 0xA011),
            "forward_rel_mean": safe_mean(bucket.paired_f_rel),
            "forward_rel_ci95": ci(bucket.paired_f_rel, seed ^ 0xA012),

            "rollback_abs_mean": safe_mean(bucket.paired_r_abs),
            "rollback_abs_ci95": ci(bucket.paired_r_abs, seed ^ 0xA021),
            "rollback_rel_mean": safe_mean(bucket.paired_r_rel),
            "rollback_rel_ci95": ci(bucket.paired_r_rel, seed ^ 0xA022),

            "rollback_vs_forward_abs_mean": safe_mean(bucket.paired_rvf_abs),
            "rollback_vs_forward_abs_ci95": ci(bucket.paired_rvf_abs, seed ^ 0xA031),
            "rollback_vs_forward_rel_mean": safe_mean(bucket.paired_rvf_rel),
            "rollback_vs_forward_rel_ci95": ci(bucket.paired_rvf_rel, seed ^ 0xA032),
        },
    }

    collapse = {
        "collapse_rate_M1": safe_mean(bucket.collapse_m1),
        "collapse_rate_M2": safe_mean(bucket.collapse_m2),
        "collapse_count_M1": int(sum(bucket.collapse_m1)) if bucket.collapse_m1 else 0,
        "collapse_count_M2": int(sum(bucket.collapse_m2)) if bucket.collapse_m2 else 0,
        "thresholds_rel_observed": {
            "min": min(bucket.thr_rel) if bucket.thr_rel else None,
            "max": max(bucket.thr_rel) if bucket.thr_rel else None,
        },
        "thresholds_abs_observed": {
            "min": min(bucket.thr_abs) if bucket.thr_abs else None,
            "max": max(bucket.thr_abs) if bucket.thr_abs else None,
        },
    }

    return {
        "unpaired": unpaired,
        "paired": paired,
        "collapse": collapse,
    }


# ----------------------------- CLI main ----------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate ROME/CounterFact metrics + PPL stability from JSONL logs.")
    ap.add_argument("path", help="Path to JSONL (or JSON array) results file.")
    ap.add_argument("--out", default="", help="Optional path to write aggregated JSON summary.")
    ap.add_argument("--mode", default="both", choices=["forward", "rollback", "both"], help="Which mode(s) to aggregate.")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap samples for CI.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for bootstrap.")
    ap.add_argument(
        "--prefer-core",
        action="store_true",
        help="Prefer metrics_core_per_case when present. Falls back to logprob recomputation otherwise.",
    )
    ap.add_argument(
        "--ppl",
        action="store_true",
        help="Also aggregate PPL (M0/M1/M2), deltas and collapse rates from ppl/be_report fields.",
    )
    args = ap.parse_args()

    path = os.path.expanduser(args.path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    modes = ["forward", "rollback"] if args.mode == "both" else [args.mode]
    buckets: Dict[str, AggBucket] = {m: AggBucket() for m in modes}
    ppl_bucket = PPLBucket()

    records_read = 0
    used_core = {m: 0 for m in modes}
    used_fallback = {m: 0 for m in modes}
    ppl_records_used = 0

    for rec in read_jsonl(path):
        records_read += 1

        if args.ppl:
            if extract_ppl(rec, ppl_bucket):
                ppl_records_used += 1

        for m in modes:
            used = 0
            if args.prefer_core:
                used = extract_from_metrics_core_per_case(rec, m, buckets[m])
                if used:
                    used_core[m] += used
            if used == 0:
                used = extract_from_metrics_per_case(rec, m, buckets[m])
                if used:
                    used_fallback[m] += used

    summaries = [summarize_metrics(buckets[m], m, n_boot=args.bootstrap, seed=args.seed) for m in modes]
    ppl_summary = summarize_ppl(ppl_bucket, n_boot=args.bootstrap, seed=args.seed) if args.ppl else None

    print(f"Read records: {records_read}")

    for s in summaries:
        pm = s["post_metrics"]
        cnt = s["counts"]
        print("\n" + "=" * 80)
        print(f"Mode: {s['mode']}")
        print(f"Cases seen: {cnt['cases_seen']}")
        print(f"Rewrite posts: {cnt['rewrite_posts']}")
        print(f"Rephrase posts: {cnt['rephrase_posts']}")
        print(f"Neighborhood prompts: {cnt['neighborhood_prompts']}")
        print("-" * 80)

        if s["forward_pre_sanity"] is not None:
            fs = s["forward_pre_sanity"]
            print("Forward pre sanity (GT preferred BEFORE forward edit):")
            print(f"  rewrite:  {fmt_pct(fs['pre_rewrite_gt_preferred_rate'])}")
            print(f"  rephrase: {fmt_pct(fs['pre_rephrase_gt_preferred_rate'])}")
            print("-" * 80)

        print("Post metrics (ROME/CounterFact-style):")
        print(f"  ES: {fmt_pct(pm['ES'])}   (95% CI: {fmt_pct(pm['ES_ci95'][0])} .. {fmt_pct(pm['ES_ci95'][1])})")
        print(f"  EM: {pm['EM']:.6f}" if pm["EM"] == pm["EM"] else "  EM: nan")
        print(f"  PS: {fmt_pct(pm['PS'])}   (95% CI: {fmt_pct(pm['PS_ci95'][0])} .. {fmt_pct(pm['PS_ci95'][1])})")
        print(f"  PM: {pm['PM']:.6f}" if pm["PM"] == pm["PM"] else "  PM: nan")
        print(f"  NS: {fmt_pct(pm['NS'])}   (95% CI: {fmt_pct(pm['NS_ci95'][0])} .. {fmt_pct(pm['NS_ci95'][1])})")
        print(f"  NM: {pm['NM']:.6f}" if pm["NM"] == pm["NM"] else "  NM: nan")
        print(f"   S: {fmt_pct(pm['S'])}")

        print("-" * 80)
        print(f"Extraction used (mode={s['mode']}): core_cases={used_core[s['mode']]}, fallback_cases={used_fallback[s['mode']]}")

    if args.ppl and ppl_summary is not None:
        print("\n" + "=" * 80)
        print("PPL stability (Butterfly Effect proxy)")
        print("-" * 80)

        up = ppl_summary["unpaired"]
        print("Unpaired (each mean uses its own finite subset):")
        print(f"  Records with M0 (finite): {up['counts']['records_with_M0']}")
        print(f"  Records with M1 (finite): {up['counts']['records_with_M1_finite']} (dropped non-finite: {up['counts']['dropped_m1_nonfinite']})")
        print(f"  Records with M2 (finite): {up['counts']['records_with_M2_finite']} (dropped non-finite: {up['counts']['dropped_m2_nonfinite']})")
        print(f"  Valid rate M1: {fmt_pct(up['counts']['valid_rate_M1'])}")
        print(f"  Valid rate M2: {fmt_pct(up['counts']['valid_rate_M2'])}")

        ppl = up["ppl"]
        print(f"  M0 mean: {ppl['M0_mean']:.6f} (CI95: {ppl['M0_ci95'][0]:.6f} .. {ppl['M0_ci95'][1]:.6f})")
        print(f"  M1 mean: {ppl['M1_mean']:.6f} (CI95: {ppl['M1_ci95'][0]:.6f} .. {ppl['M1_ci95'][1]:.6f})")
        print(f"  M2 mean: {ppl['M2_mean']:.6f} (CI95: {ppl['M2_ci95'][0]:.6f} .. {ppl['M2_ci95'][1]:.6f})")

        d = up["deltas"]
        print(f"  ΔPPL forward (M1-M0) mean: {d['forward_abs_mean']:.6f} (CI95: {d['forward_abs_ci95'][0]:.6f} .. {d['forward_abs_ci95'][1]:.6f})")
        print(f"  ΔPPL forward rel (M1/M0) mean: {d['forward_rel_mean']:.6f}")
        print(f"  ΔPPL rollback (M2-M0) mean: {d['rollback_abs_mean']:.6f} (CI95: {d['rollback_abs_ci95'][0]:.6f} .. {d['rollback_abs_ci95'][1]:.6f})")
        print(f"  ΔPPL rollback rel (M2/M0) mean: {d['rollback_rel_mean']:.6f}")
        print(f"  ΔPPL rollback_vs_forward (M2-M1) mean: {d['rollback_vs_forward_abs_mean']:.6f}")
        print(f"  ΔPPL rollback_vs_forward rel (M2/M1) mean: {d['rollback_vs_forward_rel_mean']:.6f}")

        pr = ppl_summary["paired"]
        print("-" * 80)
        print("Paired (recommended; requires M0,M1,M2 all finite for the same record):")
        print(f"  Records with M0&M1&M2 (finite): {pr['counts']['records_with_M0_M1_M2_finite']}")
        print(f"  Paired rate: {fmt_pct(pr['counts']['paired_rate'])}")

        pp = pr["ppl"]
        print(f"  M0 mean: {pp['M0_mean']:.6f} (CI95: {pp['M0_ci95'][0]:.6f} .. {pp['M0_ci95'][1]:.6f})")
        print(f"  M1 mean: {pp['M1_mean']:.6f} (CI95: {pp['M1_ci95'][0]:.6f} .. {pp['M1_ci95'][1]:.6f})")
        print(f"  M2 mean: {pp['M2_mean']:.6f} (CI95: {pp['M2_ci95'][0]:.6f} .. {pp['M2_ci95'][1]:.6f})")

        pd = pr["deltas"]
        print(f"  ΔPPL forward (M1-M0) mean: {pd['forward_abs_mean']:.6f} (CI95: {pd['forward_abs_ci95'][0]:.6f} .. {pd['forward_abs_ci95'][1]:.6f})")
        print(f"  ΔPPL rollback (M2-M0) mean: {pd['rollback_abs_mean']:.6f} (CI95: {pd['rollback_abs_ci95'][0]:.6f} .. {pd['rollback_abs_ci95'][1]:.6f})")
        print(f"  ΔPPL rollback_vs_forward (M2-M1) mean: {pd['rollback_vs_forward_abs_mean']:.6f} (CI95: {pd['rollback_vs_forward_abs_ci95'][0]:.6f} .. {pd['rollback_vs_forward_abs_ci95'][1]:.6f})")
        print(f"  ΔPPL forward rel (M1/M0) mean: {pd['forward_rel_mean']:.6f}")
        print(f"  ΔPPL rollback rel (M2/M0) mean: {pd['rollback_rel_mean']:.6f}")
        print(f"  ΔPPL rollback_vs_forward rel (M2/M1) mean: {pd['rollback_vs_forward_rel_mean']:.6f}")

        c = ppl_summary["collapse"]
        print("-" * 80)
        print(f"Collapse rate M1: {fmt_pct(c['collapse_rate_M1'])} (count={c['collapse_count_M1']})")
        print(f"Collapse rate M2: {fmt_pct(c['collapse_rate_M2'])} (count={c['collapse_count_M2']})")
        if c["thresholds_rel_observed"]["min"] is not None:
            print(f"Observed rel threshold range: {c['thresholds_rel_observed']['min']} .. {c['thresholds_rel_observed']['max']}")

    out_obj = {
        "input_path": path,
        "records_read": records_read,
        "modes": summaries,
        "ppl_summary": ppl_summary,
        "extraction": {
            "prefer_core": bool(args.prefer_core),
            "core_cases": used_core,
            "fallback_cases": used_fallback,
            "ppl_enabled": bool(args.ppl),
            "ppl_records_used": ppl_records_used,
        },
        "notes": {
            "forward_ES_PS": "post success iff logP(target_new) > logP(ground_truth)",
            "rollback_ES_PS": "post success iff logP(ground_truth) > logP(target_new) (as logged; GT/NEW not swapped in JSON)",
            "NS": "success iff logP(ground_truth) > logP(target_new) on neighborhood prompts",
            "S": "harmonic_mean(ES, PS, NS)",
            "PPL_proxy": "Compute mean PPL before/after edits on a fixed external corpus; report deltas and collapse rate using thresholds (Butterfly Effect proxy).",
            "PPL_filter": "Non-finite PPL values (inf/nan) are filtered out before aggregation; drop counts are reported.",
            "PPL_paired": "Paired statistics require M0,M1,M2 all finite in the same record; this is the recommended view for BE-style evaluation.",
        },
    }

    if args.out:
        out_path = os.path.expanduser(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())