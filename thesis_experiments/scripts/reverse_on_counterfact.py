from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Engine module
from thesis_experiments.scripts import run_edit_and_rollback_engine as engine

# Reuse the same hparams loader used in your project.
from ke_core import load_hparams


def _read_yaml(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {path}")
    return cfg


@contextlib.contextmanager
def suppress_output(enabled: bool):
    """Redirect stdout/stderr to /dev/null when enabled."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _repo_root_from_this_file() -> Path:
    # reverse_on_counterfact.py is in thesis_experiments/scripts/
    # parents[2] -> repo root
    return Path(__file__).resolve().parents[2]


def _normalize_fraction(x: Any, default: float = 1.0) -> float:
    """
    Accept:
      - float 0..1
      - int (treated as float)
      - str like "0.5" or "50%"
    """
    if x is None:
        return float(default)

    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            try:
                v = float(s[:-1].strip()) / 100.0
            except Exception:
                v = float(default)
        else:
            try:
                v = float(s)
            except Exception:
                v = float(default)
    else:
        v = float(default)

    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def _load_counterfact_with_local_priority() -> Tuple[List[Dict[str, Any]], Path]:
    """
    Always try to load CounterFact from:
      repo_root/thesis_experiments/data/counterfact/counterfact.json
    If missing, download the official ROME CounterFact JSON into that folder.
    Never uses HuggingFace.
    """
    import torch

    REMOTE_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

    repo_root = _repo_root_from_this_file()
    cf_dir = repo_root / "thesis_experiments" / "data" / "counterfact"
    cf_path = cf_dir / "counterfact.json"

    if not cf_path.exists():
        engine.log_step(f"CounterFact not found at: {cf_path}", "WARNING")
        engine.log_step(f"Downloading CounterFact from: {REMOTE_URL}", "INFO")
        cf_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(REMOTE_URL, str(cf_path))

    engine.log_step(f"Loading CounterFact from: {cf_path}", "INFO")
    with cf_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"CounterFact JSON must be a list of records. Got: {type(records)} at {cf_path}")

    return records, cf_path


def _iter_indices(*, total_n: int, start_idx: int, end_idx: Optional[int]) -> Iterable[int]:
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0. Got: {start_idx}")
    if start_idx >= total_n:
        return []

    end = total_n if end_idx is None else min(int(end_idx), total_n)
    if end <= start_idx:
        return []

    return range(start_idx, end)


# ----------------------------
# CounterFact normalization (LOCAL to wrapper)
# ----------------------------
def _format_prompt_with_subject(prompt: str, subject: str) -> str:
    p = (prompt or "").strip()
    s = (subject or "").strip()
    if not p:
        return p
    if "{}" in p:
        try:
            return p.format(s)
        except Exception:
            return p.replace("{}", s)
    if "<SUBJECT>" in p:
        return p.replace("<SUBJECT>", s)
    return p


def _extract_target_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("str", "text", "name"):
            if k in x and isinstance(x[k], str):
                return x[k]
    return str(x)


def _extract_list_of_prompts(value: Any, subject: str) -> List[str]:
    out: List[str] = []
    if value is None:
        return out

    if isinstance(value, dict):
        value = value.get("prompts", value.get("prompt", None))

    if isinstance(value, str):
        out.append(_format_prompt_with_subject(value, subject))
        return [p for p in out if p.strip()]

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(_format_prompt_with_subject(item, subject))
            elif isinstance(item, dict):
                p = item.get("prompt", item.get("text", item.get("template", "")))
                if p:
                    out.append(_format_prompt_with_subject(str(p), subject))

    return [p for p in out if p.strip()]


def normalize_counterfact_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a CounterFact raw record into the canonical sample expected by the engine:
      - prompt, subject, ground_truth, target_new, case_id
      - locality_prompts (list[str])
      - portability_prompts (list[str])
    """
    rr = rec.get("requested_rewrite", rec)

    subject = str(rr.get("subject", rec.get("subject", "")) or "").strip()
    prompt_t = str(rr.get("prompt", rec.get("prompt", "")) or "").strip()
    prompt = _format_prompt_with_subject(prompt_t, subject)

    gt = _extract_target_str(rr.get("target_true", rr.get("ground_truth", rec.get("ground_truth", ""))))
    tn = _extract_target_str(rr.get("target_new", rr.get("target", rec.get("target_new", ""))))

    portability_raw = (
        rec.get("paraphrase_prompts", None)
        or rec.get("paraphrases", None)
        or rec.get("portability_prompts", None)
        or rec.get("portability", None)
    )

    locality_raw = (
        rec.get("neighborhood_prompts", None)
        or rec.get("neighborhood", None)
        or rec.get("locality_prompts", None)
        or rec.get("locality", None)
    )

    portability_prompts = _extract_list_of_prompts(portability_raw, subject)
    locality_prompts = _extract_list_of_prompts(locality_raw, subject)

    return {
        "case_id": rec.get("case_id", rec.get("id", "")),
        "prompt": prompt,
        "subject": subject,
        "ground_truth": gt,
        "target_new": tn,
        "locality_prompts": locality_prompts,
        "portability_prompts": portability_prompts,
    }


# ----------------------------
# ROME-style aggregate metrics (post-run)
# ----------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _to_float(x: Any) -> Optional[float]:
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _exp_safe(x: float) -> float:
    # Avoid overflow: if logp is too large, clamp.
    if x > 50:
        x = 50
    if x < -50:
        x = -50
    return float(math.exp(x))


def _extract_prob_pair(d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Try to extract (p_new, p_true) from a dict using common key conventions.
    Returns probabilities (not log-probabilities). If only log-probs are present,
    converts with exp().
    """
    # probability keys
    prob_key_pairs = [
        ("p_new", "p_true"),
        ("prob_new", "prob_true"),
        ("target_new_prob", "ground_truth_prob"),
        ("p_target_new", "p_ground_truth"),
        ("p_tn", "p_gt"),
        ("p_new_token", "p_true_token"),
    ]
    for kn, kt in prob_key_pairs:
        if kn in d and kt in d:
            pn = _to_float(d.get(kn))
            pt = _to_float(d.get(kt))
            if pn is not None and pt is not None:
                return pn, pt

    # log-probability keys
    log_key_pairs = [
        ("logp_new", "logp_true"),
        ("log_prob_new", "log_prob_true"),
        ("target_new_logprob", "ground_truth_logprob"),
        ("logp_target_new", "logp_ground_truth"),
        ("logp_tn", "logp_gt"),
    ]
    for kn, kt in log_key_pairs:
        if kn in d and kt in d:
            ln = _to_float(d.get(kn))
            lt = _to_float(d.get(kt))
            if ln is not None and lt is not None:
                return _exp_safe(ln), _exp_safe(lt)

    return None


def _walk(obj: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    """Yield (path, node) for all nodes in a nested dict/list structure."""
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (f"[{i}]",))


def _score_candidate_path(path: Tuple[str, ...], group_tokens: List[str]) -> int:
    """
    Higher is better. Prefer:
      - paths containing M1 / edited / after
      - paths containing group token(s)
    """
    p = "/".join(path).lower()
    score = 0

    # Prefer edited model outputs if both M0 and M1 are present
    if any(x in p for x in ("m1", "edited", "after", "post")):
        score += 6
    if any(x in p for x in ("m0", "before", "pre", "orig", "base")):
        score += 1

    # Token matches
    hits = sum(1 for t in group_tokens if t in p)
    score += 4 * hits

    # Slight preference for more direct paths
    score -= max(0, len(path) - 10)
    return score


def _extract_group_probs(result: Dict[str, Any], group: str) -> List[Tuple[float, float]]:
    """
    Extract a list of (p_new, p_true) pairs for a group:
      - group == "rewrite": main prompt (usually 1 pair)
      - group == "portability": paraphrases (0..n pairs)
      - group == "locality": neighborhood (0..n pairs)

    Robust strategy:
      1) Pass A: require token match in path (high precision)
      2) Pass B (fallback): allow any path, score by heuristics (improves coverage)
    """
    group_token_map = {
        "rewrite": ["rewrite", "requested", "main", "prompt"],
        "portability": ["portability", "paraphrase", "rephrase", "paraphrases"],
        "locality": ["locality", "neighborhood", "neighbor", "specificity"],
    }
    tokens = group_token_map.get(group, [group])

    candidates_a: List[Tuple[int, Tuple[str, ...], Tuple[float, float]]] = []
    candidates_b: List[Tuple[int, Tuple[str, ...], Tuple[float, float]]] = []

    for path, node in _walk(result):
        if not isinstance(node, dict):
            continue
        pair = _extract_prob_pair(node)
        if pair is None:
            continue

        p_low = "/".join(path).lower()
        has_token = any(t in p_low for t in tokens)

        score = _score_candidate_path(path, tokens)

        if has_token:
            candidates_a.append((score, path, pair))
        # Always keep fallback candidate too (lower precision but avoids 0-coverage)
        candidates_b.append((score, path, pair))

    selected_src = candidates_a if candidates_a else candidates_b
    if not selected_src:
        return []

    selected_src.sort(key=lambda x: x[0], reverse=True)
    top_score = selected_src[0][0]

    # Keep all candidates close to the best score (often one per prompt)
    selected_pairs = [pair for s, _, pair in selected_src if s >= top_score - 2]

    # For rewrite we usually want exactly one pair (best one)
    if group == "rewrite":
        return [selected_pairs[0]]

    return selected_pairs


def _harmonic_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals or any(v <= 0 for v in vals):
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


def compute_rome_aggregates_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    es_list, em_list = [], []
    ps_list, pm_list = [], []
    ns_list, nm_list = [], []

    cov_rewrite = 0
    cov_port = 0
    cov_local = 0
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                result = json.loads(line)
            except Exception:
                continue

            phase = "rollback" if "rollback" in result.get("metrics", {}) else "forward"
            post = result.get("metrics", {}).get(phase, {}).get("post", {})

            # ---------- REWRITE ----------
            ln = post.get("rewrite_target_new_logprob")
            lt = post.get("rewrite_ground_truth_logprob")
            if ln is not None and lt is not None:
                cov_rewrite += 1
                es_list.append(1.0 if ln > lt else 0.0)
                em_list.append(float(ln - lt))

            # ---------- PORTABILITY (rephrase) ----------
            ln = post.get("rephrase_target_new_logprob")
            lt = post.get("rephrase_ground_truth_logprob")
            if ln is not None and lt is not None:
                cov_port += 1
                ps_list.append(1.0 if ln > lt else 0.0)
                pm_list.append(float(ln - lt))

            # ---------- LOCALITY (neighborhood) ----------
            loc = post.get("locality", {})
            ln = loc.get("neighborhood_target_new_logprob")
            lt = loc.get("neighborhood_ground_truth_logprob")
            if ln is not None and lt is not None:
                cov_local += 1
                ns_list.append(1.0 if lt > ln else 0.0)
                nm_list.append(float(lt - ln))

    ES = float(np.mean(es_list)) if es_list else 0.0
    EM = float(np.mean(em_list)) if em_list else 0.0
    PS = float(np.mean(ps_list)) if ps_list else 0.0
    PM = float(np.mean(pm_list)) if pm_list else 0.0
    NS = float(np.mean(ns_list)) if ns_list else 0.0
    NM = float(np.mean(nm_list)) if nm_list else 0.0
    S = float(np.mean([ES, PS, NS])) if total > 0 else 0.0

    return {
        "total": total,
        "coverage_rewrite": cov_rewrite,
        "coverage_portability": cov_port,
        "coverage_locality": cov_local,
        "ES": ES,
        "EM": EM,
        "PS": PS,
        "PM": PM,
        "NS": NS,
        "NM": NM,
        "S": S,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--mode", default="both", choices=["forward", "inverse", "both"])
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _read_yaml(cfg_path)

    raw_records, cf_json_path = _load_counterfact_with_local_priority()
    total_n = len(raw_records)
    engine.log_step(f"CounterFact size: {total_n} | Local path: {cf_json_path}", "INFO")

    # Allow choosing which YAML key controls the fraction of CounterFact to run.
    # Default is exp_counterfact_percent, but you can override with exp_counterfact_percent_key.
    percent_key = str(cfg.get("exp_counterfact_percent_key", "exp_counterfact_percent")).strip() or "exp_counterfact_percent"
    frac = _normalize_fraction(cfg.get(percent_key, 1.0), default=1.0)

    start_idx = int(cfg.get("exp_reverse_start_index", 0))
    end_idx = int(total_n * frac)

    engine.log_step(
        f"Running on CounterFact indices: start={start_idx}, end={end_idx} "
        f"(fraction={frac:.2%}, percent_key={percent_key})",
        "INFO",
    )

    indices = list(_iter_indices(total_n=total_n, start_idx=start_idx, end_idx=end_idx))
    if not indices:
        raise RuntimeError(f"No indices to run (start={start_idx}, end={end_idx}, total={total_n}).")

    method = str(cfg.get("exp_method", "rome")).strip().lower()
    exp_hparams_path = str(cfg.get("exp_hparams_path", "")).strip()
    if not exp_hparams_path:
        raise ValueError("Missing exp_hparams_path in config (needed to load hparams once).")

    hp_path = Path(exp_hparams_path).expanduser()
    if not hp_path.is_absolute():
        hp_path = (_repo_root_from_this_file() / hp_path).resolve()
    if not hp_path.exists():
        raise FileNotFoundError(f"Hparams file not found: {hp_path}")

    hparams_obj = load_hparams(method, str(hp_path))

    out_path_raw = str(cfg.get("exp_reverse_out_path", "logs/reverse_on_counterfact_results.jsonl")).strip()
    if not out_path_raw:
        raise ValueError("exp_reverse_out_path in YAML is empty.")
    out_path = Path(out_path_raw).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: this is the only flag used to suppress internal prints
    suppress = bool(cfg.get("exp_suppress_internal_prints", False))

    iterator = indices
    if tqdm is not None:
        iterator = tqdm(indices, total=len(indices), desc="CounterFact", unit="case", dynamic_ncols=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in iterator:
            raw = raw_records[int(i)]
            sample = normalize_counterfact_record(raw)

            # Guard: prevent empty prompts causing tokenizer/generate crash
            if not str(sample.get("prompt", "")).strip():
                engine.log_step(f"[SKIP] Empty prompt at index={i} case_id={sample.get('case_id','')}", "WARNING")
                continue

            with suppress_output(suppress):
                result = engine.run_edit_and_rollback_engine(
                    config_path=str(cfg_path),
                    mode=args.mode or "both",
                    hparams=hparams_obj,
                    sample=sample,
                )

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                try:
                    ppl0 = result.get("ppl", {}).get("M0", None)
                    ppl1 = result.get("ppl", {}).get("M1", None)
                    iterator.set_postfix(ppl0=ppl0, ppl1=ppl1)
                except Exception:
                    pass

    print(f"Done. Wrote results to: {out_path}")

    out_path = Path(cfg.get("exp_reverse_out_path", "logs/reverse_counterfact.jsonl")).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Results aggregation (ROME-style metrics)
    if out_path.exists():
        aggr = compute_rome_aggregates_from_jsonl(out_path)
        total = aggr["total"]
        print("ROME aggregates (from JSONL):")
        print(f"  Coverage rewrite     : {aggr['coverage_rewrite']}/{total} ({(aggr['coverage_rewrite']/total*100 if total else 0):.1f}%)")
        print(f"  Coverage portability : {aggr['coverage_portability']}/{total} ({(aggr['coverage_portability']/total*100 if total else 0):.1f}%)")
        print(f"  Coverage locality    : {aggr['coverage_locality']}/{total} ({(aggr['coverage_locality']/total*100 if total else 0):.1f}%)")
        print(f"  ES={aggr['ES']:.4f} | EM={aggr['EM']:.6f}")
        print(f"  PS={aggr['PS']:.4f} | PM={aggr['PM']:.6f}")
        print(f"  NS={aggr['NS']:.4f} | NM={aggr['NM']:.6f}")
        print(f"  S ={aggr['S']:.4f}")
        print(f"  Phase used: {aggr['metrics_phase_used']}")
    else:
        print(f"[WARN] JSONL not found: {out_path}")


if __name__ == "__main__":
    main()