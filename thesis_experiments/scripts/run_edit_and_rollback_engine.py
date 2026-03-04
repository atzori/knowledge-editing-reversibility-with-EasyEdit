from __future__ import annotations

import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate.evaluate_utils import batch_target_log_likelihood

from thesis_experiments.scripts.ke_core import (
    force_hf_home,
    get_tokenizer,
    load_hparams,
)

from thesis_experiments.scripts.butterfly_effect_ppl import (
    butterfly_report,
    compute_ppl,
    is_collapse,
    load_ppl_texts_from_json,
)

from thesis_experiments.scripts.utils_io import (
    _as_bool,
    _as_int,
    _as_str,
    _is_cuda_oom,
    _log_stats_cache_status,
    log_step,
    raise_path_error,
    set_start_time,
)

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# Global startTime used by existing logging helpers
startTime = str(datetime.now().isoformat()).replace(":", "")
set_start_time(startTime)


# ----------------------------
# Path utilities (portable)
# ----------------------------
def _resolve_cfg_path(cfg_file: Path, raw_path: Any) -> Path:
    """Resolve a path from YAML in a cross-platform way.

    - Expands env vars and ~
    - If relative, resolves against the YAML file directory
    - Normalizes and returns an absolute Path
    """
    if raw_path is None:
        raise ValueError("Path value is None")

    if isinstance(raw_path, Path):
        p = raw_path
    else:
        p = Path(os.path.expandvars(os.path.expanduser(str(raw_path).strip())))

    if not p.is_absolute():
        p = (cfg_file.parent / p)

    return p.resolve()


def _hparams_to_dict(hparams_obj: Any) -> Any:
    """Best-effort conversion of hparams (usually a dataclass) to a JSON-serializable dict."""
    if hparams_obj is None:
        return None
    try:
        if is_dataclass(hparams_obj):
            return asdict(hparams_obj)
    except Exception:
        pass
    try:
        return dict(vars(hparams_obj))
    except Exception:
        return str(hparams_obj)


def _call_apply_algo(
    *,
    editor: BaseEditor,
    model,
    requests: List[Dict[str, Any]],
    hparams: Any,
    suppress_internal_prints: bool,
):
    """Apply edit via EasyEdit and optionally suppress internal prints."""
    kwargs = dict(
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
    )
    if suppress_internal_prints:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            return editor.apply_algo(model, editor.tok, requests, hparams, **kwargs)
    return editor.apply_algo(model, editor.tok, requests, hparams, **kwargs)


def _metric_scalar(v: Any) -> Optional[float]:
    """Convert a metric leaf (scalar or list of scalars) into a single float."""
    if v is None:
        return None
    if isinstance(v, np.generic):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        xs: List[float] = []
        for it in v:
            if isinstance(it, np.generic):
                xs.append(float(it))
            elif isinstance(it, (int, float)):
                xs.append(float(it))
            else:
                return None
        if not xs:
            return None
        m = float(np.mean(xs))
        if np.isnan(m) or np.isinf(m):
            return None
        return m
    return None


def _mean_metrics_dict(per_case_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively average numeric leaves across cases, weighting each case equally."""
    out: Dict[str, Any] = {}
    keys: set = set()
    for d in per_case_dicts:
        if isinstance(d, dict):
            keys.update(d.keys())

    for k in sorted(keys):
        vals = [d.get(k) for d in per_case_dicts if isinstance(d, dict) and k in d]
        if not vals:
            continue

        if all(isinstance(v, dict) for v in vals):
            child = _mean_metrics_dict(vals)  # type: ignore[arg-type]
            if child:
                out[k] = child
            continue

        scalars: List[float] = []
        for v in vals:
            s = _metric_scalar(v)
            if s is not None:
                scalars.append(s)

        if scalars:
            out[k] = float(np.mean(scalars))

    return out


def _compute_and_store_be(
    *,
    results_json: Dict[str, Any],
    slot: str,
    slot_desc: str,
    ppl_before: float,
    texts: List[str],
    model,
    tokenizer,
    device,
    batch_size: int,
    add_start_token: bool,
    max_length: Optional[int],
    collapse_rel_threshold: float,
    collapse_abs_threshold: Optional[float],
) -> None:
    log_step(f"BE: computing PPL on {slot} ({slot_desc}) (time-consuming).")
    ppl_after, _ = compute_ppl(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        add_start_token=add_start_token,
        max_length=max_length,
    )
    rep = butterfly_report(ppl_before=ppl_before, ppl_after=ppl_after)
    collapse = is_collapse(
        rep,
        rel_threshold=collapse_rel_threshold,
        abs_threshold=collapse_abs_threshold,
    )

    print(
        f"\n=== BE ({slot}) ===\n"
        f"mean_ppl: {ppl_after:.4f}\n"
        f"delta_abs: {rep.ppl_delta_abs:.4f}\n"
        f"delta_rel: {rep.ppl_delta_rel:.4f}\n"
        f"is_collapse: {collapse}"
    )
    results_json["ppl"][slot] = float(ppl_after)
    results_json["be_report"][slot] = {
        **rep.to_dict(),
        "is_collapse": bool(collapse),
        "collapse_rel_threshold": float(collapse_rel_threshold),
        "collapse_abs_threshold": (
            float(collapse_abs_threshold) if collapse_abs_threshold is not None else None
        ),
    }


def _resolve_eval_metric(exp_eval_metric: Optional[str], method: str) -> str:
    metric_raw = (exp_eval_metric or "").strip().lower()
    if metric_raw:
        aliases = {
            "logprob": "log_prob",
            "log-likelihood": "log_likelihood",
            "loglikelihood": "log_likelihood",
            "rome": "rome",
            "log_prob": "log_prob",
            "log_likelihood": "log_likelihood",
            "token_em": "token_em",
            "exact_match": "exact match",
            "exact match": "exact match",
            "ppl": "ppl",
            "ood_ppl": "ood_ppl",
        }
        if metric_raw not in aliases:
            raise ValueError(
                f"Invalid exp_eval_metric='{exp_eval_metric}'. "
                "Supported: log_prob|log_likelihood|rome|token_em|exact match|ppl|ood_ppl."
            )
        return aliases[metric_raw]

    # Default to ROME-style probabilistic evaluation for editing experiments.
    if method in ("rome", "memit"):
        return "log_prob"
    return "exact match"
    
# ============================
# PART 2: request builders + eval helpers (single-sample)
# ============================

@dataclass(frozen=True)
class PairSpec:
    desired: str
    undesired: str


def get_pair_spec(record: Dict[str, Any], metric: str, phase: str, time: str) -> PairSpec:
    """
    Central policy:
    - keep canonical fields untouched:
        ground_truth = original fact, target_new = edited fact
    - vary desired/undesired only via phase/time.
    """
    if metric not in ("rewrite", "rephrase"):
        raise ValueError(f"Unsupported metric='{metric}' for pair spec.")
    if phase not in ("forward", "rollback") or time not in ("pre", "post"):
        raise ValueError(f"Invalid phase/time: phase={phase}, time={time}")

    gt = str(record["ground_truth"]).rstrip()
    tn = str(record["target_new"]).rstrip()

    # Forward stage:
    # - pre: model should prefer GT over NEW
    # - post: model should prefer NEW over GT
    if phase == "forward":
        return PairSpec(desired=gt, undesired=tn) if time == "pre" else PairSpec(desired=tn, undesired=gt)

    # Rollback stage:
    # - pre (before inverse edit): model is edited -> prefer NEW
    # - post (after inverse edit): model should return to GT
    return PairSpec(desired=tn, undesired=gt) if time == "pre" else PairSpec(desired=gt, undesired=tn)


def score_pair_logprob(
    model,
    tok,
    prompt: str,
    desired: str,
    undesired: str,
    device,
) -> Dict[str, List[float]]:
    """
    Compute logprob margin between desired and undesired completions.
    Returns per-example lists (len=1 for single prompt).
    """
    desired_stats = batch_target_log_likelihood(model, tok, prompt, desired, device)
    undesired_stats = batch_target_log_likelihood(model, tok, prompt, undesired, device)

    desired_lp = desired_stats["sum_logprobs"]
    undesired_lp = undesired_stats["sum_logprobs"]
    margin = [d - u for d, u in zip(desired_lp, undesired_lp)]

    return {
        "acc": [float(m > 0) for m in margin],
        "margin": margin,
        "desired_logprob": desired_lp,
        "undesired_logprob": undesired_lp,
        "desired_token_count": desired_stats["token_counts"],
        "undesired_token_count": undesired_stats["token_counts"],
    }


def _build_requests_common_single(
    record: Dict[str, Any],
    *,
    target_new: str,
    ground_truth: str,
    locality_key: str = "neighborhood",
    portability_key: str = "rephrase",
    enable_portability_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build EasyEdit-style request list for a single record.

    NOTE: We still return a list because EasyEdit expects a batch.
    """
    prompt = str(record["prompt"]).rstrip()
    subject = str(record.get("subject", "")).rstrip()

    # If portability prompts exist, use first as rephrase_prompt; otherwise fallback to main prompt.
    ports = record.get("portability_prompts", []) or []
    rephrase_prompt = str(ports[0]).rstrip() if isinstance(ports, list) and ports else prompt

    # Locality prompts (neighborhood)
    locs = record.get("locality_prompts", []) or []
    loc_list = [str(x).rstrip() for x in (locs if isinstance(locs, list) else [locs]) if str(x).strip()]
    locality_inputs = None
    if loc_list:
        locality_inputs = {
            locality_key: {
                "prompt": [loc_list],  # list per case
                "ground_truth": [[str(ground_truth).rstrip()] * len(loc_list)],
            }
        }

    # Portability metrics (optional; uses portability_prompts list)
    portability_inputs = None
    if enable_portability_metrics:
        por_list = [str(x).rstrip() for x in (ports if isinstance(ports, list) else [ports]) if str(x).strip()]
        if por_list:
            portability_inputs = {
                portability_key: {
                    "prompt": [por_list],
                    "ground_truth": [[str(target_new).rstrip()] * len(por_list)],
                }
            }

    requests = _prepare_requests(
        [prompt],
        [str(target_new).rstrip()],
        [str(ground_truth).rstrip()],
        rephrase_prompts=[rephrase_prompt],
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject=[subject],
    )

    # Keep case_id for traceability
    if "case_id" in record:
        requests[0]["case_id"] = record["case_id"]

    return requests


def _build_eval_requests_single(
    record: Dict[str, Any],
    *,
    locality_key: str = "neighborhood",
) -> List[Dict[str, Any]]:
    """
    Canonical semantics for evaluation in every phase:
      ground_truth = original fact, target_new = edited fact
    """
    return _build_requests_common_single(
        record,
        target_new=str(record["target_new"]).rstrip(),
        ground_truth=str(record["ground_truth"]).rstrip(),
        locality_key=locality_key,
        enable_portability_metrics=False,
    )


def _build_apply_requests_single(
    record: Dict[str, Any],
    *,
    direction: str,
    locality_key: str = "neighborhood",
    portability_key: str = "rephrase",
    enable_portability_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build request for applying edits:
      - forward: GT -> NEW
      - inverse: NEW -> GT (implemented as target_new := ground_truth while keeping ground_truth unchanged)
    """
    if direction not in ("forward", "inverse"):
        raise ValueError(f"Invalid direction='{direction}'. Expected: forward|inverse.")

    gt = str(record["ground_truth"]).rstrip()
    tn = str(record["target_new"]).rstrip()

    apply_target = tn if direction == "forward" else gt

    return _build_requests_common_single(
        record,
        target_new=apply_target,
        ground_truth=gt,
        locality_key=locality_key,
        portability_key=portability_key,
        enable_portability_metrics=enable_portability_metrics,
    )


def _eval_quality_per_request_single(
    *,
    model,
    tok,
    eval_request: Dict[str, Any],
    phase: str,
    time: str,
    device,
) -> Dict[str, Any]:
    """
    Deterministic log-prob evaluation driven by explicit desired/undesired specs.
    Returns the same structure as the old per-request quality dict.
    """
    quality: Dict[str, Any] = {}

    # Rewrite
    rewrite_spec = get_pair_spec(eval_request, metric="rewrite", phase=phase, time=time)
    rw = score_pair_logprob(model, tok, eval_request["prompt"], rewrite_spec.desired, rewrite_spec.undesired, device)

    rw_new = batch_target_log_likelihood(model, tok, eval_request["prompt"], eval_request["target_new"], device)
    rw_gt = batch_target_log_likelihood(model, tok, eval_request["prompt"], eval_request["ground_truth"], device)

    quality["rewrite_acc"] = rw["acc"]
    quality["rewrite_margin"] = rw["margin"]
    quality["rewrite_target_new_logprob"] = rw_new["sum_logprobs"]
    quality["rewrite_ground_truth_logprob"] = rw_gt["sum_logprobs"]
    quality["rewrite_target_new_token_count"] = rw_new["token_counts"]
    quality["rewrite_ground_truth_token_count"] = rw_gt["token_counts"]

    # Rephrase (paraphrase)
    if eval_request.get("rephrase_prompt") is not None:
        rephrase_spec = get_pair_spec(eval_request, metric="rephrase", phase=phase, time=time)
        rp = score_pair_logprob(
            model, tok, eval_request["rephrase_prompt"], rephrase_spec.desired, rephrase_spec.undesired, device
        )
        rp_new = batch_target_log_likelihood(model, tok, eval_request["rephrase_prompt"], eval_request["target_new"], device)
        rp_gt = batch_target_log_likelihood(model, tok, eval_request["rephrase_prompt"], eval_request["ground_truth"], device)

        quality["rephrase_acc"] = rp["acc"]
        quality["rephrase_margin"] = rp["margin"]
        quality["rephrase_target_new_logprob"] = rp_new["sum_logprobs"]
        quality["rephrase_ground_truth_logprob"] = rp_gt["sum_logprobs"]
        quality["rephrase_target_new_token_count"] = rp_new["token_counts"]
        quality["rephrase_ground_truth_token_count"] = rp_gt["token_counts"]

    # Locality (neighborhood): phase-independent desired=locality ground-truth, undesired=edited target_new
    locality = eval_request.get("locality", {})
    quality["locality"] = {}
    if isinstance(locality, dict):
        for locality_key, loc_block in locality.items():
            prompts = loc_block.get("prompt", []) if isinstance(loc_block, dict) else []
            loc_gts = loc_block.get("ground_truth", []) if isinstance(loc_block, dict) else []
            accs: List[float] = []
            margins: List[float] = []
            gt_lps: List[float] = []
            tn_lps: List[float] = []
            gt_counts: List[int] = []
            tn_counts: List[int] = []

            # prompts and loc_gts are lists-of-lists (one per case)
            if prompts and isinstance(prompts[0], list):
                prompts_list = prompts[0]
                gts_list = loc_gts[0] if loc_gts and isinstance(loc_gts[0], list) else []
            else:
                prompts_list = prompts
                gts_list = loc_gts

            for p, loc_gt in zip(prompts_list, gts_list):
                loc = score_pair_logprob(model, tok, p, str(loc_gt), eval_request["target_new"], device)
                loc_gt_sc = batch_target_log_likelihood(model, tok, p, str(loc_gt), device)
                loc_tn_sc = batch_target_log_likelihood(model, tok, p, eval_request["target_new"], device)

                accs.extend(loc["acc"])
                margins.extend(loc["margin"])
                gt_lps.extend(loc_gt_sc["sum_logprobs"])
                tn_lps.extend(loc_tn_sc["sum_logprobs"])
                gt_counts.extend(loc_gt_sc["token_counts"])
                tn_counts.extend(loc_tn_sc["token_counts"])

            quality["locality"][f"{locality_key}_acc"] = accs
            quality["locality"][f"{locality_key}_margin"] = margins
            quality["locality"][f"{locality_key}_ground_truth_logprob"] = gt_lps
            quality["locality"][f"{locality_key}_target_new_logprob"] = tn_lps
            quality["locality"][f"{locality_key}_ground_truth_token_count"] = gt_counts
            quality["locality"][f"{locality_key}_target_new_token_count"] = tn_counts

    # Portability metrics are not computed in this single-sample runner (kept for schema compatibility)
    quality["portability"] = {}

    return quality


def _build_metrics_case_and_mean_single(
    *,
    eval_request: Dict[str, Any],
    pre_quality: Dict[str, Any],
    post_quality: Dict[str, Any],
    edit_time_sec: float,
    mean_case_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Keep the same shape as the old multi-case runner:
      - metrics_cases: list with one element
      - metrics_mean: mean over the single element
    """
    case = {
        "case_id": eval_request.get("case_id", ""),
        "requested_rewrite": eval_request,
        "time": float(edit_time_sec),
        "pre": pre_quality,
        "post": post_quality,
    }
    cases = [case]

    mean_pre = _mean_metrics_dict([case["pre"]])
    mean_post = _mean_metrics_dict([case["post"]])
    mean_metrics = {
        "case_id": str(mean_case_id),
        "time": float(edit_time_sec),
        "pre": mean_pre,
        "post": mean_post,
    }
    return cases, mean_metrics


def _store_stage_metrics(
    *,
    results_json: Dict[str, Any],
    stage_key: str,
    metrics_mean: Dict[str, Any],
    metrics_cases: List[Dict[str, Any]],
) -> None:
    results_json["metrics"][stage_key] = metrics_mean
    results_json.setdefault("metrics_per_case", {})
    results_json["metrics_per_case"][stage_key] = metrics_cases
    
# ============================
# PART 3: core engine (single CounterFact sample) — NO CLI / NO main
# ============================

def _pick_device_for_tensors(hparams_obj: Any, model: Any) -> str:
    """
    Pick a device string for input tensors / eval loops.

    - If model parallel (HF device_map), inputs should go to the model's first parameter device.
    - Otherwise, prefer hparams.device (e.g. cuda:1) to avoid accidental moves to cuda:0.
    """
    if not torch.cuda.is_available():
        return "cpu"

    model_parallel = bool(getattr(hparams_obj, "model_parallel", False))
    has_device_map = hasattr(model, "hf_device_map")

    if model_parallel or has_device_map:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return "cuda"

    dev = getattr(hparams_obj, "device", None)
    if dev is None or str(dev) == "-1":
        return "cuda"
    return f"cuda:{dev}"


def _maybe_move_model_to_device(model, device: str, label: str) -> None:
    if device == "cpu":
        return
    if hasattr(model, "hf_device_map"):
        log_step(f"{label}: device_map detected; skipping .to(...) to preserve sharding.", "INFO")
        return
    try:
        cur = str(next(model.parameters()).device)
    except Exception:
        cur = None
    if cur == device:
        log_step(f"{label}: already on {cur}; skipping .to(...).", "INFO")
        return
    log_step(f"{label}: moving to {device} (was {cur}).", "INFO")
    model.to(device)


def run_edit_and_rollback_engine(
    config_path: str,
    mode: str = "both",
    hparams: Optional[Any] = None,
    sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single-sample runner.

    Interface preserved:
      - returns results_json dict
      - keeps `config_path`, `mode`, `hparams`
    New:
      - accepts `sample` (normalized CounterFact record)
      - NO dataset loading here
    """
    if sample is None:
        raise ValueError("Missing required argument: sample (single CounterFact record).")

    if mode not in ("forward", "inverse", "both"):
        raise ValueError(f"Invalid mode='{mode}'. Expected forward|inverse|both.")

    t0 = perf_counter()

    results_json: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": None,  # filled only at the very end
        "config_path": str(Path(config_path)),
        "mode": mode,
        "method": None,
        "dataset_type": "counterfact",
        "dataset_source": "wrapper_sample",
        "dataset_size_loaded": 1,
        "sample_index": None,
        "case_id": sample.get("case_id", ""),
        "prompt": str(sample.get("prompt", "")).rstrip(),
        "subject": str(sample.get("subject", "")).rstrip(),
        "ground_truth": str(sample.get("ground_truth", "")).rstrip(),
        "target_new": str(sample.get("target_new", "")).rstrip(),
        "counts": {
            "locality_prompts": len(sample.get("locality_prompts", []) or []),
            "portability_prompts": len(sample.get("portability_prompts", []) or []),
            "ppl_texts": None,
        },
        "ppl": {},       # M0/M1/M2 mean_ppl
        "be_report": {}, # M1/M2 report dicts
        "metrics": {},   # forward/inverse_only/rollback
    }

    # ----------------------------
    # Experiment config loading
    # ----------------------------
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise_path_error("Experiment config file", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {cfg_path}")

    results_json["exp_config"] = cfg

    method = str(cfg.get("exp_method", "rome")).lower().strip()
    if method not in ("rome", "memit"):
        raise ValueError(f"Invalid exp_method='{method}'. Must be 'rome' or 'memit'.")
    results_json["method"] = method

    # Runner params
    forced_model_name = str(cfg.get("exp_forced_model_name", "gpt2-xl")).strip()
    suppress_internal = _as_bool(cfg.get("exp_suppress_internal_prints", True), True)
    eval_metric = _resolve_eval_metric(cfg.get("exp_eval_metric", None), method)
    enable_portability_metrics = _as_bool(cfg.get("exp_enable_portability_metrics", False), False)
    model_parallel_override = cfg.get("model_parallel", None)
    if model_parallel_override is not None:
        model_parallel_override = _as_bool(model_parallel_override, False)

    results_json["eval_metric"] = eval_metric

    # ----------------------------
    # Butterfly Effect (PPL) config
    # ----------------------------
    be_enabled = _as_bool(cfg.get("be_enabled", True), True)

    be_ppl_data_path_raw = _as_str(cfg.get("be_ppl_data_path", ""), "").strip()
    be_ppl_text_key = _as_str(cfg.get("be_ppl_text_key", "Text"), "Text").strip()
    be_ppl_max_items_raw = cfg.get("be_ppl_max_items", None)
    be_ppl_batch_size = _as_int(cfg.get("be_ppl_batch_size", 16), 16)
    be_ppl_add_start_token = _as_bool(cfg.get("be_ppl_add_start_token", False), False)
    be_ppl_max_length_raw = cfg.get("be_ppl_max_length", None)

    be_collapse_rel_threshold = float(cfg.get("be_collapse_rel_threshold", 1.5) or 1.5)
    be_collapse_abs_threshold = cfg.get("be_collapse_abs_threshold", None)

    be_ppl_max_items = None
    if be_ppl_max_items_raw is not None:
        be_ppl_max_items = _as_int(be_ppl_max_items_raw, 0)
        if be_ppl_max_items <= 0:
            raise ValueError(f"be_ppl_max_items must be > 0 when provided. Got: {be_ppl_max_items_raw}")

    be_ppl_max_length = None
    if be_ppl_max_length_raw is not None:
        be_ppl_max_length = _as_int(be_ppl_max_length_raw, 0)
        if be_ppl_max_length <= 0:
            raise ValueError(f"be_ppl_max_length must be > 0 when provided. Got: {be_ppl_max_length_raw}")

    # ----------------------------
    # Hparams config + editor build
    # ----------------------------
    exp_hparams_path_raw = str(cfg.get("exp_hparams_path", "")).strip()
    if not exp_hparams_path_raw and hparams is None:
        raise ValueError("Missing exp_hparams_path in experiment config or `hparams` parameter.")

    exp_hparams_path: Optional[Path] = None
    hparams_cfg: Dict[str, Any] = {}
    if exp_hparams_path_raw:
        exp_hparams_path_s = "".join(piece.strip() for piece in exp_hparams_path_raw.split("//"))
        exp_hparams_path = Path(exp_hparams_path_s)
        if exp_hparams_path.exists():
            hparams_cfg = yaml.safe_load(exp_hparams_path.read_text(encoding="utf-8")) or {}
            if not isinstance(hparams_cfg, dict):
                raise ValueError(f"Invalid hparams YAML structure (expected mapping) in: {exp_hparams_path}")
        elif hparams is None:
            raise_path_error("Hparams config file", exp_hparams_path)

    results_json["hparams_config"] = hparams_cfg if hparams_cfg else None

    force_hf_home()
    if hparams is None:
        if exp_hparams_path is None:
            raise ValueError("Cannot load hparams: exp_hparams_path is missing.")
        runtime_hparams = load_hparams(method, str(exp_hparams_path))
    else:
        runtime_hparams = hparams

    # Runner-level override: honor `model_parallel` from experiment config.
    if model_parallel_override is not None:
        runtime_hparams.model_parallel = bool(model_parallel_override)
        log_step(f"Override: model_parallel={runtime_hparams.model_parallel} (from exp config).", "INFO")

    runtime_hparams.model_name = forced_model_name

    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(runtime_hparams, attr):
            setattr(runtime_hparams, attr, None)

    results_json["hparams"] = _hparams_to_dict(runtime_hparams)

    # Ensure stats_dir exists if present in hparams
    if hasattr(runtime_hparams, "stats_dir"):
        stats_dir = getattr(runtime_hparams, "stats_dir")
        if stats_dir in (None, "", "null"):
            raise ValueError(
                "stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML."
            )
        stats_anchor = exp_hparams_path if exp_hparams_path is not None else cfg_path
        stats_dir_path = _resolve_cfg_path(stats_anchor, stats_dir)
        stats_dir_path.mkdir(parents=True, exist_ok=True)

    _log_stats_cache_status(runtime_hparams)

    # Build editor/model
    try:
        log_step("Instantiating BaseEditor and model (may download/load from cache).")
        editor: BaseEditor = BaseEditor.from_hparams(runtime_hparams)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM while instantiating the model.", "ERROR")
            return results_json
        raise

    tok = get_tokenizer(editor, forced_model_name)

    # Pick device + move model if needed
    device = _pick_device_for_tensors(runtime_hparams, editor.model)
    log_step(
        f"Selected device for tensors: {device} "
        f"(model_parallel={bool(getattr(runtime_hparams, 'model_parallel', False))}, "
        f"device_map={'yes' if hasattr(editor.model, 'hf_device_map') else 'no'})"
    )
    try:
        _maybe_move_model_to_device(editor.model, device, "Baseline model")
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM while moving baseline model.", "ERROR")
            return results_json
        raise

    # ----------------------------
    # BE: load PPL texts + baseline PPL (M0)
    # ----------------------------
    ppl_texts: List[str] = []
    ppl_m0: Optional[float] = None

    if be_enabled:
        if not be_ppl_data_path_raw:
            raise ValueError("be_enabled=True but 'be_ppl_data_path' is missing in config YAML.")
        be_ppl_data_path = Path(be_ppl_data_path_raw)
        if not be_ppl_data_path.exists():
            raise_path_error("BE PPL dataset file", be_ppl_data_path)

        log_step("Loading BE PPL evaluation texts (from JSON).")
        ppl_texts = load_ppl_texts_from_json(
            path=be_ppl_data_path,
            text_key=be_ppl_text_key,
            max_items=be_ppl_max_items,
        )
        results_json["counts"]["ppl_texts"] = len(ppl_texts)

        try:
            log_step("BE: computing PPL on M0 (baseline) (time-consuming).")
            ppl_m0, _ = compute_ppl(
                texts=ppl_texts,
                model=editor.model,
                tokenizer=tok,
                device=device,
                batch_size=be_ppl_batch_size,
                add_start_token=be_ppl_add_start_token,
                max_length=be_ppl_max_length,
            )
            results_json["ppl"]["M0"] = float(ppl_m0)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM while computing PPL on M0.", "ERROR")
                return results_json
            raise

    # ----------------------------
    # Build requests (single)
    # ----------------------------
    eval_request = _build_eval_requests_single(sample)[0]
    fwd_apply_requests = _build_apply_requests_single(sample, direction="forward", enable_portability_metrics=enable_portability_metrics)
    inv_apply_requests = _build_apply_requests_single(sample, direction="inverse", enable_portability_metrics=enable_portability_metrics)

    edited_model = None
    rollback_model = None

    def _run_stage(
        *,
        stage_key: str,
        phase: str,
        model_before,
        apply_requests: List[Dict[str, Any]],
        move_label: Optional[str] = None,
    ):
        pre_q = _eval_quality_per_request_single(
            model=model_before,
            tok=tok,
            eval_request=eval_request,
            phase=phase,
            time="pre",
            device=getattr(runtime_hparams, "device", 0),
        )

        t_edit = perf_counter()
        model_after, _ = _call_apply_algo(
            editor=editor,
            model=model_before,
            requests=apply_requests,
            hparams=runtime_hparams,
            suppress_internal_prints=suppress_internal,
        )
        edit_time = perf_counter() - t_edit

        if move_label is not None:
            _maybe_move_model_to_device(model_after, device, move_label)

        post_q = _eval_quality_per_request_single(
            model=model_after,
            tok=tok,
            eval_request=eval_request,
            phase=phase,
            time="post",
            device=getattr(runtime_hparams, "device", 0),
        )

        cases, mean = _build_metrics_case_and_mean_single(
            eval_request=eval_request,
            pre_quality=pre_q,
            post_quality=post_q,
            edit_time_sec=edit_time,
            mean_case_id="mean_n=1",
        )
        _store_stage_metrics(results_json=results_json, stage_key=stage_key, metrics_mean=mean, metrics_cases=cases)
        return model_after

    # Forward stage
    if mode in ("forward", "both"):
        log_step(f"{method.upper()} FORWARD: applying 1 edit (GT -> NEW).", "INFO")
        try:
            edited_model = _run_stage(
                stage_key="forward",
                phase="forward",
                model_before=editor.model,
                apply_requests=fwd_apply_requests,
                move_label="Edited model (M1)",
            )
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during FORWARD stage.", "ERROR")
                return results_json
            raise

        if be_enabled and ppl_m0 is not None:
            try:
                _compute_and_store_be(
                    results_json=results_json,
                    slot="M1",
                    slot_desc="after forward edit",
                    ppl_before=ppl_m0,
                    texts=ppl_texts,
                    model=edited_model,
                    tokenizer=tok,
                    device=device,
                    batch_size=be_ppl_batch_size,
                    add_start_token=be_ppl_add_start_token,
                    max_length=be_ppl_max_length,
                    collapse_rel_threshold=be_collapse_rel_threshold,
                    collapse_abs_threshold=be_collapse_abs_threshold,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while computing PPL on M1.", "ERROR")
                    return results_json
                raise

    # Inverse-only stage
    if mode == "inverse":
        log_step(f"{method.upper()} INVERSE-only: applying 1 edit (NEW -> GT).", "INFO")
        try:
            inv_model = _run_stage(
                stage_key="inverse_only",
                phase="rollback",
                model_before=editor.model,
                apply_requests=inv_apply_requests,
                move_label=None,
            )
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during INVERSE-only stage.", "ERROR")
                return results_json
            raise

    # Rollback stage
    if mode == "both":
        if edited_model is None:
            raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

        log_step(f"{method.upper()} ROLLBACK: applying 1 edit (NEW -> GT) on M1.", "INFO")
        try:
            rollback_model = _run_stage(
                stage_key="rollback",
                phase="rollback",
                model_before=edited_model,
                apply_requests=inv_apply_requests,
                move_label="Rollback model (M2)",
            )
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during ROLLBACK stage.", "ERROR")
                return results_json
            raise

        if be_enabled and ppl_m0 is not None:
            try:
                _compute_and_store_be(
                    results_json=results_json,
                    slot="M2",
                    slot_desc="after rollback",
                    ppl_before=ppl_m0,
                    texts=ppl_texts,
                    model=rollback_model,
                    tokenizer=tok,
                    device=device,
                    batch_size=be_ppl_batch_size,
                    add_start_token=be_ppl_add_start_token,
                    max_length=be_ppl_max_length,
                    collapse_rel_threshold=be_collapse_rel_threshold,
                    collapse_abs_threshold=be_collapse_abs_threshold,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while computing PPL on M2.", "ERROR")
                    return results_json
                raise

    results_json["elapsed_sec"] = float(perf_counter() - t0)
    return results_json
