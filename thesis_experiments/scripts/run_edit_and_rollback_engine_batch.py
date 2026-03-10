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
# PART 2: request builders + eval helpers (single + batch)
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


def _build_requests_common(
    records: List[Dict[str, Any]],
    *,
    target_new_list: List[str],
    ground_truth_list: List[str],
    locality_key: str = "neighborhood",
    portability_key: str = "rephrase",
    enable_portability_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build EasyEdit-style requests for a batch of records.

    This is used both by ROME (batch size 1) and MEMIT (true multi-edit batches).
    """
    prompts = [str(r["prompt"]).rstrip() for r in records]
    subjects = [str(r.get("subject", "")).rstrip() for r in records]

    rephrase_prompts: List[str] = []
    for r in records:
        ports = r.get("portability_prompts", []) or []
        rephrase_prompts.append(str(ports[0]).rstrip() if isinstance(ports, list) and ports else str(r["prompt"]).rstrip())

    loc_prompts: List[Optional[List[str]]] = []
    loc_gts: List[Optional[List[str]]] = []
    any_loc = False
    for gt, r in zip(ground_truth_list, records):
        locs = r.get("locality_prompts", []) or []
        loc_list = [str(x).rstrip() for x in (locs if isinstance(locs, list) else [locs]) if str(x).strip()]
        if loc_list:
            any_loc = True
            loc_prompts.append(loc_list)
            loc_gts.append([str(gt).rstrip()] * len(loc_list))
        else:
            loc_prompts.append(None)
            loc_gts.append(None)

    locality_inputs = None
    if any_loc:
        locality_inputs = {
            locality_key: {
                "prompt": loc_prompts,
                "ground_truth": loc_gts,
            }
        }

    por_prompts: List[Optional[List[str]]] = []
    por_gts: List[Optional[List[str]]] = []
    any_por = False
    for desired_target, r in zip(target_new_list, records):
        ports = r.get("portability_prompts", []) or []
        por_list = [str(x).rstrip() for x in (ports if isinstance(ports, list) else [ports]) if str(x).strip()]
        if por_list:
            any_por = True
            por_prompts.append(por_list)
            por_gts.append([str(desired_target).rstrip()] * len(por_list))
        else:
            por_prompts.append(None)
            por_gts.append(None)

    portability_inputs = None
    if enable_portability_metrics and any_por:
        portability_inputs = {
            portability_key: {
                "prompt": por_prompts,
                "ground_truth": por_gts,
            }
        }

    requests = _prepare_requests(
        prompts,
        target_new_list,
        ground_truth_list,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject=subjects,
    )
    for req, r in zip(requests, records):
        if "case_id" in r:
            req["case_id"] = r["case_id"]
    return requests


def _build_eval_requests(
    records: List[Dict[str, Any]],
    *,
    locality_key: str = "neighborhood",
) -> List[Dict[str, Any]]:
    """
    Canonical semantics for evaluation in every phase:
      ground_truth = original fact, target_new = edited fact
    """
    return _build_requests_common(
        records,
        target_new_list=[str(r["target_new"]).rstrip() for r in records],
        ground_truth_list=[str(r["ground_truth"]).rstrip() for r in records],
        locality_key=locality_key,
        enable_portability_metrics=False,
    )


def _build_apply_requests(
    records: List[Dict[str, Any]],
    *,
    direction: str,
    locality_key: str = "neighborhood",
    portability_key: str = "rephrase",
    enable_portability_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build requests used to apply edits:
      - forward: GT -> NEW
      - inverse: NEW -> GT (implemented as target_new := ground_truth, keeping ground_truth unchanged)
    """
    if direction not in ("forward", "inverse"):
        raise ValueError(f"Invalid direction='{direction}'. Expected: forward|inverse.")

    ground_truth_list = [str(r["ground_truth"]).rstrip() for r in records]
    if direction == "forward":
        target_new_list = [str(r["target_new"]).rstrip() for r in records]
    else:
        target_new_list = [str(r["ground_truth"]).rstrip() for r in records]

    return _build_requests_common(
        records,
        target_new_list=target_new_list,
        ground_truth_list=ground_truth_list,
        locality_key=locality_key,
        portability_key=portability_key,
        enable_portability_metrics=enable_portability_metrics,
    )


def _eval_quality_per_request(
    *,
    model,
    tok,
    requests: List[Dict[str, Any]],
    phase: str,
    time: str,
    device,
) -> List[Dict[str, Any]]:
    """
    Deterministic log-prob evaluation driven by explicit desired/undesired specs.
    Returns one quality dict per request.
    """
    outs: List[Dict[str, Any]] = []
    for req in requests:
        quality: Dict[str, Any] = {}

        rewrite_spec = get_pair_spec(req, metric="rewrite", phase=phase, time=time)
        rw = score_pair_logprob(model, tok, req["prompt"], rewrite_spec.desired, rewrite_spec.undesired, device)
        rw_new = batch_target_log_likelihood(model, tok, req["prompt"], req["target_new"], device)
        rw_gt = batch_target_log_likelihood(model, tok, req["prompt"], req["ground_truth"], device)
        quality["rewrite_acc"] = rw["acc"]
        quality["rewrite_margin"] = rw["margin"]
        quality["rewrite_target_new_logprob"] = rw_new["sum_logprobs"]
        quality["rewrite_ground_truth_logprob"] = rw_gt["sum_logprobs"]
        quality["rewrite_target_new_token_count"] = rw_new["token_counts"]
        quality["rewrite_ground_truth_token_count"] = rw_gt["token_counts"]

        if req.get("rephrase_prompt") is not None:
            rephrase_spec = get_pair_spec(req, metric="rephrase", phase=phase, time=time)
            rp = score_pair_logprob(model, tok, req["rephrase_prompt"], rephrase_spec.desired, rephrase_spec.undesired, device)
            rp_new = batch_target_log_likelihood(model, tok, req["rephrase_prompt"], req["target_new"], device)
            rp_gt = batch_target_log_likelihood(model, tok, req["rephrase_prompt"], req["ground_truth"], device)
            quality["rephrase_acc"] = rp["acc"]
            quality["rephrase_margin"] = rp["margin"]
            quality["rephrase_target_new_logprob"] = rp_new["sum_logprobs"]
            quality["rephrase_ground_truth_logprob"] = rp_gt["sum_logprobs"]
            quality["rephrase_target_new_token_count"] = rp_new["token_counts"]
            quality["rephrase_ground_truth_token_count"] = rp_gt["token_counts"]

        locality = req.get("locality", {})
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

                for p, loc_gt in zip(prompts, loc_gts):
                    loc = score_pair_logprob(model, tok, p, str(loc_gt), req["target_new"], device)
                    loc_gt_sc = batch_target_log_likelihood(model, tok, p, str(loc_gt), device)
                    loc_tn_sc = batch_target_log_likelihood(model, tok, p, req["target_new"], device)
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

        quality["portability"] = {}
        outs.append(quality)
    return outs


def _build_metrics_cases_and_mean(
    *,
    requests: List[Dict[str, Any]],
    pre_qualities: List[Dict[str, Any]],
    post_qualities: List[Dict[str, Any]],
    edit_time_sec: float,
    mean_case_id: str,
    aggregate_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build per-case metrics plus mean metrics.

    For MEMIT we store a single aggregate pseudo-case in `metrics_per_case` to keep one
    tuple per iteration while preserving the JSON schema used by ROME logs.
    """
    raw_cases: List[Dict[str, Any]] = []
    for i, (req, pre_q, post_q) in enumerate(zip(requests, pre_qualities, post_qualities)):
        raw_cases.append(
            {
                "case_id": req.get("case_id", i),
                "requested_rewrite": req,
                "time": float(edit_time_sec),
                "pre": pre_q,
                "post": post_q,
            }
        )

    mean_pre = _mean_metrics_dict([c["pre"] for c in raw_cases])
    mean_post = _mean_metrics_dict([c["post"] for c in raw_cases])
    mean_metrics = {
        "case_id": str(mean_case_id),
        "time": float(edit_time_sec),
        "pre": mean_pre,
        "post": mean_post,
    }

    if aggregate_only:
        aggregate_case = {
            "case_id": str(mean_case_id),
            "requested_rewrite": {"batch_level": True, "num_edits": len(raw_cases)},
            "time": float(edit_time_sec),
            "pre": mean_pre,
            "post": mean_post,
        }
        return [aggregate_case], mean_metrics

    return raw_cases, mean_metrics


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
# PART 3: core engine (ROME single edit + MEMIT batch edit) — NO CLI / NO main
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
    samples: Optional[List[Dict[str, Any]]] = None,
    indices: Optional[List[int]] = None,
    alg: Optional[str] = None,
    ppl_start: Optional[float] = None,
    sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Editing runner used by reverse_on_counterfact.

    Interface preserved:
      - returns results_json dict
      - keeps `config_path`, `mode`, `hparams`
      - `sample` (single record) is still accepted for backward compatibility
    New:
      - `samples`: list of normalized CounterFact records
      - `indices`: list of dataset indices aligned with samples
      - `alg`: optional explicit method override ('rome'|'memit')
      - `ppl_start`: optional precomputed M0 PPL to skip baseline PPL computation
    """
    if mode not in ("forward", "both"):
        raise ValueError(f"Invalid mode='{mode}'. Expected forward|both.")

    t0 = perf_counter()

    results_json: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": None,  # filled only at the very end
        "mode": mode,
        "method": None,
        "sample_index": None,
        "num_edits": None,
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

    method_raw = alg if alg is not None else cfg.get("alg", cfg.get("exp_method", "rome"))
    method = str(method_raw).lower().strip()
    if method not in ("rome", "memit"):
        raise ValueError(f"Invalid method='{method}'. Must be 'rome' or 'memit'.")
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

    # Inputs: preserve backward compatibility with the old single-sample signature.
    input_samples: List[Dict[str, Any]] = list(samples or ([] if sample is None else [sample]))
    if not input_samples:
        raise ValueError("Missing input samples. Pass `samples=[...]` or legacy `sample=...`.")
    input_indices: List[int] = [int(i) for i in (indices or [])]

    if method == "rome":
        if len(input_samples) > 1:
            log_step("alg=rome received multiple samples; only the first one will be used.", "WARNING")
        if len(input_indices) > 1:
            log_step("alg=rome received multiple indices; only indices[0] will be used.", "WARNING")
        selected_samples = [input_samples[0]]
        selected_indices = [input_indices[0]] if input_indices else []
    else:
        selected_samples = input_samples
        selected_indices = input_indices
        if selected_indices and len(selected_indices) != len(selected_samples):
            raise ValueError(
                "For alg=memit, `indices` length must match `samples` length "
                f"(got indices={len(selected_indices)}, samples={len(selected_samples)})."
            )

    if not selected_samples:
        raise ValueError("No valid samples available after method selection.")

    if selected_indices:
        results_json["sample_index"] = int(selected_indices[0])
    results_json["num_edits"] = int(len(selected_samples))
    if method == "memit":
        results_json["batch"] = {
            "indices": [int(i) for i in selected_indices],
            "num_edits": int(len(selected_samples)),
        }

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
        if ppl_start is not None:
            try:
                ppl_m0 = float(ppl_start)
            except Exception as e:
                raise ValueError(f"Invalid ppl_start value: {ppl_start}") from e
            if np.isnan(ppl_m0) or np.isinf(ppl_m0) or ppl_m0 < 0:
                raise ValueError(f"ppl_start must be a finite non-negative float. Got: {ppl_start}")

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

        if ppl_m0 is not None:
            log_step(f"BE: using provided ppl_start for M0: {ppl_m0:.6f}", "INFO")
            results_json["ppl"]["M0"] = float(ppl_m0)
        else:
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

    for i, s in enumerate(selected_samples):
        if not str(s.get("prompt", "")).strip():
            raise ValueError(f"Empty prompt in selected_samples[{i}] (method={method}).")

    # ----------------------------
    # Build requests (ROME: n=1, MEMIT: n=batch_size)
    # ----------------------------
    eval_requests = _build_eval_requests(selected_samples)
    if not eval_requests:
        raise RuntimeError("No evaluation requests could be built from input samples.")
    fwd_apply_requests = _build_apply_requests(
        selected_samples,
        direction="forward",
        enable_portability_metrics=enable_portability_metrics,
    )
    inv_apply_requests = _build_apply_requests(
        selected_samples,
        direction="inverse",
        enable_portability_metrics=enable_portability_metrics,
    )

    # For MEMIT, we report one aggregate tuple per iteration (batch-level metrics),
    # following the MEMIT paper evaluation setting: apply all edits jointly, then
    # evaluate rewrite/paraphrase/neighborhood over the full batch and aggregate.
    # Reference: Meng et al. (2022), "Mass-editing memory in a transformer",
    # experiments/evaluation section.
    aggregate_only = method == "memit"

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
        pre_q = _eval_quality_per_request(
            model=model_before,
            tok=tok,
            requests=eval_requests,
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

        post_q = _eval_quality_per_request(
            model=model_after,
            tok=tok,
            requests=eval_requests,
            phase=phase,
            time="post",
            device=getattr(runtime_hparams, "device", 0),
        )

        cases, mean = _build_metrics_cases_and_mean(
            requests=eval_requests,
            pre_qualities=pre_q,
            post_qualities=post_q,
            edit_time_sec=edit_time,
            mean_case_id=f"mean_n={len(eval_requests)}",
            aggregate_only=aggregate_only,
        )
        _store_stage_metrics(results_json=results_json, stage_key=stage_key, metrics_mean=mean, metrics_cases=cases)
        return model_after

    # Forward stage
    if mode in ("forward", "both"):
        log_step(
            f"{method.upper()} FORWARD: applying {len(fwd_apply_requests)} edit(s) (GT -> NEW).",
            "INFO",
        )
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

    # Rollback stage
    if mode == "both":
        if edited_model is None:
            raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

        log_step(
            f"{method.upper()} ROLLBACK: applying {len(inv_apply_requests)} edit(s) (NEW -> GT) on M1.",
            "INFO",
        )
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
