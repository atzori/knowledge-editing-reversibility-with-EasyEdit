from thesis_experiments.scripts.utils_io import (
    raise_path_error,
    _load_records,
    _normalize_record,
    _as_str,
    _as_int,
    _as_bool,
    save_results_json,
    _is_cuda_oom,
    _log_stats_cache_status,
    log_step,
    set_start_time
)

from thesis_experiments.scripts.pretty_print_utilities import (
    print_metrics_table,
    print_hparams_table,
    print_color,
)

from thesis_experiments.scripts.butterfly_effect_ppl import (
    load_ppl_texts_from_json,
    compute_ppl,
    butterfly_report,
    is_collapse,
)

from ke_core import (
    load_hparams,
    force_hf_home,
    get_tokenizer,
    generate_completion,
    apply_edit,
)

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate import compute_edit_quality
from time import perf_counter
from datetime import datetime
import json
import inspect
from dataclasses import asdict, is_dataclass
from pathlib import Path
import argparse
import yaml
import warnings
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
import numpy as np

import torch

import os

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

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

startTime = str(datetime.now().isoformat()).replace(":", "")
set_start_time(startTime)

def _print_azure(text: str) -> None:
    # Bright cyan ("azure") for terminal readability.
    print(f"\033[96m{text}\033[0m")

def _hparams_to_dict(hparams_obj: Any) -> Any:
    """
    Best-effort conversion of hparams (usually a dataclass) to a JSON-serializable dict.
    Falls back to vars()/str() if needed.
    """
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

def _load_counterfact_with_local_priority(
    cfg_path: Path,
    cfg: dict,
    allow_hf_fallback: bool,
):
    """
    Load CounterFact with priority:
    - If exp_local_dataset is set:
        * load from local path if it exists
        * if it does NOT exist:
            - fallback to HF only if allow_hf_fallback=True
            - otherwise raise a hard error
    - If exp_local_dataset is null:
        * load from HF only if allow_hf_fallback=True
        * otherwise raise a hard error
    """

    local_path_raw = cfg.get("exp_local_dataset", None)
    
    # ----------------------------
    # Case 1: local dataset specified
    # ----------------------------
    if local_path_raw:
        local_path = Path(local_path_raw)

        if local_path.exists():
            log_step(f"Loading CounterFact from LOCAL path: {local_path}", "INFO")

            with local_path.open("r", encoding="utf-8") as f:
                records = json.load(f)

            if not isinstance(records, list):
                raise ValueError(
                    f"Local CounterFact dataset must be a JSON list: {local_path}"
                )

            return records, "local"

        # Local path specified but NOT found
        msg = f"Local CounterFact dataset not found at path: {local_path}"

        if not allow_hf_fallback:
            raise FileNotFoundError(
                msg
                + " | HF fallback is disabled (exp_allow_hf_fallback=false)."
            )

        log_step(
            msg + " — falling back to HuggingFace (allowed).",
            "WARNING",
        )

    # ----------------------------
    # Case 2: HF fallback
    # ----------------------------
    if not allow_hf_fallback:
        raise RuntimeError(
            "HF fallback is disabled (exp_allow_hf_fallback=false) "
            "and no valid local CounterFact dataset was provided."
        )

    hf_dataset = cfg.get("hf_dataset", None)
    hf_split = cfg.get("hf_split", "test")
    hf_subset = cfg.get("hf_subset", None)

    if not hf_dataset:
        raise ValueError(
            "HF fallback requested but 'hf_dataset' is missing in config."
        )

    log_step(
        f"Loading CounterFact from HF: {hf_dataset} "
        f"(split={hf_split}, subset={hf_subset})",
        "INFO",
    )

    ds = load_dataset(hf_dataset, hf_subset, split=hf_split)
    records = [dict(r) for r in ds]

    return records, "hf"


# ----------------------------
# Unified logging helper
# ----------------------------
def _call_apply_edit(editor: BaseEditor, **kwargs):
    """
    Call apply_edit(...) safely:
    - If apply_edit supports extra kwargs (locality/portability), pass them.
    - Otherwise, silently drop unsupported keys.
    """
    sig = inspect.signature(apply_edit)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if accepts_var_kw:
        return apply_edit(editor, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = set(kwargs.keys()) - set(filtered.keys())

    if dropped:
        log_step(f"apply_edit does not support keys; dropped: {sorted(dropped)}", "WARNING")

    return apply_edit(editor, **filtered)


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

def _locality_outputs_to_acc(
    *,
    pre_quality: Dict[str, Any],
    post_quality: Dict[str, Any],
    request: Dict[str, Any],
) -> None:
    """Convert locality outputs -> acc exactly like EasyEdit's editor.py."""
    # Only if locality outputs exist
    pre_loc = pre_quality.get("locality", {})
    post_loc = post_quality.get("locality", {})
    if not isinstance(pre_loc, dict) or not isinstance(post_loc, dict):
        pre_quality.pop("locality", None)
        return

    req_loc = request.get("locality", {})
    if not isinstance(req_loc, dict):
        req_loc = {}

    for locality_key in req_loc.keys():
        out_key = f"{locality_key}_output"
        if out_key not in pre_loc or out_key not in post_loc:
            continue

        locality_result: List[float] = []
        try:
            for ans, label in zip(post_loc[out_key], pre_loc[out_key]):
                locality_result.append(float(np.mean(np.equal(ans, label))))
        except Exception:
            for ans, label in zip(post_loc[out_key], pre_loc[out_key]):
                locality_result.append(float(ans == label))

        post_loc[f"{locality_key}_acc"] = locality_result
        post_loc.pop(out_key, None)

    # Match EasyEdit: drop PRE locality entirely
    pre_quality.pop("locality", None)


def _build_batch_requests(
    records: List[Dict[str, Any]],
    *,
    direction: str,
    locality_key: str = "neighborhood",
    portability_key: str = "rephrase",
) -> List[Dict[str, Any]]:
    """
    Build a list of EasyEdit request dicts for a single MEMIT batch call.
    direction:
      - 'forward': GT -> NEW
      - 'inverse': NEW -> GT
    """
    if direction not in ("forward", "inverse"):
        raise ValueError(f"Invalid direction='{direction}'. Expected: forward|inverse.")

    prompts = [str(r["prompt"]).rstrip() for r in records]
    subjects = [str(r["subject"]).rstrip() for r in records]
    gt = [str(r["ground_truth"]).rstrip() for r in records]
    tn = [str(r["target_new"]).rstrip() for r in records]

    if direction == "forward":
        target_new = tn
        ground_truth = gt
    else:
        target_new = gt
        ground_truth = tn

    # Rephrase prompt: first portability prompt if present, else fallback to main prompt.
    rephrase_prompts: List[Optional[str]] = []
    for p, r in zip(prompts, records):
        ports = r.get("portability_prompts", []) or []
        if isinstance(ports, list) and ports:
            rephrase_prompts.append(str(ports[0]).rstrip())
        else:
            rephrase_prompts.append(p)

    # Locality inputs: one entry per edit (list[str] or None).
    loc_prompts: List[Optional[List[str]]] = []
    loc_gts: List[Optional[List[str]]] = []
    any_loc = False
    for r in records:
        locs = r.get("locality_prompts", []) or []
        loc_list = [str(x).rstrip() for x in (locs if isinstance(locs, list) else [locs]) if str(x).strip()]
        if loc_list:
            any_loc = True
            loc_prompts.append(loc_list)
            # Use original GT for consistent slicing in locality evaluation.
            loc_gts.append([str(r["ground_truth"]).rstrip()] * len(loc_list))
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

    # Portability inputs: paraphrases should map to the desired target for this direction.
    por_prompts: List[Optional[List[str]]] = []
    por_gts: List[Optional[List[str]]] = []
    any_por = False
    for desired, r in zip(target_new, records):
        ports = r.get("portability_prompts", []) or []
        por_list = [str(x).rstrip() for x in (ports if isinstance(ports, list) else [ports]) if str(x).strip()]
        if por_list:
            any_por = True
            por_prompts.append(por_list)
            por_gts.append([str(desired).rstrip()] * len(por_list))
        else:
            por_prompts.append(None)
            por_gts.append(None)

    portability_inputs = None
    if any_por:
        portability_inputs = {
            portability_key: {
                "prompt": por_prompts,
                "ground_truth": por_gts,
            }
        }

    requests = _prepare_requests(
        prompts,
        target_new,
        ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject=subjects,
    )

    # Keep original case_id for traceability (and cache keys if enabled).
    for req, r in zip(requests, records):
        if "case_id" in r:
            req["case_id"] = r["case_id"]

    return requests


def _eval_quality_per_request(
    *,
    model,
    model_name: str,
    hparams: Any,
    tok,
    requests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute compute_edit_quality(...) for each request."""
    device_id = getattr(hparams, "device", 0)
    outs: List[Dict[str, Any]] = []
    for req in requests:
        outs.append(
            compute_edit_quality(
                model,
                model_name,
                hparams,
                tok,
                req,
                device_id,
                test_generation=False,
            )
        )
    return outs


def _build_metrics_cases_and_mean(
    *,
    requests: List[Dict[str, Any]],
    pre_qualities: List[Dict[str, Any]],
    post_qualities: List[Dict[str, Any]],
    edit_time_sec: float,
    mean_case_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for i, (req, pre_q, post_q) in enumerate(zip(requests, pre_qualities, post_qualities)):
        _locality_outputs_to_acc(pre_quality=pre_q, post_quality=post_q, request=req)
        cases.append(
            {
                "case_id": req.get("case_id", i),
                "requested_rewrite": req,
                "time": float(edit_time_sec),
                "pre": pre_q,
                "post": post_q,
            }
        )

    mean_pre = _mean_metrics_dict([c["pre"] for c in cases])
    mean_post = _mean_metrics_dict([c["post"] for c in cases])
    mean_metrics = {
        "case_id": str(mean_case_id),
        "time": float(edit_time_sec),
        "pre": mean_pre,
        "post": mean_post,
    }
    return cases, mean_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Run sample with forward + inverse (rollback) + Butterfly Effect (PPL)."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--mode", default="both", choices=["forward", "inverse", "both"])
    args = parser.parse_args()
    t0 = perf_counter()

    results_json: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": None,  # filled only at the very end
        "config_path": str(Path(args.config)),
        "mode": args.mode,
        "method": None,
        "dataset_type": None,
        "dataset_source": None,
        "dataset_size_loaded": None,
        "sample_index": None,
        "case_id": None,
        "prompt": None,
        "subject": None,
        "ground_truth": None,
        "target_new": None,
        "counts": {
            "locality_prompts": None,
            "portability_prompts": None,
            "ppl_texts": None,
        },
        "ppl": {},       # M0/M1/M2 mean_ppl
        "be_report": {}, # M1/M2 report dicts (delta_abs/delta_rel/collapse)
        "metrics": {},   # forward/inverse_only/rollback
    }


    # ----------------------------
    # Experiment config loading
    # ----------------------------
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise_path_error("Experiment config file", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {cfg_path}")

    # Persist full experiment config into results.json for reproducibility.
    results_json["exp_config"] = cfg

    method = str(cfg.get("exp_method", "rome")).lower().strip()
    if method not in ("rome", "memit"):
        raise ValueError(f"Invalid exp_method='{method}'. Must be 'rome' or 'memit'.")

    dataset_type = _as_str(cfg.get("exp_dataset_type", "knowedit"), "knowedit").strip().lower()

    # ----------------------------
    # Hparams config (separate file)
    # ----------------------------
    exp_hparams_path_raw = str(cfg.get("exp_hparams_path", "")).strip()
    if not exp_hparams_path_raw:
        raise ValueError("Missing exp_hparams_path in experiment config (must point to a separate hparams YAML).")

    exp_hparams_path_pieces = exp_hparams_path_raw.split("//")
    exp_hparams_path = ""
    for piece in exp_hparams_path_pieces:
        exp_hparams_path += piece.strip()
    print_color(f"Resolved exp_hparams_path: {exp_hparams_path}", "green")
    exp_hparams_path = Path(exp_hparams_path)
    if not exp_hparams_path.exists():
        raise_path_error("Hparams config file", exp_hparams_path)

    hparams_cfg = yaml.safe_load(exp_hparams_path.read_text(encoding="utf-8")) or {}
    if not isinstance(hparams_cfg, dict):
        raise ValueError(f"Invalid hparams YAML structure (expected mapping) in: {exp_hparams_path}")
    if "alg_name" not in hparams_cfg:
        raise ValueError(f"Hparams YAML must contain 'alg_name'. Missing in: {exp_hparams_path}")

    # Persist raw hparams YAML contents into results.json (before runtime overrides).
    results_json["hparams_config"] = hparams_cfg

    # ----------------------------
    # Sample selection + runner params
    # ----------------------------
    sample_index = _as_int(cfg.get("exp_sample_index", 0), 0)
    forced_model_name = str(cfg.get("exp_forced_model_name", "gpt2-xl")).strip()
    max_new_tokens = _as_int(cfg.get("exp_max_new_tokens", 20), 20)
    temperature = float(cfg.get("exp_temperature", 1.0) or 1.0)
    num_edits = _as_int(cfg.get("exp_num_edits", 1), 1)
    do_sample = _as_bool(cfg.get("exp_do_sample", False), False)
    suppress_internal = _as_bool(cfg.get("exp_suppress_internal_prints", True), True)
    verbose = _as_bool(cfg.get("exp_verbose", False), False)
    print_metrics_foreach_case = _as_bool(cfg.get("exp_print_metrics_foreach_case", False), False)
    model_parallel_override = cfg.get("model_parallel", None)
    if model_parallel_override is not None:
        model_parallel_override = _as_bool(model_parallel_override, False)

    if max_new_tokens <= 0:
        raise ValueError(f"exp_max_new_tokens must be > 0. Got: {max_new_tokens}")
    if num_edits <= 0:
        raise ValueError(f"exp_num_edits must be > 0. Got: {num_edits}")

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
    # Load dataset records
    # ----------------------------
    log_step(f"Loading records (dataset_type={dataset_type})...", "INFO")

    if dataset_type == "counterfact":
        allow_hf_fallback = _as_bool(
            cfg.get("exp_allow_hf_fallback", True), True
        )

        raw_records, dataset_source = _load_counterfact_with_local_priority(
            cfg_path=cfg_path,
            cfg=cfg,
            allow_hf_fallback=allow_hf_fallback,
        )
    else:
        raw_records, dataset_source = _load_records(os.path.join("thesis_experiments", "data"), cfg)

    if not raw_records:
        raise RuntimeError("No records loaded from dataset (empty list).")
    if sample_index < 0 or sample_index >= len(raw_records):
        raise IndexError(f"exp_sample_index={sample_index} out of range. Dataset size={len(raw_records)}")

    if method != "memit" and num_edits != 1:
        raise ValueError(
            f"exp_method='{method}' does not support batch editing. "
            "Set exp_num_edits: 1, or use exp_method: 'memit'."
        )

    batch_start = sample_index
    batch_end = batch_start + num_edits
    if batch_end > len(raw_records):
        raise IndexError(
            f"Requested exp_num_edits={num_edits} starting at exp_sample_index={sample_index} "
            f"exceeds dataset size={len(raw_records)} (end_index={batch_end - 1})."
        )

    batch_indices = list(range(batch_start, batch_end))
    batch_recs = [_normalize_record(raw_records[i], dataset_type) for i in batch_indices]

    class _Sample:
        def __init__(self, d: Dict[str, Any]):
            self.case_id = d.get("case_id", "")
            self.prompt = d["prompt"]
            self.subject = d["subject"]
            self.ground_truth = d["ground_truth"]
            self.target_new = d["target_new"]
            self.locality_prompts = d.get("locality_prompts", [])
            self.portability_prompts = d.get("portability_prompts", [])

    samples = [_Sample(r) for r in batch_recs]
    probe = samples[0]
    probe_prompt = str(probe.prompt).rstrip()

    # ----------------------------
    # CounterFact: print all fetched prompts in azure
    # ----------------------------
    if dataset_type == "counterfact":
        _print_azure("\n=== COUNTERFACT BATCH PROMPTS (azure) ===")
        for i, r in enumerate(batch_recs):
            p = str(r.get("prompt", "")).rstrip()
            gt = str(r.get("ground_truth", "")).rstrip()
            tn = str(r.get("target_new", "")).rstrip()
            _print_azure(f"prompt: {p}| ground_truth: {gt} --> target_new: {tn}\n")

    # ----------------------------
    # Load hparams + build editor
    # ----------------------------
    log_step("Loading hparams + building editor (this can take time).")
    log_step(f"Using hparams file: {exp_hparams_path}")

    force_hf_home()
    hparams = load_hparams(method, str(exp_hparams_path))

    # Runner-level override: honor `model_parallel` from experiment config.
    # This controls EasyEdit's HF `device_map` selection inside BaseEditor.
    if model_parallel_override is not None:
        hparams.model_parallel = bool(model_parallel_override)
        log_step(f"Override: model_parallel={hparams.model_parallel} (from exp config).", "INFO")

    # Force model_name to avoid local-cache weirdness
    hparams.model_name = forced_model_name

    # Avoid custom paths that may conflict with your environment
    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(hparams, attr):
            setattr(hparams, attr, None)

    # Persist effective (runtime) hparams after overrides.
    results_json["hparams"] = _hparams_to_dict(hparams)

    # Ensure stats_dir exists if present in hparams (ROME uses it)
    if hasattr(hparams, "stats_dir"):
        stats_dir = getattr(hparams, "stats_dir")
        if stats_dir in (None, "", "null"):
            raise ValueError("stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML.")
        stats_dir_path = _resolve_cfg_path(exp_hparams_path, stats_dir)
        stats_dir_path.mkdir(parents=True, exist_ok=True)

    _log_stats_cache_status(hparams)

    # Build editor/model (time-consuming)
    try:
        log_step("Instantiating BaseEditor and model (may download/load from cache).")
        editor: BaseEditor = BaseEditor.from_hparams(hparams)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM while instantiating the model.", "ERROR")
            return
        raise

    tok = get_tokenizer(editor, forced_model_name)

    if verbose:
        print_hparams_table(hparams)

    # ----------------------------
    # BE: load PPL texts + choose device
    # ----------------------------
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

    device = _pick_device_for_tensors(hparams, editor.model)
    model_parallel = bool(getattr(hparams, "model_parallel", False))
    has_device_map = hasattr(editor.model, "hf_device_map")
    log_step(
        f"Selected device for tensors: {device} "
        f"(model_parallel={model_parallel}, device_map={'yes' if has_device_map else 'no'})"
    )

    def _maybe_move_model_to_device(model, label: str) -> None:
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
        try:
            log_step(f"{label}: moving to {device} (was {cur}).", "INFO")
            model.to(device)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step(f"Aborted: CUDA OOM while moving {label.lower()} to device.", "ERROR")
                raise
            raise

    # Baseline model is usually already placed by BaseEditor; keep this as a safety net.
    try:
        _maybe_move_model_to_device(editor.model, "Baseline model")
    except RuntimeError:
        return

    ppl_texts: List[str] = []
    ppl_m0: Optional[float] = None

    if be_enabled:
        if not be_ppl_data_path_raw:
            log_step("BE enabled but 'be_ppl_data_path' is missing in config YAML.", "ERROR")
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

        log_step(f"BE enabled. Loaded {len(ppl_texts)} PPL texts from: {be_ppl_data_path}")
        results_json["counts"]["ppl_texts"] = len(ppl_texts)
        if be_ppl_max_length is None:
            log_step("be_ppl_max_length is None. Long sequences may slow down PPL a lot.", "WARNING")

    # ----------------------------
    # Print sample summary
    # ----------------------------
    batch_case_ids = [s.case_id for s in samples]
    batch_locality_counts = [len(getattr(s, "locality_prompts", []) or []) for s in samples]
    batch_portability_counts = [len(getattr(s, "portability_prompts", []) or []) for s in samples]

    print("\n=== BATCH ===")
    print(f"method: {method}")
    print(f"dataset_type: {dataset_type}")
    print(f"dataset_source: {dataset_source}")
    if dataset_source == "local":
        print(f"dataset_path: {cfg.get('exp_dataset_path', '')}")
    else:
        print(f"hf_dataset: {cfg.get('hf_dataset', None)}")
        print(f"hf_split: {cfg.get('hf_split', 'test')}")
        print(f"hf_subset: {cfg.get('hf_subset', None)}")
    print(f"dataset_size_loaded: {len(raw_records)}")
    print(f"batch_num_edits: {num_edits}")
    print(f"batch_indices: {batch_indices}")
    print(f"batch_case_ids: {batch_case_ids}")

    print("\n--- PROBE SAMPLE (first in batch) ---")
    print(f"sample_index: {batch_indices[0]}")
    print(f"case_id: {probe.case_id}")
    print(f"prompt: {probe_prompt}")
    print(f"subject: {probe.subject}")
    print(f"ground_truth: {probe.ground_truth}")
    print(f"target_new: {probe.target_new}")
    print(f"locality_prompts: {len(probe.locality_prompts)}")
    print(f"portability_prompts: {len(probe.portability_prompts)}\n")

    # ----------------------------
    # Populate results metadata (for results.json)
    # ----------------------------
    results_json.update({
        "method": method,
        "dataset_type": dataset_type,
        "dataset_source": dataset_source,
        "dataset_size_loaded": len(raw_records),
        "sample_index": sample_index,
        "case_id": probe.case_id,
        "prompt": probe_prompt,
        "subject": probe.subject,
        "ground_truth": probe.ground_truth,
        "target_new": probe.target_new,
        "batch": {
            "num_edits": int(num_edits),
            "indices": batch_indices,
            "case_ids": batch_case_ids,
            "counts": {
                "locality_prompts_per_case": batch_locality_counts,
                "portability_prompts_per_case": batch_portability_counts,
            },
        },
    })
    results_json["counts"]["locality_prompts"] = len(probe.locality_prompts)
    results_json["counts"]["portability_prompts"] = len(probe.portability_prompts)


    # ----------------------------
    # BE: PPL on M0 (baseline)
    # ----------------------------
    if be_enabled:
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
            print(f"\n=== BE (M0) ===\nmean_ppl: {ppl_m0:.4f}")
            results_json["ppl"]["M0"] = float(ppl_m0)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM while computing PPL on M0.", "ERROR")
                return
            raise

    # ----------------------------
    # M0: behavioral probe
    # ----------------------------
    try:
        log_step("Behavioral probe (M0) on probe sample.")
        comp0 = generate_completion(
            editor.model,
            tok,
            probe_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print("\n=== BEHAVIORAL (M0, probe) ===")
        print(comp0)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM during behavioral probe (M0).", "ERROR")
            return
        raise

    results_json.setdefault("metrics_per_case", {})

    metrics_fwd_mean: Optional[Dict[str, Any]] = None
    edited_model = None  # M1
    rollback_model = None  # M2

    # ----------------------------
    # MEMIT batch path (single algo call per stage)
    # ----------------------------
    if method == "memit":
        fwd_requests = _build_batch_requests(batch_recs, direction="forward")
        inv_requests = _build_batch_requests(batch_recs, direction="inverse")

        # ----------------------------
        # Forward edit (M0 -> M1) in ONE batch call
        # ----------------------------
        if args.mode in ("forward", "both"):
            log_step(
                f"MEMIT batch FORWARD: applying {len(fwd_requests)} edits in one call (GT -> NEW).",
                "INFO",
            )

            try:
                pre_q = _eval_quality_per_request(
                    model=editor.model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=fwd_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during FORWARD pre-evaluation.", "ERROR")
                    return
                raise

            try:
                t_edit = perf_counter()
                edited_model, _ = editor.apply_algo(
                    editor.model,
                    editor.tok,
                    fwd_requests,
                    hparams,
                    copy=False,
                    return_orig_weights=False,
                    keep_original_weight=False,
                )
                edit_time = perf_counter() - t_edit
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during FORWARD MEMIT batch edit.", "ERROR")
                    return
                raise

            try:
                _maybe_move_model_to_device(edited_model, "Edited model (M1)")
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while moving edited model to device.", "ERROR")
                    return
                raise

            try:
                post_q = _eval_quality_per_request(
                    model=edited_model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=fwd_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during FORWARD post-evaluation.", "ERROR")
                    return
                raise

            fwd_cases, metrics_fwd_mean = _build_metrics_cases_and_mean(
                requests=fwd_requests,
                pre_qualities=pre_q,
                post_qualities=post_q,
                edit_time_sec=edit_time,
                mean_case_id=f"mean_n={len(fwd_requests)}",
            )

            results_json["metrics"]["forward"] = metrics_fwd_mean
            results_json["metrics_per_case"]["forward"] = fwd_cases

            print_metrics_table(metrics_fwd_mean, title="FORWARD METRICS (MEAN)")
            if print_metrics_foreach_case:
                print_metrics_table(fwd_cases, title="FORWARD METRICS (PER CASE)")

            # ----------------------------
            # BE: PPL on M1 (after forward edit)
            # ----------------------------
            if be_enabled and ppl_m0 is not None:
                try:
                    log_step("BE: computing PPL on M1 (after forward edit) (time-consuming).")
                    ppl_m1, _ = compute_ppl(
                        texts=ppl_texts,
                        model=edited_model,
                        tokenizer=tok,
                        device=device,
                        batch_size=be_ppl_batch_size,
                        add_start_token=be_ppl_add_start_token,
                        max_length=be_ppl_max_length,
                    )
                    rep = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m1)
                    collapse = is_collapse(
                        rep,
                        rel_threshold=be_collapse_rel_threshold,
                        abs_threshold=be_collapse_abs_threshold,
                    )

                    print(
                        "\n=== BE (M1) ===\n"
                        f"mean_ppl: {ppl_m1:.4f}\n"
                        f"delta_abs: {rep.ppl_delta_abs:.4f}\n"
                        f"delta_rel: {rep.ppl_delta_rel:.4f}\n"
                        f"is_collapse: {collapse}"
                    )

                    results_json["ppl"]["M1"] = float(ppl_m1)
                    results_json["be_report"]["M1"] = {
                        **rep.to_dict(),
                        "is_collapse": bool(collapse),
                        "collapse_rel_threshold": float(be_collapse_rel_threshold),
                        "collapse_abs_threshold": (
                            float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None
                        ),
                    }
                except RuntimeError as e:
                    if _is_cuda_oom(e):
                        log_step("Aborted: CUDA OOM while computing PPL on M1.", "ERROR")
                        return
                    raise

            # Behavioral probe on edited model (probe sample)
            try:
                log_step("Behavioral probe (M1) on probe sample.")
                comp1 = generate_completion(
                    edited_model,
                    tok,
                    probe_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                print("\n=== BEHAVIORAL (M1, probe) ===")
                print(comp1)
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during behavioral probe (M1).", "ERROR")
                    return
                raise

        # ----------------------------
        # Inverse only (M0 -> M1) in ONE batch call
        # ----------------------------
        if args.mode == "inverse":
            log_step(
                f"MEMIT batch INVERSE-only: applying {len(inv_requests)} edits in one call (NEW -> GT).",
                "INFO",
            )
            try:
                pre_q = _eval_quality_per_request(
                    model=editor.model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=inv_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during INVERSE-only pre-evaluation.", "ERROR")
                    return
                raise

            try:
                t_edit = perf_counter()
                inv_model, _ = editor.apply_algo(
                    editor.model,
                    editor.tok,
                    inv_requests,
                    hparams,
                    copy=False,
                    return_orig_weights=False,
                    keep_original_weight=False,
                )
                edit_time = perf_counter() - t_edit
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during INVERSE-only MEMIT batch edit.", "ERROR")
                    return
                raise

            try:
                post_q = _eval_quality_per_request(
                    model=inv_model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=inv_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during INVERSE-only post-evaluation.", "ERROR")
                    return
                raise

            inv_cases, metrics_inv_mean = _build_metrics_cases_and_mean(
                requests=inv_requests,
                pre_qualities=pre_q,
                post_qualities=post_q,
                edit_time_sec=edit_time,
                mean_case_id=f"mean_n={len(inv_requests)}",
            )
            results_json["metrics"]["inverse_only"] = metrics_inv_mean
            results_json["metrics_per_case"]["inverse_only"] = inv_cases
            print_metrics_table(metrics_inv_mean, title="INVERSE METRICS (MEAN)")
            if print_metrics_foreach_case:
                print_metrics_table(inv_cases, title="INVERSE METRICS (PER CASE)")

            try:
                log_step("Behavioral probe (after inverse-only) on probe sample.")
                comp2 = generate_completion(
                    inv_model,
                    tok,
                    probe_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                print("\n=== BEHAVIORAL (after inverse-only, probe) ===")
                print(comp2)
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during behavioral probe (inverse-only).", "ERROR")
                    return
                raise

        # ----------------------------
        # Both: rollback (M1 -> M2) in ONE batch call
        # ----------------------------
        if args.mode == "both":
            if edited_model is None:
                raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

            log_step(
                f"MEMIT batch ROLLBACK: applying {len(inv_requests)} edits in one call (NEW -> GT) on M1.",
                "INFO",
            )
            try:
                pre_q = _eval_quality_per_request(
                    model=edited_model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=inv_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during ROLLBACK pre-evaluation.", "ERROR")
                    return
                raise

            try:
                t_edit = perf_counter()
                rollback_model, _ = editor.apply_algo(
                    editor.model,
                    editor.tok,
                    inv_requests,
                    hparams,
                    copy=False,
                    return_orig_weights=False,
                    keep_original_weight=False,
                )
                edit_time = perf_counter() - t_edit
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during ROLLBACK MEMIT batch edit.", "ERROR")
                    return
                raise

            try:
                _maybe_move_model_to_device(rollback_model, "Rollback model (M2)")
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while moving rollback model to device.", "ERROR")
                    return
                raise

            try:
                post_q = _eval_quality_per_request(
                    model=rollback_model,
                    model_name=forced_model_name,
                    hparams=hparams,
                    tok=tok,
                    requests=inv_requests,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during ROLLBACK post-evaluation.", "ERROR")
                    return
                raise

            rb_cases, metrics_rb_mean = _build_metrics_cases_and_mean(
                requests=inv_requests,
                pre_qualities=pre_q,
                post_qualities=post_q,
                edit_time_sec=edit_time,
                mean_case_id=f"mean_n={len(inv_requests)}",
            )
            results_json["metrics"]["rollback"] = metrics_rb_mean
            results_json["metrics_per_case"]["rollback"] = rb_cases

            # ----------------------------
            # BE: PPL on M2 (after rollback)
            # ----------------------------
            if be_enabled and ppl_m0 is not None:
                try:
                    log_step("BE: computing PPL on M2 (after rollback) (time-consuming).")
                    ppl_m2, _ = compute_ppl(
                        texts=ppl_texts,
                        model=rollback_model,
                        tokenizer=tok,
                        device=device,
                        batch_size=be_ppl_batch_size,
                        add_start_token=be_ppl_add_start_token,
                        max_length=be_ppl_max_length,
                    )
                    rep2 = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m2)
                    collapse2 = is_collapse(
                        rep2,
                        rel_threshold=be_collapse_rel_threshold,
                        abs_threshold=be_collapse_abs_threshold,
                    )

                    print(
                        "\n=== BE (M2) ===\n"
                        f"mean_ppl: {ppl_m2:.4f}\n"
                        f"delta_abs: {rep2.ppl_delta_abs:.4f}\n"
                        f"delta_rel: {rep2.ppl_delta_rel:.4f}\n"
                        f"is_collapse: {collapse2}"
                    )

                    results_json["ppl"]["M2"] = float(ppl_m2)
                    results_json["be_report"]["M2"] = {
                        **rep2.to_dict(),
                        "is_collapse": bool(collapse2),
                        "collapse_rel_threshold": float(be_collapse_rel_threshold),
                        "collapse_abs_threshold": (
                            float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None
                        ),
                    }
                except RuntimeError as e:
                    if _is_cuda_oom(e):
                        log_step("Aborted: CUDA OOM while computing PPL on M2.", "ERROR")
                        return
                    raise

            print_metrics_table(metrics_rb_mean, title="INVERSE (ROLLBACK) METRICS (MEAN)")
            if print_metrics_foreach_case:
                print_metrics_table(rb_cases, title="INVERSE (ROLLBACK) METRICS (PER CASE)")

            try:
                log_step("Behavioral probe (M2) on probe sample.")
                comp2 = generate_completion(
                    rollback_model,
                    tok,
                    probe_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                print("\n=== BEHAVIORAL (M2, probe) ===")
                print(comp2)
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during behavioral probe (M2).", "ERROR")
                    return
                raise

    # ----------------------------
    # ROME single-edit fallback (first sample only)
    # ----------------------------
    else:
        if args.mode in ("forward", "both"):
            log_step("Applying FORWARD edit (GT -> NEW) on probe sample (ROME single-edit).")

            extra_eval = {}
            if isinstance(probe.locality_prompts, list) and probe.locality_prompts:
                extra_eval["locality_prompts"] = probe.locality_prompts
            if isinstance(probe.portability_prompts, list) and probe.portability_prompts:
                extra_eval["portability_prompts"] = probe.portability_prompts

            try:
                metrics_fwd, edited_model = _call_apply_edit(
                    editor,
                    prompt=probe_prompt,
                    subject=probe.subject,
                    ground_truth=probe.ground_truth,
                    target_new=probe.target_new,
                    verbose=verbose,
                    suppress_internal_prints=suppress_internal,
                    **extra_eval,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during FORWARD edit.", "ERROR")
                    return
                raise

            results_json["metrics"]["forward"] = metrics_fwd

            try:
                _maybe_move_model_to_device(edited_model, "Edited model (M1)")
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while moving edited model to device.", "ERROR")
                    return
                raise

            if be_enabled and ppl_m0 is not None:
                try:
                    log_step("BE: computing PPL on M1 (after forward edit) (time-consuming).")
                    ppl_m1, _ = compute_ppl(
                        texts=ppl_texts,
                        model=edited_model,
                        tokenizer=tok,
                        device=device,
                        batch_size=be_ppl_batch_size,
                        add_start_token=be_ppl_add_start_token,
                        max_length=be_ppl_max_length,
                    )
                    rep = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m1)
                    collapse = is_collapse(
                        rep,
                        rel_threshold=be_collapse_rel_threshold,
                        abs_threshold=be_collapse_abs_threshold,
                    )
                    print(
                        "\n=== BE (M1) ===\n"
                        f"mean_ppl: {ppl_m1:.4f}\n"
                        f"delta_abs: {rep.ppl_delta_abs:.4f}\n"
                        f"delta_rel: {rep.ppl_delta_rel:.4f}\n"
                        f"is_collapse: {collapse}"
                    )
                    results_json["ppl"]["M1"] = float(ppl_m1)
                    results_json["be_report"]["M1"] = {
                        **rep.to_dict(),
                        "is_collapse": bool(collapse),
                        "collapse_rel_threshold": float(be_collapse_rel_threshold),
                        "collapse_abs_threshold": (
                            float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None
                        ),
                    }
                except RuntimeError as e:
                    if _is_cuda_oom(e):
                        log_step("Aborted: CUDA OOM while computing PPL on M1.", "ERROR")
                        return
                    raise

            print_metrics_table(metrics_fwd, title="FORWARD METRICS")

        if args.mode == "inverse":
            log_step("Applying INVERSE edit only (NEW -> GT) on probe sample (ROME single-edit).")

            extra_eval = {}
            if isinstance(probe.locality_prompts, list) and probe.locality_prompts:
                extra_eval["locality_prompts"] = probe.locality_prompts
            if isinstance(probe.portability_prompts, list) and probe.portability_prompts:
                extra_eval["portability_prompts"] = probe.portability_prompts

            try:
                metrics_inv, inv_model = _call_apply_edit(
                    editor,
                    prompt=probe_prompt,
                    subject=probe.subject,
                    ground_truth=probe.target_new,
                    target_new=probe.ground_truth,
                    verbose=verbose,
                    suppress_internal_prints=suppress_internal,
                    **extra_eval,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during inverse-only edit.", "ERROR")
                    return
                raise

            results_json["metrics"]["inverse_only"] = metrics_inv
            print_metrics_table(metrics_inv, title="INVERSE METRICS")

        if args.mode == "both":
            if edited_model is None:
                raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

            log_step("Switching editor to edited model for rollback (ROME single-edit).")
            editor.model = edited_model

            log_step("Applying INVERSE edit (rollback: NEW -> GT) on probe sample (ROME single-edit).")

            extra_eval = {}
            if isinstance(probe.locality_prompts, list) and probe.locality_prompts:
                extra_eval["locality_prompts"] = probe.locality_prompts
            if isinstance(probe.portability_prompts, list) and probe.portability_prompts:
                extra_eval["portability_prompts"] = probe.portability_prompts

            try:
                metrics_inv, rollback_model = _call_apply_edit(
                    editor,
                    prompt=probe_prompt,
                    subject=probe.subject,
                    ground_truth=probe.target_new,
                    target_new=probe.ground_truth,
                    verbose=verbose,
                    suppress_internal_prints=suppress_internal,
                    **extra_eval,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM during rollback edit.", "ERROR")
                    return
                raise

            results_json["metrics"]["rollback"] = metrics_inv

            try:
                _maybe_move_model_to_device(rollback_model, "Rollback model (M2)")
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while moving rollback model to device.", "ERROR")
                    return
                raise

            if be_enabled and ppl_m0 is not None:
                try:
                    log_step("BE: computing PPL on M2 (after rollback) (time-consuming).")
                    ppl_m2, _ = compute_ppl(
                        texts=ppl_texts,
                        model=rollback_model,
                        tokenizer=tok,
                        device=device,
                        batch_size=be_ppl_batch_size,
                        add_start_token=be_ppl_add_start_token,
                        max_length=be_ppl_max_length,
                    )
                    rep2 = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m2)
                    collapse2 = is_collapse(
                        rep2,
                        rel_threshold=be_collapse_rel_threshold,
                        abs_threshold=be_collapse_abs_threshold,
                    )
                    print(
                        "\n=== BE (M2) ===\n"
                        f"mean_ppl: {ppl_m2:.4f}\n"
                        f"delta_abs: {rep2.ppl_delta_abs:.4f}\n"
                        f"delta_rel: {rep2.ppl_delta_rel:.4f}\n"
                        f"is_collapse: {collapse2}"
                    )
                    results_json["ppl"]["M2"] = float(ppl_m2)
                    results_json["be_report"]["M2"] = {
                        **rep2.to_dict(),
                        "is_collapse": bool(collapse2),
                        "collapse_rel_threshold": float(be_collapse_rel_threshold),
                        "collapse_abs_threshold": (
                            float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None
                        ),
                    }
                except RuntimeError as e:
                    if _is_cuda_oom(e):
                        log_step("Aborted: CUDA OOM while computing PPL on M2.", "ERROR")
                        return
                    raise

            print_metrics_table(metrics_inv, title="INVERSE (ROLLBACK) METRICS")

    # ----------------------------
    # Finalize + persist results.json ONLY after everything is computed
    # ----------------------------
    results_json["elapsed_sec"] = float(perf_counter() - t0)
    logs_dir = Path("logs") / f"logs-{startTime}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(results_json, path=logs_dir / "results.json")
    log_step(f"Saved metrics JSON to {logs_dir / 'results.json'} (elapsed_sec={results_json['elapsed_sec']:.3f}).", "INFO")

if __name__ == "__main__":
    main()
