from time import perf_counter
from datetime import datetime
import json
import sys
import inspect
from pathlib import Path
import argparse
import yaml
import warnings
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
import json


import torch

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

from thesis_experiments.scripts.utils_io import (
    print_log,
    _resolve_path_like_cfg,
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
startTime = str(datetime.now().isoformat()).replace(":", "")
set_start_time(startTime)

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
        local_path = _resolve_path_like_cfg(cfg_path, local_path_raw)

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
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise_path_error("Experiment config file", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {cfg_path}")

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

    exp_hparams_path = _resolve_path_like_cfg(cfg_path, exp_hparams_path_raw)
    if not exp_hparams_path.exists():
        raise_path_error("Hparams config file", exp_hparams_path)

    hparams_cfg = yaml.safe_load(exp_hparams_path.read_text(encoding="utf-8")) or {}
    if not isinstance(hparams_cfg, dict):
        raise ValueError(f"Invalid hparams YAML structure (expected mapping) in: {exp_hparams_path}")
    if "alg_name" not in hparams_cfg:
        raise ValueError(f"Hparams YAML must contain 'alg_name'. Missing in: {exp_hparams_path}")

    # ----------------------------
    # Sample selection + runner params
    # ----------------------------
    sample_index = _as_int(cfg.get("exp_sample_index", 0), 0)
    forced_model_name = str(cfg.get("exp_forced_model_name", "gpt2-xl")).strip()
    max_new_tokens = _as_int(cfg.get("exp_max_new_tokens", 20), 20)
    temperature = float(cfg.get("exp_temperature", 1.0) or 1.0)
    do_sample = _as_bool(cfg.get("exp_do_sample", False), False)
    suppress_internal = _as_bool(cfg.get("exp_suppress_internal_prints", True), True)
    verbose = _as_bool(cfg.get("exp_verbose", False), False)

    if max_new_tokens <= 0:
        raise ValueError(f"exp_max_new_tokens must be > 0. Got: {max_new_tokens}")

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
        raw_records, dataset_source = _load_records(cfg_path, cfg)

    if not raw_records:
        raise RuntimeError("No records loaded from dataset (empty list).")
    if sample_index < 0 or sample_index >= len(raw_records):
        raise IndexError(f"exp_sample_index={sample_index} out of range. Dataset size={len(raw_records)}")

    rec_raw = raw_records[sample_index]
    rec = _normalize_record(rec_raw, dataset_type)

    class _Sample:
        def __init__(self, d: Dict[str, Any]):
            self.case_id = d.get("case_id", "")
            self.prompt = d["prompt"]
            self.subject = d["subject"]
            self.ground_truth = d["ground_truth"]
            self.target_new = d["target_new"]
            self.locality_prompts = d.get("locality_prompts", [])
            self.portability_prompts = d.get("portability_prompts", [])

    sample = _Sample(rec)
    sample_prompt = str(sample.prompt).rstrip()

    # ----------------------------
    # Load hparams + build editor
    # ----------------------------
    log_step("Loading hparams + building editor (this can take time).")
    log_step(f"Using hparams file: {exp_hparams_path}")

    force_hf_home()
    hparams = load_hparams(method, str(exp_hparams_path))

    # Force model_name to avoid local-cache weirdness
    hparams.model_name = forced_model_name

    # Avoid custom paths that may conflict with your environment
    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(hparams, attr):
            setattr(hparams, attr, None)

    # Ensure stats_dir exists if present in hparams (ROME uses it)
    if hasattr(hparams, "stats_dir"):
        stats_dir = getattr(hparams, "stats_dir")
        if stats_dir in (None, "", "null"):
            raise ValueError("stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML.")
        Path(str(stats_dir)).mkdir(parents=True, exist_ok=True)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_step(f"Selected device: {device}")

    # Move baseline model once (time-consuming + can OOM)
    try:
        log_step("Moving baseline model to device (may take time).")
        editor.model.to(device)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM while moving baseline model to device.", "ERROR")
            return
        raise

    ppl_texts: List[str] = []
    ppl_m0: Optional[float] = None

    if be_enabled:
        if not be_ppl_data_path_raw:
            log_step("BE enabled but 'be_ppl_data_path' is missing in config YAML.", "ERROR")
            raise ValueError("be_enabled=True but 'be_ppl_data_path' is missing in config YAML.")

        be_ppl_data_path = _resolve_path_like_cfg(cfg_path, be_ppl_data_path_raw)
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
    print("\n=== SAMPLE ===")
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
    print(f"sample_index: {sample_index}")
    print(f"case_id: {sample.case_id}")
    print(f"prompt: {sample_prompt}")
    print(f"subject: {sample.subject}")
    print(f"ground_truth: {sample.ground_truth}")
    print(f"target_new: {sample.target_new}")
    print(f"locality_prompts: {len(sample.locality_prompts)}")
    print(f"portability_prompts: {len(sample.portability_prompts)}\n")

    # ----------------------------
    # Populate results metadata (for results.json)
    # ----------------------------
    results_json.update({
        "method": method,
        "dataset_type": dataset_type,
        "dataset_source": dataset_source,
        "dataset_size_loaded": len(raw_records),
        "sample_index": sample_index,
        "case_id": sample.case_id,
        "prompt": sample_prompt,
        "subject": sample.subject,
        "ground_truth": sample.ground_truth,
        "target_new": sample.target_new,
    })
    results_json["counts"]["locality_prompts"] = len(sample.locality_prompts)
    results_json["counts"]["portability_prompts"] = len(sample.portability_prompts)


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
        log_step("Behavioral probe (M0).")
        comp0 = generate_completion(
            editor.model,
            tok,
            sample_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print("\n=== BEHAVIORAL (M0) ===")
        print(comp0)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            log_step("Aborted: CUDA OOM during behavioral probe (M0).", "ERROR")
            return
        raise

    metrics_fwd = None
    metrics_inv = None
    edited_model = None

    # ----------------------------
    # Forward edit
    # ----------------------------
    if args.mode in ("forward", "both"):
        log_step("Applying FORWARD edit (GT -> NEW) (time-consuming).")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        try:
            metrics_fwd, edited_model = _call_apply_edit(
                editor,
                prompt=sample_prompt,
                subject=sample.subject,
                ground_truth=sample.ground_truth,
                target_new=sample.target_new,
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

        # Ensure edited model is on the right device for PPL / generation
        try:
            log_step("Moving edited model to device (may take time).")
            edited_model.to(device)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM while moving edited model to device.", "ERROR")
                return
            raise

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

                be_fwd = {
                    "stage": "M1_after_forward",
                    **rep.to_dict(),
                    "collapse_rel_threshold": float(be_collapse_rel_threshold),
                    "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                    "is_collapse": bool(collapse),
                }

                # Keep EasyEdit metrics structure untouched if it's a list.
                if isinstance(metrics_fwd, dict):
                    metrics_fwd["butterfly_effect"] = be_fwd
                else:
                    log_step("Forward metrics is a list; BE report kept separate (not injected).", "WARNING")

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
                    "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                }
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while computing PPL on M1.", "ERROR")
                    return
                raise

        print_metrics_table(metrics_fwd, title="FORWARD METRICS")

        # Behavioral probe on edited model
        try:
            log_step("Behavioral probe (M1).")
            comp1 = generate_completion(
                edited_model,
                tok,
                sample_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            print("\n=== BEHAVIORAL (M1) ===")
            print(comp1)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during behavioral probe (M1).", "ERROR")
                return
            raise

    # ----------------------------
    # Inverse only
    # ----------------------------
    if args.mode == "inverse":
        log_step("Applying INVERSE edit only (NEW -> GT) on M0 (time-consuming).")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        try:
            metrics_inv, inv_model = _call_apply_edit(
                editor,
                prompt=sample_prompt,
                subject=sample.subject,
                ground_truth=sample.target_new,
                target_new=sample.ground_truth,
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

        try:
            log_step("Behavioral probe (after inverse-only).")
            comp2 = generate_completion(
                inv_model,
                tok,
                sample_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            print("\n=== BEHAVIORAL (after inverse-only) ===")
            print(comp2)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during behavioral probe (inverse-only).", "ERROR")
                return
            raise

    # ----------------------------
    # Both: rollback
    # ----------------------------
    if args.mode == "both":
        if edited_model is None:
            raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

        log_step("Switching editor to edited model for rollback.")
        editor.model = edited_model

        log_step("Applying INVERSE edit (rollback: NEW -> GT) on M1 (time-consuming).")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        try:
            metrics_inv, rollback_model = _call_apply_edit(
                editor,
                prompt=sample_prompt,
                subject=sample.subject,
                ground_truth=sample.target_new,
                target_new=sample.ground_truth,
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

        # Ensure rollback model is on the right device
        try:
            log_step("Moving rollback model to device (may take time).")
            rollback_model.to(device)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM while moving rollback model to device.", "ERROR")
                return
            raise

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

                be_inv = {
                    "stage": "M2_after_rollback",
                    **rep2.to_dict(),
                    "collapse_rel_threshold": float(be_collapse_rel_threshold),
                    "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                    "is_collapse": bool(collapse2),
                }

                if isinstance(metrics_inv, dict):
                    metrics_inv["butterfly_effect"] = be_inv
                else:
                    log_step("Rollback metrics is a list; BE report kept separate (not injected).", "WARNING")

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
                    "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                }
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log_step("Aborted: CUDA OOM while computing PPL on M2.", "ERROR")
                    return
                raise

        print_metrics_table(metrics_inv, title="INVERSE (ROLLBACK) METRICS")

        try:
            log_step("Behavioral probe (M2).")
            comp2 = generate_completion(
                rollback_model,
                tok,
                sample_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            print("\n=== BEHAVIORAL (M2) ===")
            print(comp2)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                log_step("Aborted: CUDA OOM during behavioral probe (M2).", "ERROR")
                return
            raise

    # ----------------------------
    # Finalize + persist results.json ONLY after everything is computed
    # ----------------------------
    results_json["elapsed_sec"] = float(perf_counter() - t0)
    save_results_json( results_json, path="logs/logs-" + startTime+"/results.json")
    log_step(f"Saved metrics JSON to logs/logs-{startTime}/results.json (elapsed_sec={results_json['elapsed_sec']:.3f}).", "INFO")

if __name__ == "__main__":
    main()
