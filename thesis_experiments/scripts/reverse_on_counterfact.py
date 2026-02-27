#!/usr/bin/env python3
"""
Run edit+rollback on a percentage of CounterFact samples defined in a YAML config.

Usage:
  python reverse_on_counterfact.py --config thesis_experiments/configs/exp_gpt2xl_rome.yaml --mode both
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Engine: must provide run_edit_and_rollback_engine(config_path, mode="both", hparams=None)
from thesis_experiments.scripts import run_edit_and_rollback_engine as engine

# Reuse the same hparams loader used in your project.
from ke_core import load_hparams


def _read_yaml(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {path}")
    return cfg


def _normalize_fraction(x: Any) -> float:
    """
    Accepts:
      - 0..1  (fraction)
      - 0..100 (percent)
    Returns:
      - 0..1
    """
    if x is None:
        raise ValueError("Missing percentage field in YAML config.")
    f = float(x)
    if f <= 0:
        raise ValueError(f"Percentage/fraction must be > 0. Got: {x}")
    if f <= 1.0:
        return f
    if f <= 100.0:
        return f / 100.0
    raise ValueError(f"Percentage/fraction looks invalid (>100). Got: {x}")


def _load_counterfact_len(cfg: Dict[str, Any], cfg_path: Path) -> int:
    """
    Minimal loader to get dataset length to compute k.
    Mirrors the same priority rules your engine uses (local first, HF fallback).
    """
    dataset_type = str(cfg.get("exp_dataset_type", "")).strip().lower()
    if dataset_type != "counterfact":
        raise ValueError(f"This runner expects exp_dataset_type=counterfact. Got: {dataset_type}")

    local_path_raw = cfg.get("exp_local_dataset", None)
    allow_hf_fallback = bool(cfg.get("exp_allow_hf_fallback", True))

    # Local JSON list
    if local_path_raw:
        p = Path(str(local_path_raw)).expanduser()
        if not p.is_absolute():
            # Keep your existing behavior: paths are relative to project root
            p = (cfg_path.parent.parent.parent / p).resolve()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Local CounterFact must be a JSON list: {p}")
            return len(data)
        if not allow_hf_fallback:
            raise FileNotFoundError(
                f"Local CounterFact dataset not found at: {p} and HF fallback disabled."
            )

    # HF fallback (only for length)
    if not allow_hf_fallback:
        raise RuntimeError("HF fallback disabled and no valid local dataset provided.")

    from datasets import load_dataset  # local import to avoid cost if unused

    hf_dataset = cfg.get("hf_dataset", None)
    hf_split = cfg.get("hf_split", "test")
    hf_subset = cfg.get("hf_subset", None)
    if not hf_dataset:
        raise ValueError("HF fallback requested but 'hf_dataset' missing in YAML config.")

    ds = load_dataset(hf_dataset, hf_subset, split=hf_split)
    return len(ds)


def _write_tmp_cfg(base_cfg: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")


@contextlib.contextmanager
def suppress_output(enabled: bool):
    """Redirect stdout/stderr to /dev/null when enabled."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--mode",
        default="both",
        choices=["forward", "inverse", "both"],
        help="Override mode (forward|inverse|both).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _read_yaml(cfg_path)

    # Determine how many samples to run
    percent_key = str(cfg.get("exp_counterfact_percent_key", "exp_counterfact_percent")).strip()
    frac = _normalize_fraction(cfg.get(percent_key, 0.5))
    total_n = _load_counterfact_len(cfg, cfg_path)
    k = int(math.floor(total_n * frac))
    if k <= 0:
        raise ValueError(f"Computed k={k}. Check {percent_key} and dataset size={total_n}.")

    start_idx = int(cfg.get("exp_reverse_start_index", 0))
    if start_idx < 0:
        raise ValueError(f"exp_reverse_start_index must be >= 0. Got: {start_idx}")
    end_idx = min(total_n, start_idx + k)

    # Load hparams ONCE and reuse
    method = str(cfg.get("exp_method", "rome")).strip().lower()
    exp_hparams_path = str(cfg.get("exp_hparams_path", "")).strip()
    if not exp_hparams_path:
        raise ValueError("Missing exp_hparams_path in config (needed to load hparams once).")

    hp_path = Path(exp_hparams_path).expanduser()
    if not hp_path.is_absolute():
        hp_path = (cfg_path.parent.parent.parent / hp_path).resolve()
    if not hp_path.exists():
        raise FileNotFoundError(f"Hparams file not found: {hp_path}")

    hparams_obj = load_hparams(method, str(hp_path))

    # Output setup
    out_path_raw = str(cfg.get("exp_reverse_out_path", "logs/run_percent_counterfact_results.jsonl")).strip()
    if not out_path_raw:
        raise ValueError("exp_reverse_out_path in YAML is empty.")
    out_path = Path(out_path_raw).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare a temp YAML path (we overwrite it each iteration)
    tmp_cfg_path = out_path.parent / f".tmp_{cfg_path.stem}_iter.yaml"

    # Force one edit per call unless you explicitly want batch/memit behavior
    base_cfg = deepcopy(cfg)
    base_cfg["exp_num_edits"] = 1

    # Suppression flag from YAML (required by your request)
    suppress = bool(cfg.get("exp_suppress_internal_prints", False))

    total_iters = max(end_idx - start_idx, 0)
    iterator = range(start_idx, end_idx)

    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=total_iters,
            desc="CounterFact",
            unit="case",
            dynamic_ncols=True,
        )

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in iterator:
            iter_cfg = deepcopy(base_cfg)
            iter_cfg["exp_sample_index"] = int(i)

            _write_tmp_cfg(iter_cfg, tmp_cfg_path)

            with suppress_output(suppress):
                result = engine.run_edit_and_rollback_engine(
                    config_path=str(tmp_cfg_path),
                    mode=args.mode or "both",
                    hparams=hparams_obj,
                )

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            # Optional: add a small postfix to tqdm if available
            if tqdm is not None and hasattr(iterator, "set_postfix"):
                try:
                    ppl0 = result.get("ppl", {}).get("M0", None)
                    ppl1 = result.get("ppl", {}).get("M1", None)
                    iterator.set_postfix(ppl0=ppl0, ppl1=ppl1)
                except Exception:
                    pass

    # Cleanup temp cfg (optional)
    try:
        tmp_cfg_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Done. Wrote {end_idx - start_idx} runs to: {out_path}")


if __name__ == "__main__":
    main()