from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Engine module
from thesis_experiments.scripts import run_edit_and_rollback_engine_batch as engine_batch

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
        engine_batch.log_step(f"CounterFact not found at: {cf_path}", "WARNING")
        engine_batch.log_step(f"Downloading CounterFact from: {REMOTE_URL}", "INFO")
        cf_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(REMOTE_URL, str(cf_path))

    engine_batch.log_step(f"Loading CounterFact from: {cf_path}", "INFO")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--mode", default="both", choices=["forward", "both"])
    ap.add_argument("--alg", default=None, choices=["rome", "memit"], help="Override algorithm method.")
    ap.add_argument("--ppl-start", type=float, default=None, help="Optional precomputed M0 PPL to skip baseline PPL computation.")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _read_yaml(cfg_path)

    raw_records, cf_json_path = _load_counterfact_with_local_priority()
    total_n = len(raw_records)
    engine_batch.log_step(f"CounterFact size: {total_n} | Local path: {cf_json_path}", "INFO")

    start_idx = int(cfg.get("exp_reverse_start_index", 0))
    n_samples_key = str(cfg.get("exp_counterfact_n_samples_key", "exp_counterfact_n_samples")).strip() or "exp_counterfact_n_samples"
    n_samples_raw = cfg.get(n_samples_key, None)
    if n_samples_raw is None:
        # Backward-compatible fallback for old configs.
        percent_key = str(cfg.get("exp_counterfact_percent_key", "exp_counterfact_percent")).strip() or "exp_counterfact_percent"
        frac = _normalize_fraction(cfg.get(percent_key, 1.0), default=1.0)
        n_samples = max(0, int(total_n * frac) - start_idx)
        engine_batch.log_step(
            f"'{n_samples_key}' not found. Fallback to '{percent_key}' ({frac:.2%}) -> n_samples={n_samples}.",
            "WARNING",
        )
    else:
        try:
            n_samples = int(n_samples_raw)
        except Exception as e:
            raise ValueError(f"{n_samples_key} must be an integer. Got: {n_samples_raw}") from e
        if n_samples <= 0:
            raise ValueError(f"{n_samples_key} must be > 0. Got: {n_samples}")

    end_idx = min(total_n, start_idx + n_samples)

    method = str(args.alg or cfg.get("alg", cfg.get("exp_method", "rome"))).strip().lower()
    if method not in ("rome", "memit"):
        raise ValueError(f"Invalid alg/method='{method}'. Must be 'rome' or 'memit'.")
    cfg["exp_method"] = method

    engine_batch.log_step(
        f"Running on CounterFact indices: start={start_idx}, end={end_idx} "
        f"(n_samples_requested={n_samples}, n_samples_effective={max(0, end_idx - start_idx)}, "
        f"n_samples_key={n_samples_key}, alg={method})",
        "INFO",
    )

    index_batches: List[List[int]] = []
    if method == "memit":
        exp_num_edits = int(cfg.get("exp_num_edits", 1))
        if exp_num_edits <= 0:
            raise ValueError(f"exp_num_edits must be > 0 for MEMIT. Got: {exp_num_edits}")

        # Sliding window with overlap=1:
        # - exp_counterfact_n_samples controls only the number of START positions.
        # - each window can extend beyond (start_idx + n_samples - 1), up to dataset end.
        # iter i -> [start+i, ..., start+i+exp_num_edits-1]
        i = 0
        while True:
            win_start = start_idx + i
            win_end = win_start + exp_num_edits
            if win_start >= end_idx:
                break
            if win_end > total_n:
                break
            index_batches.append(list(range(win_start, win_end)))
            i += 1

        if not index_batches:
            raise RuntimeError(
                f"No MEMIT windows to run: start={start_idx}, start_range_end={end_idx}, "
                f"exp_num_edits={exp_num_edits}, total={total_n}. "
                "Need at least one valid window fully contained in dataset bounds."
            )
        engine_batch.log_step(
            f"MEMIT sliding windows prepared: n_windows={len(index_batches)}, "
            f"window_size={exp_num_edits}, overlap=1.",
            "INFO",
        )
    else:
        indices = list(_iter_indices(total_n=total_n, start_idx=start_idx, end_idx=end_idx))
        if not indices:
            raise RuntimeError(f"No indices to run (start={start_idx}, end={end_idx}, total={total_n}).")
        index_batches = [[int(i)] for i in indices]
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

    iterator: Iterable[List[int]] = index_batches
    if tqdm is not None:
        iterator = tqdm(index_batches, total=len(index_batches), desc="CounterFact", unit="iter", dynamic_ncols=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for indices in iterator:
            samples: List[Dict[str, Any]] = []
            skip_batch = False
            for idx in indices:
                raw = raw_records[int(idx)]
                sample = normalize_counterfact_record(raw)

                # Guard: prevent empty prompts causing tokenizer/generate crash
                if not str(sample.get("prompt", "")).strip():
                    engine_batch.log_step(
                        f"[SKIP] Empty prompt at index={idx} case_id={sample.get('case_id','')} "
                        f"(alg={method}, batch_start={indices[0]}).",
                        "WARNING",
                    )
                    skip_batch = True
                    break
                samples.append(sample)
            if skip_batch:
                continue

            with suppress_output(suppress):
                result = engine_batch.run_edit_and_rollback_engine(
                    config_path=str(cfg_path),
                    mode=args.mode or "both",
                    hparams=hparams_obj,
                    samples=samples,
                    indices=indices,
                    alg=method,
                    ppl_start=args.ppl_start,
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


if __name__ == "__main__":
    main()
