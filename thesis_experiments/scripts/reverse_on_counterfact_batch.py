from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from thesis_experiments.scripts import run_edit_and_rollback_engine_batch as engine
from thesis_experiments.scripts.counterfact_io import (
    BATCH_METHODS,
    iter_index_batches,
    load_counterfact,
    normalize_record,
)
from ke_core import load_hparams


@contextlib.contextmanager
def _suppress(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as nul, \
         contextlib.redirect_stdout(nul), \
         contextlib.redirect_stderr(nul):
        yield


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run edit-and-rollback on CounterFact.")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config.")
    ap.add_argument("--mode", default="both", choices=["forward", "both"])
    ap.add_argument("--alg", default=None, help="Override algorithm (rome, memit, mend, ...).")
    ap.add_argument("--ppl-start", type=float, default=None,
                    help="Precomputed M0 PPL — skips baseline computation.")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    method = str(args.alg or cfg.get("alg", cfg.get("exp_method", "rome"))).strip().lower()

    records, cf_path = load_counterfact()
    engine.log_step(f"CounterFact: {len(records)} records | {cf_path}", "INFO")

    start_idx = int(cfg.get("exp_reverse_start_index", 0))
    n_samples = int(cfg.get("exp_counterfact_n_samples", len(records) - start_idx))
    batch_size = int(cfg.get("exp_num_edits", 1)) if method in BATCH_METHODS else 1

    hp_raw = str(cfg.get("exp_hparams_path", "")).strip()
    if not hp_raw:
        raise ValueError("Missing 'exp_hparams_path' in config.")
    hp_path = Path(hp_raw).expanduser()
    if not hp_path.is_absolute():
        hp_path = (_repo_root() / hp_path).resolve()
    if not hp_path.exists():
        raise FileNotFoundError(f"Hparams file not found: {hp_path}")

    hparams_obj = load_hparams(method, str(hp_path))

    out_path = Path(str(cfg.get("exp_reverse_out_path", "logs/results.jsonl"))).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quiet = bool(cfg.get("exp_suppress_internal_prints", False))

    batches = list(iter_index_batches(
        total_n=len(records), start_idx=start_idx, n_samples=n_samples,
        method=method, batch_size=batch_size,
    ))
    if not batches:
        raise RuntimeError(
            f"No batches to run (method={method}, start={start_idx}, "
            f"n_samples={n_samples}, batch_size={batch_size}, total={len(records)})."
        )
    engine.log_step(
        f"method={method} | start={start_idx} | n_samples={n_samples} | "
        f"batch_size={batch_size} | n_batches={len(batches)}",
        "INFO",
    )

    it = (
        tqdm(batches, desc=method.upper(), unit="batch", dynamic_ncols=True)
        if tqdm is not None else batches
    )

    ppl_m0 = args.ppl_start  # computed once on first iteration, reused afterwards

    with out_path.open("w", encoding="utf-8") as f_out:
        for idx_batch in it:
            samples, skip = [], False
            for idx in idx_batch:
                s = normalize_record(records[idx])
                if not s["prompt"].strip():
                    engine.log_step(f"[SKIP] Empty prompt at idx={idx}.", "WARNING")
                    skip = True
                    break
                samples.append(s)
            if skip:
                continue

            with _suppress(quiet):
                result = engine.run_edit_and_rollback_engine(
                    config_path=str(cfg_path),
                    mode=args.mode,
                    hparams=hparams_obj,
                    samples=samples,
                    indices=idx_batch,
                    alg=method,
                    ppl_start=ppl_m0,
                )

            if ppl_m0 is None:
                ppl_m0 = result.get("ppl", {}).get("M0")
                if ppl_m0 is not None:
                    engine.log_step(f"M0 PPL computed: {ppl_m0:.4f} — reusing for remaining iterations.", "INFO")

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            if tqdm is not None and hasattr(it, "set_postfix"):
                try:
                    it.set_postfix(
                        ppl0=result.get("ppl", {}).get("M0"),
                        ppl1=result.get("ppl", {}).get("M1"),
                    )
                except Exception:
                    pass

    print(f"Done. Results written to: {out_path}")


if __name__ == "__main__":
    main()
