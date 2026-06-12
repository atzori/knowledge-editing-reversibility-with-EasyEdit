from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from thesis_experiments.scripts.utils_io import log_step


_COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

# Methods that apply multiple edits per call (sliding-window batching).
# Keep in sync with _BATCH_METHODS in run_edit_and_rollback_engine_batch.py.
BATCH_METHODS: frozenset = frozenset({"memit"})


def _repo_root() -> Path:
    # this file lives at thesis_experiments/scripts/ -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def load_counterfact(cf_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Path]:
    """Load (or download) the CounterFact dataset JSON."""
    import torch

    if cf_path is None:
        cf_path = _repo_root() / "thesis_experiments" / "data" / "counterfact" / "counterfact.json"

    if not cf_path.exists():
        log_step(f"Downloading CounterFact from {_COUNTERFACT_URL}", "INFO")
        cf_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(_COUNTERFACT_URL, str(cf_path))

    with cf_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"CounterFact must be a JSON list. Got: {type(records)}")
    return records, cf_path


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------

def _fmt_prompt(prompt: str, subject: str) -> str:
    p, s = (prompt or "").strip(), (subject or "").strip()
    if not p:
        return p
    if "{}" in p:
        try:
            return p.format(s)
        except Exception:
            return p.replace("{}", s)
    return p.replace("<SUBJECT>", s) if "<SUBJECT>" in p else p


def _extract_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("str", "text", "name"):
            if k in x and isinstance(x[k], str):
                return x[k]
    return str(x) if x is not None else ""


def _extract_prompts(value: Any, subject: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("prompts", value.get("prompt"))
    items = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(_fmt_prompt(item, subject))
        elif isinstance(item, dict):
            p = item.get("prompt", item.get("text", item.get("template", "")))
            if p:
                out.append(_fmt_prompt(str(p), subject))
    return [p for p in out if p.strip()]


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw CounterFact record to the canonical engine format."""
    rr = rec.get("requested_rewrite", rec)
    subject = str(rr.get("subject", rec.get("subject", "")) or "").strip()
    prompt = _fmt_prompt(str(rr.get("prompt", rec.get("prompt", "")) or "").strip(), subject)
    gt = _extract_str(rr.get("target_true", rr.get("ground_truth", rec.get("ground_truth", ""))))
    tn = _extract_str(rr.get("target_new", rr.get("target", rec.get("target_new", ""))))
    return {
        "case_id": rec.get("case_id", rec.get("id", "")),
        "prompt": prompt,
        "subject": subject,
        "ground_truth": gt,
        "target_new": tn,
        "locality_prompts": _extract_prompts(
            rec.get("neighborhood_prompts") or rec.get("neighborhood")
            or rec.get("locality_prompts") or rec.get("locality"),
            subject,
        ),
        "portability_prompts": _extract_prompts(
            rec.get("paraphrase_prompts") or rec.get("paraphrases")
            or rec.get("portability_prompts") or rec.get("portability"),
            subject,
        ),
    }


# ---------------------------------------------------------------------------
# Index batching
# ---------------------------------------------------------------------------

def iter_index_batches(
    *,
    total_n: int,
    start_idx: int,
    n_samples: int,
    method: str,
    batch_size: int = 1,
) -> Generator[List[int], None, None]:
    """Yield index batches for the experiment loop.

    BATCH_METHODS (e.g. MEMIT) use a sliding window of size `batch_size` with
    step 1, where `n_samples` limits the number of start positions.
    All other methods yield single-element batches.
    """
    end_idx = min(total_n, start_idx + n_samples)
    if method in BATCH_METHODS:
        for i in range(end_idx - start_idx):
            win_start = start_idx + i
            win_end = win_start + batch_size
            if win_end > total_n:
                break
            yield list(range(win_start, win_end))
    else:
        for idx in range(start_idx, end_idx):
            yield [idx]
