"""
Butterfly Effect utilities based on perplexity monitoring.

This module is designed to be plugged into single-edit or multi-edit pipelines.
It follows the same core idea as Collapse-in-Model-Editing: measure model-wide
degradation via perplexity on a neutral text set (not the edited prompt). :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm


@dataclass
class ButterflyPPLReport:
    """Compact butterfly-effect report computed from PPL before/after an edit."""
    ppl_before: float
    ppl_after: float
    ppl_delta_abs: float
    ppl_delta_rel: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ppl_before": float(self.ppl_before),
            "ppl_after": float(self.ppl_after),
            "ppl_delta_abs": float(self.ppl_delta_abs),
            "ppl_delta_rel": float(self.ppl_delta_rel),
        }


def load_ppl_texts_from_json(
    path: str | Path,
    text_key: str = "Text",
    max_items: Optional[int] = None,
) -> List[str]:
    """
    Load a list of evaluation texts from a JSON file.

    Expected format (like ME-PPL_50.json): a JSON list of objects, each having `text_key`,
    typically "Text". :contentReference[oaicite:2]{index=2}
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at: {p}. Got: {type(data)}")

    texts: List[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get(text_key, None), str):
            t = item[text_key].strip()
            if t:
                texts.append(t)
        elif isinstance(item, str):
            t = item.strip()
            if t:
                texts.append(t)

        if max_items is not None and len(texts) >= max_items:
            break

    if not texts:
        raise ValueError(f"No texts loaded from: {p} (text_key='{text_key}').")

    return texts


def ensure_pad_token(tokenizer) -> None:
    """
    Ensure tokenizer has a pad token for batched perplexity computation.
    Mirrors the idea used in the reference ppl implementation. :contentReference[oaicite:3]{index=3}
    """
    if tokenizer.pad_token is not None:
        return

    # Use any existing special token as pad token (common workaround for GPT-style tokenizers).
    existing_special_tokens = list(getattr(tokenizer, "special_tokens_map_extended", {}).values())
    if not existing_special_tokens:
        raise ValueError(
            "Tokenizer has no pad_token and no special tokens available. "
            "Cannot safely run batched PPL."
        )
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})


@torch.no_grad()
def compute_ppl(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 16,
    add_start_token: bool = False,
    max_length: Optional[int] = None,
) -> Tuple[float, List[float]]:
    """
    Compute global perplexity over `texts` and return (global_ppl, []).

    Notes:
    - Uses token-level cross-entropy with shifting (next-token prediction).
    - Applies attention-mask to ignore padding tokens.
    - Global PPL is computed as exp(total_nll / total_valid_tokens), i.e.
      the standard LM definition exp(mean token-level NLL).
    - The second returned value is kept as an empty list only for backward
      compatibility with existing call-sites that unpack two outputs.
    """
    if not texts:
        raise ValueError("texts is empty.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. Got: {batch_size}")

    model.eval()
    # NOTE: model.to(device) moved outside for efficiency.

    # Padding is required when batch_size > 1.
    if batch_size > 1:
        ensure_pad_token(tokenizer)

    # If we prepend a BOS token, we must reduce max_length accordingly for the original content.
    max_tokenized_len = None
    if max_length is not None:
        if max_length <= 2:
            raise ValueError(f"max_length too small. Got: {max_length}")
        max_tokenized_len = max_length - 1 if add_start_token else max_length

    loss_fct = CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_tokens = 0

    # Iterate in mini-batches and move only the active batch to GPU.
    n = len(texts)
    num_batches = (n + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches),
        desc="Computing PPL",
        unit="batch",
        leave=True,
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)

        # Tokenize with padding/truncation as needed. We disable special tokens here to have more control,
        enc = tokenizer(
            texts[start:end],
            add_special_tokens=False,
            padding=True,
            truncation=(max_tokenized_len is not None),
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch_ids = enc["input_ids"].to(device)
        batch_attn = enc["attention_mask"].to(device)


        if add_start_token:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is None:
                raise ValueError("add_start_token=True but tokenizer.bos_token_id is None.")
            bos = torch.full((batch_ids.size(0), 1), bos_id, dtype=batch_ids.dtype, device=device)
            batch_ids = torch.cat([bos, batch_ids], dim=1)
            batch_attn = torch.cat([torch.ones_like(bos, dtype=batch_attn.dtype), batch_attn], dim=1)

        # Disable KV cache: it's useful for autoregressive generation, not for PPL.
        out = model(batch_ids, attention_mask=batch_attn, use_cache=False)
        logits = out.logits  # [B, T, V]

        # Shift for next-token prediction
        logits_device = logits.device
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_ids[..., 1:].contiguous().to(logits_device)
        shift_attn = batch_attn[..., 1:].contiguous().to(logits_device)

        # Token-wise NLL: [B, T-1]
        token_nll = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        # Mask padding tokens
        token_nll = token_nll * shift_attn

        total_nll += float(token_nll.sum().item())
        total_tokens += int(shift_attn.sum().item())

    if total_tokens <= 0:
        raise ValueError("No valid (non-padding) tokens found for PPL computation.")

    mean_nll = total_nll / float(total_tokens)
    global_ppl = float(math.exp(mean_nll))
    return global_ppl, []


def butterfly_report(ppl_before: float, ppl_after: float) -> ButterflyPPLReport:
    """Build a ButterflyPPLReport with absolute and relative PPL changes."""
    if ppl_before < 0 or ppl_after < 0:
        raise ValueError("Perplexity values must be non-negative.")

    delta_abs = float(ppl_after - ppl_before)
    delta_rel = float(ppl_after / ppl_before) if ppl_before > 0 else float("inf")

    return ButterflyPPLReport(
        ppl_before=float(ppl_before),
        ppl_after=float(ppl_after),
        ppl_delta_abs=delta_abs,
        ppl_delta_rel=delta_rel,
    )


def is_collapse(
    report: ButterflyPPLReport,
    rel_threshold: float = 1.5,
    abs_threshold: Optional[float] = None,
) -> bool:
    """
    Heuristic collapse detector.

    - rel_threshold: triggers if ppl_after / ppl_before >= rel_threshold
    - abs_threshold: triggers if ppl_after >= abs_threshold (optional)

    This is intentionally simple; thresholds are experiment-dependent.
    """
    if rel_threshold <= 0:
        raise ValueError(f"rel_threshold must be > 0. Got: {rel_threshold}")

    if report.ppl_delta_rel >= rel_threshold:
        return True

    if abs_threshold is not None:
        if abs_threshold <= 0:
            raise ValueError(f"abs_threshold must be > 0 when provided. Got: {abs_threshold}")
        if report.ppl_after >= abs_threshold:
            return True

    return False
