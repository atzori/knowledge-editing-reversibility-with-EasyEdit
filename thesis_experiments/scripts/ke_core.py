import io
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams


# ----------------------------
# Keep console clean
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# Transformers sometimes emits deprecation notices via its own logger, not `warnings`.
# This helps keep experiment logs readable.
hf_logging.set_verbosity_error()


def load_hparams(method: str, hparams_path: str):
    method = method.lower()
    if method == "rome":
        return ROMEHyperParams.from_hparams(hparams_path)
    if method == "memit":
        return MEMITHyperParams.from_hparams(hparams_path)
    raise ValueError("Unsupported method. Use 'rome' or 'memit'.")


def get_tokenizer(editor: BaseEditor, model_name: str):
    tok = getattr(editor, "tok", None)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def apply_edit(
    editor: BaseEditor,
    *,
    prompt: str,
    subject: str,
    ground_truth: str,
    target_new: str,
    verbose: bool,
    suppress_internal_prints: bool,
    eval_metric: str = None,
    enable_portability_metrics: bool = False,
    locality_prompts: List = None,
    portability_prompts: List = None,
):
    """
    Persistent edit by design: sequential_edit=True.

    EasyEdit expects:
      - rephrase_prompts: List[str] (paraphrases for the rewrite prompt)
      - locality_inputs: Dict[str, Dict[str, List[List[str]] or List[str]]]
        where outer list length == num_edits (here: 1)
      - portability_inputs: similar structure (optional)

    We provide a minimal nested structure:
      locality_inputs["neighborhood"]["prompt"] = [ [p1, p2, ...] ]
      locality_inputs["neighborhood"]["ground_truth"] = [ [gt1, gt2, ...] ]
    """
    kwargs: Dict[str, Any] = dict(
        prompts=[prompt],
        ground_truth=[ground_truth],
        target_new=[target_new],
        subject=[subject],  # accepted via **kwargs in EasyEdit
        sequential_edit=True,
        verbose=verbose,
        test_generation=False,
    )
    if eval_metric:
        kwargs["eval_metric"] = eval_metric

    # ----------------------------
    # Portability -> rephrase_prompts (+ optional portability_inputs)
    # ----------------------------
    if portability_prompts:
        rp: List[str] = []
        for x in portability_prompts:
            if isinstance(x, dict):
                rp.append(str(x.get("prompt", "")))
            else:
                rp.append(str(x))
        rp = [p for p in rp if p.strip()]

        if rp:
            kwargs["rephrase_prompts"] = rp
            if enable_portability_metrics:
                kwargs["portability_inputs"] = {
                    "rephrase": {
                        "prompt": [rp],
                        "ground_truth": [[target_new] * len(rp)],
                    }
                }

    # ----------------------------
    # Locality -> locality_inputs (nested lists: one entry per edit)
    # ----------------------------
    if locality_prompts:
        lp: List[str] = []
        gt_lp: List[str] = []

        for x in locality_prompts:
            if isinstance(x, dict):
                lp.append(str(x.get("prompt", "")))
                # Try common keys if present
                gt_lp.append(str(x.get("ground_truth", x.get("answer", x.get("target", "")))))
            else:
                lp.append(str(x))
                gt_lp.append(str(ground_truth))

        lp = [p for p in lp if p.strip()]

        # IMPORTANT: outer list length must match num_edits (=1)
        kwargs["locality_inputs"] = {
            "neighborhood": {
                "prompt": [lp],
                "ground_truth": [gt_lp],
            }
        }

    # ----------------------------
    # Execute
    # ----------------------------
    if suppress_internal_prints:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            metrics, edited_model, _ = editor.edit(**kwargs)
        return metrics, edited_model

    metrics, edited_model, _ = editor.edit(**kwargs)
    return metrics, edited_model


def force_hf_home():
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = hf_cache
