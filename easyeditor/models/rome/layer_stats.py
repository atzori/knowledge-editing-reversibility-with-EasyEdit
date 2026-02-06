import os
import sys
from pathlib import Path
from contextlib import nullcontext

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Load or compute cached ROME stats (e.g., mom2).
    Adds an optional progress bar (tqdm-like factory) shown only during computation.
    """

    def get_ds():
        # raw_ds = load_dataset(ds_name, subset_name)
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )

        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
            if hasattr(model.config, "sliding_window") and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096

        if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens

        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once

    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
        if hasattr(model.config, "sliding_window") and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096

    if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
        npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens

    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = f"_t{batch_tokens}" + size_suffix

    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/"
        f"{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    )
    filename = stats_dir / file_extension
    filename.parent.mkdir(parents=True, exist_ok=True)
    print("\n[ROME STATS CHECK]", flush=True)
    print(f"[ROME STATS CHECK] stats_dir base : {stats_dir}", flush=True)
    print(f"[ROME STATS CHECK] cache file    : {filename}", flush=True)
    print(f"[ROME STATS CHECK] exists         : {filename.exists()}", flush=True)

    # Decide whether we compute or load from cache
    # - compute if force_recompute OR cache file doesn't exist
    # - otherwise load from cache (ds=None)
    will_compute = force_recompute or (not filename.exists())
    print(
        f"[ROME STATS CHECK] mode           : "
        f"{'COMPUTE (rebuild stats)' if will_compute else 'CACHE (load existing)'}",
        flush=True
    )
    ds = get_ds() if will_compute else None

    # Cache handling:
    # - if recompute -> do not use cache during tally; tally will compute and write new file
    # - if not recompute -> allow tally to load from cache
    cache_path = None if force_recompute else filename

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=cache_path,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )

    # Compute total number of batches for progress
    if sample_size is not None:
        n_items = sample_size
    else:
        n_items = len(ds) if ds is not None else None

    batch_count = (n_items + batch_size - 1) // batch_size if n_items is not None else None

    # Resolve device safely
    if hparams is not None and hasattr(hparams, "device") and hparams.device is not None:
        # Convention: -1 means CPU
        dev = "cpu" if str(hparams.device) == "-1" else f"cuda:{hparams.device}"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Progress bar: show only when we are actually computing
    # Use `progress` factory if provided; otherwise, no progress output.
    pbar_cm = nullcontext()
    if will_compute and progress is not None:
        pbar_cm = progress(
            total=batch_count,
            desc="Computing ROME mom2",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
            mininterval=0.5,
        )

    print(f"[STATS] stats_path: {filename}", flush=True)
    print(f"[STATS] mode: {'COMPUTE' if will_compute else 'CACHE'}", flush=True)
    print("[STATS] loader ready, entering loop...", flush=True)

    with torch.no_grad():
        with pbar_cm as pbar:
            for batch_group in loader:
                for batch in batch_group:
                    batch = dict_to_(batch, dev)
                    with Trace(
                        model,
                        layer_name,
                        retain_input=True,
                        retain_output=False,
                        stop=True,
                    ) as tr:
                        model(**batch)

                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)

                if pbar is not None:
                    pbar.update(1)

    return stat


if __name__ == "__main__":
    main()
