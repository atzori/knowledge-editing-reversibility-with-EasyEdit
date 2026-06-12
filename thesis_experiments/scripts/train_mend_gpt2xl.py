#!/usr/bin/env python
"""
Train MEND for GPT-2XL on ZsRE and save the checkpoint to:
  thesis_experiments/checkpoints/mend/gpt2-xl

Dataset required (download from https://rome.baulab.info/data/dsets/):
  data/zsre/zsre_mend_train.json
  data/zsre/zsre_mend_eval.json
"""
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from easyeditor.trainer import EditTrainer
from easyeditor.trainer.training_hparams import MENDTrainingHparams
from easyeditor.dataset.zsre import ZsreDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

HPARAMS_PATH = os.path.join(REPO_ROOT, "thesis_experiments", "configs", "hparams_mend_gpt2xl.yaml")
TRAIN_DATA   = os.path.join(REPO_ROOT, "data", "zsre", "zsre_mend_train.json")
EVAL_DATA    = os.path.join(REPO_ROOT, "data", "zsre", "zsre_mend_eval.json")
CHECKPOINT   = os.path.join(REPO_ROOT, "thesis_experiments", "checkpoints", "mend", "gpt2-xl")


def main():
    hparams = MENDTrainingHparams.from_hparams(HPARAMS_PATH)
    hparams.eval_only = False  # enable training
    hparams.archive   = None   # train from scratch

    for path in (TRAIN_DATA, EVAL_DATA):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Download ZsRE from https://rome.baulab.info/data/dsets/ "
                "and place zsre_mend_train.json / zsre_mend_eval.json under data/zsre/."
            )

    LOG.info("Loading training set …")
    train_set = ZsreDataset(TRAIN_DATA, config=hparams)
    LOG.info("Loading validation set …")
    val_set   = ZsreDataset(EVAL_DATA,  config=hparams)

    trainer = EditTrainer(hparams, train_set, val_set)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    trainer.save_path = CHECKPOINT
    LOG.info(f"Checkpoint will be saved to: {CHECKPOINT}")

    trainer.run()


if __name__ == "__main__":
    main()
