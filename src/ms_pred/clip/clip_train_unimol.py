"""
Training script for CLIP model with UniMol2 molecular encoder.

Usage:
    python src/ms_pred/clip/clip_train_unimol.py --config configs/clip/train_msg_unimol.yaml
    or via launcher:
    python launcher_scripts/run_from_config.py configs/clip/train_msg_unimol.yaml
"""
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.clip.clip_data_unimol import CLIPDatasetUniMol
from ms_pred.clip.clip_model_unimol import CLIPModelUniMol

import shutil
import os
import numpy as np
import math


def add_train_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_clip_unimol/")
    parser.add_argument("--dataset-name", default="msg")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_msg.tsv")

    parser.add_argument("--batch-size", default=32, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)
    parser.add_argument("--learning-rate", default=3e-4, action="store", type=float)
    parser.add_argument("--weight-decay", default=1e-5, action="store", type=float)
    parser.add_argument("--grad-accumulate", default=1, type=int, action="store")
    parser.add_argument("--num-gpu", default=1, action="store", type=int)
    parser.add_argument("--precision", default=32, type=str, action="store",
                        help="Training precision: 32, 16-mixed, bf16-mixed")

    # UniMol2 params
    parser.add_argument("--unimol-model-size", default="84m", type=str,
                        choices=["84m", "164m", "310m", "570m", "1.1b"])
    parser.add_argument("--frozen-unimol", default=False, action="store_true")
    parser.add_argument("--unfreeze-unimol-epoch", default=0, action="store", type=int)

    # Projection params
    parser.add_argument("--projection-dim", default=512, action="store", type=int)
    parser.add_argument("--temperature", default=0.07, action="store", type=float)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--hidden-size", default=512, action="store", type=int)

    # Spectrum encoder
    parser.add_argument("--spec-ckpt", default="/home/weiwentao/workspace/DreaMS/dreams/models/pretrained/ssl_model.ckpt",
                        type=str, action="store")
    parser.add_argument("--dreams", default=False, action="store_true")
    parser.add_argument("--pooling-strategy", default='cls', action="store", type=str)
    parser.add_argument("--frozen-dreams", default=False, action="store_true")
    parser.add_argument("--frozen-epochs", default=10, action="store", type=int)

    # Training schedule
    parser.add_argument("--lr-decay-rate", default=0.825, action="store", type=float)
    parser.add_argument("--warm-up", default=1000, action="store", type=int)
    parser.add_argument("--patience", default=20, action="store", type=float)

    # Feature flags
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--embed-ce", default=False, action="store_true")
    parser.add_argument("--local-contra", default=False, action="store_true")
    parser.add_argument("--local-weight", default=0.5, type=float, action="store")
    parser.add_argument("--local-threshold", default=0.5, type=float, action="store")
    parser.add_argument("--local-start-epochs", default=0, action="store", type=int)

    parser.add_argument("--decoys", default=False, action="store_true")
    parser.add_argument("--decoys-num", default=5, action="store", type=int)
    parser.add_argument("--decoys-path", default="", action="store", type=str)
    parser.add_argument("--easy-decoys-ratio", default=0.2, action="store", type=float)

    parser.add_argument("--frag-supervised", default=False, action="store_true")
    parser.add_argument("--frag-path", default='', action="store", type=str)
    parser.add_argument("--frag-supervised-weight", default=0.5, type=float, action="store")

    parser.add_argument("--spec-sim-entropy", default=False, action="store_true")
    parser.add_argument("--wo-global", default=False, action="store_true")
    parser.add_argument("--queue-size", default=0, type=int, action="store",
                        help="Size of embedding queue for extra contrastive negatives (0=disabled)")

    parser.add_argument("--random-split", default=False, action="store_true")
    parser.add_argument("--train-val", default=False, action="store_true")
    parser.add_argument("--debug-test", default=False, action="store_true")

    parser.add_argument("--augment", default=False, action="store_true")
    parser.add_argument("--mz-shift-aug-p", default=0.3, action="store", type=float)
    parser.add_argument("--mz-shift-aug-max", default=50, action="store", type=float)

    # Data paths
    parser.add_argument("--magma-dag-folder", default=None, type=str, action="store")
    parser.add_argument("--unimol-cache-dir", default=None, type=str, action="store",
                        help="Directory with precomputed .npz UniMol2 features (from precompute_unimol_features.py)")

    # Checkpoint
    parser.add_argument("--clip-ckpt", default=None, action="store", type=str)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    return parser.parse_args()


def cal_total_steps(epochs, batch_size, num_gpus, dataset_size, accumulate_grad_batches=1):
    effective_batch_size = batch_size * num_gpus * accumulate_grad_batches
    if effective_batch_size == 0:
        raise ValueError("Effective batch size cannot be zero.")
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    return steps_per_epoch * epochs


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="clip_unimol_train.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]

    df = pd.read_csv(labels, sep='\t')
    if kwargs["debug"]:
        df = df[:100]

    spec_names = df["spec"].values
    split_file = data_dir / "splits" / kwargs["split_name"]
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)

    if kwargs["random_split"]:
        all_inds = list(train_inds) + list(val_inds)
        random.shuffle(all_inds)
        n_total = len(all_inds)
        n_train = int(round(n_total * 0.9))
        train_inds = all_inds[:n_train]
        val_inds = all_inds[n_train:]

    if kwargs["train_val"]:
        train_inds = np.concatenate([train_inds, val_inds])

    if kwargs["debug_test"]:
        train_inds = np.concatenate([train_inds, val_inds, test_inds])
        val_inds = np.concatenate([val_inds, test_inds])

    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    # Decoys
    train_decoys_df = None
    val_decoys_df = None
    if kwargs["decoys"]:
        decoys_path = kwargs['decoys_path']
        decoys_df = pd.read_csv(decoys_path, sep='\t')
        if kwargs["debug"]:
            specs = df['spec'].values.tolist()
            decoys_df = decoys_df[decoys_df['spec'].isin(specs)]
        train_specs = train_df['spec'].values.tolist()
        train_decoys_df = decoys_df[decoys_df['spec'].isin(train_specs)]
        val_specs = val_df['spec'].values.tolist()
        val_decoys_df = decoys_df[decoys_df['spec'].isin(val_specs)]

    # MAGMA dag folder (still needed for spectrum loading)
    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    print(f"Found {len(name_to_json)} magma json files in {magma_dag_folder}")

    # Create datasets
    unimol_cache_dir = kwargs.get("unimol_cache_dir", None)
    train_dataset = CLIPDatasetUniMol(
        train_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        decoys=kwargs["decoys"],
        decoys_num=kwargs["decoys_num"],
        decoys_df=train_decoys_df,
        easy_decoys_ratio=kwargs["easy_decoys_ratio"],
        augment=kwargs["augment"],
        mz_shift_aug_max=kwargs["mz_shift_aug_max"],
        mz_shift_aug_p=kwargs["mz_shift_aug_p"],
        emb_ce=kwargs["embed_ce"],
        frag_supervised=kwargs['frag_supervised'],
        frags_path=kwargs['frag_path'],
        unimol_cache_dir=unimol_cache_dir,
    )
    val_dataset = CLIPDatasetUniMol(
        val_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        decoys=kwargs["decoys"],
        decoys_num=kwargs["decoys_num"],
        decoys_df=val_decoys_df,
        easy_decoys_ratio=kwargs["easy_decoys_ratio"],
        emb_ce=kwargs["embed_ce"],
        frag_supervised=kwargs['frag_supervised'],
        frags_path=kwargs['frag_path'],
        unimol_cache_dir=unimol_cache_dir,
    )

    total_steps = cal_total_steps(
        kwargs["max_epochs"], kwargs["batch_size"],
        kwargs["num_gpu"], len(train_dataset)
    )

    persistent_workers = kwargs["num_workers"] > 0
    collate_fn = train_dataset.get_collate_fn()

    train_loader = DataLoader(
        train_dataset,
        num_workers=min(kwargs["num_workers"], kwargs["batch_size"]),
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=min(kwargs["num_workers"], kwargs["batch_size"]),
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )

    if kwargs["clip_ckpt"] is not None:
        print(f"Loading clip ckpt from {kwargs['clip_ckpt']}")
        model = CLIPModelUniMol.load_from_checkpoint(kwargs["clip_ckpt"], strict=False)
    else:
        model = CLIPModelUniMol(
            unimol_model_size=kwargs["unimol_model_size"],
            frozen_unimol=kwargs["frozen_unimol"],
            unfreeze_unimol_epoch=kwargs["unfreeze_unimol_epoch"],
            projection_dim=kwargs["projection_dim"],
            temperature=kwargs["temperature"],
            dropout=kwargs["dropout"],
            spec_ckpt_path=kwargs["spec_ckpt"],
            dreams=kwargs["dreams"],
            pooling_strategy=kwargs["pooling_strategy"],
            frozen_dreams=kwargs["frozen_dreams"],
            unfreeze_epoch=kwargs["frozen_epochs"],
            lr=kwargs["learning_rate"],
            lr_decay_rate=kwargs["lr_decay_rate"],
            warmup=kwargs["warm_up"],
            weight_decay=kwargs["weight_decay"],
            total_steps=total_steps,
            emb_adducts=kwargs["embed_adduct"],
            embed_ce=kwargs["embed_ce"],
            local_contra=kwargs["local_contra"],
            local_weight=kwargs["local_weight"],
            local_threshold=kwargs["local_threshold"],
            local_start_epoch=kwargs["local_start_epochs"],
            decoys=kwargs["decoys"],
            frag_supervised=kwargs["frag_supervised"],
            spec_sim_entropy=kwargs["spec_sim_entropy"],
            wo_global=kwargs["wo_global"],
            queue_size=kwargs["queue_size"],
            hidden_size=kwargs["hidden_size"],
        )

    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2
    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 1000
        kwargs["max_epochs"] = kwargs["min_epochs"]
        monitor = "train_loss"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    os.makedirs(os.path.dirname(tb_path), exist_ok=True)
    # Save a snapshot of source code and configs alongside the checkpoint
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    _src_dir = os.path.join(_project_root, 'src', 'ms_pred')
    _cfg_dir = os.path.join(_project_root, 'configs', 'clip')
    if os.path.isdir(_src_dir):
        shutil.copytree(_src_dir, f'{tb_path}/ms_pred')
    if os.path.isdir(_cfg_dir):
        shutil.copytree(_cfg_dir, f'{tb_path}/configs')

    checkpoint_best = ModelCheckpoint(
        monitor=monitor, dirpath=tb_path,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=5, mode="min", save_weights_only=False,
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=tb_path, filename="last-{epoch:02d}",
        save_top_k=1, every_n_epochs=5, save_weights_only=False,
        monitor=None,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=kwargs['patience'])
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [earlystop_callback, checkpoint_best, checkpoint_last, lr_monitor]

    from pytorch_lightning.strategies import DDPStrategy
    strategy = DDPStrategy(find_unused_parameters=True) if kwargs["gpu"] and kwargs["num_gpu"] > 1 else None
    precision = kwargs.get("precision", 32)
    if kwargs["num_gpu"] > 1:
        trainer = pl.Trainer(
            logger=[tb_logger, console_logger],
            accelerator="gpu" if kwargs["gpu"] else "cpu",
            devices=kwargs["num_gpu"] if kwargs["gpu"] else 1,
            strategy=strategy,
            callbacks=callbacks,
            gradient_clip_val=5, min_epochs=kwargs["min_epochs"],
            max_epochs=kwargs["max_epochs"], gradient_clip_algorithm="value",
            accumulate_grad_batches=kwargs["grad_accumulate"],
            check_val_every_n_epoch=1, enable_progress_bar=True,
            sync_batchnorm=True if kwargs["gpu"] and kwargs["num_gpu"] > 1 else False,
            precision=precision,
        )
    else:
        trainer = pl.Trainer(
            logger=[tb_logger, console_logger],
            accelerator="gpu" if kwargs["gpu"] else "cpu",
            devices=1 if kwargs["gpu"] else 1,
            callbacks=callbacks,
            gradient_clip_val=5, min_epochs=kwargs["min_epochs"],
            max_epochs=kwargs["max_epochs"], gradient_clip_algorithm="value",
            accumulate_grad_batches=kwargs["grad_accumulate"],
            precision=precision,
        )

    if kwargs["debug_overfit"]:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path

    try:
        best_checkpoint_score = checkpoint_callback.best_model_score.item()
    except AttributeError:
        best_checkpoint_score = 'errors'

    logging.info(f"Best model with val loss of {best_checkpoint_score}")
    return best_checkpoint


if __name__ == "__main__":
    import time
    start_time = time.time()
    best_checkpoint = train_model()
    end_time = time.time()
    logging.info(f"Training Program finished in: {end_time - start_time} seconds")
    path_obj = Path(best_checkpoint)
    parent_path = str(path_obj.parent)
