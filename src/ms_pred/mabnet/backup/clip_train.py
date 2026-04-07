"""train_inten.py

Train model to predict emit intensities for each fragment

"""
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.mabnet import clip_data

from ms_pred.DreaMS.dreams.models.dreams.dreams import DreaMS 
import ms_pred.DreaMS.dreams.utils.data as du
import ms_pred.DreaMS.dreams.utils.dformats as dformats
from ms_pred.mabnet.clip_data import CLIPDataset
from ms_pred.mabnet.clip_model import CLIPModel


def add_align_train_args(parser):
    # Simplified arguments for alignment training
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-overfit", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_align/")
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_1.tsv")

    parser.add_argument("--batch-size", default=32, action="store", type=int)  # Increased for contrastive learning
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--min-epochs", default=0, action="store", type=int)
    parser.add_argument("--learning-rate", default=1e-3, action="store", type=float)
    parser.add_argument("--weight-decay", default=0, action="store", type=float)
    parser.add_argument("--grad-accumulate", default=1, type=int, action="store")

    # Alignment-specific params
    parser.add_argument("--hidden-size", default=128, action="store", type=int)
    parser.add_argument("--projection-dim", default=128, action="store", type=int)
    parser.add_argument("--temperature", default=0.07, action="store", type=float)
    parser.add_argument("--num-gpu", default=1, action="store", type=int)
    parser.add_argument("--pe-embed-k", default=0, action="store", type=int)
    parser.add_argument("--root-encode", default='egt2d', action="store", type=str)
    parser.add_argument("--binned-targs", default=False, action="store_true")
    parser.add_argument("--add-hs", default=False, action="store_true")
    

    # Encoder checkpoints
    parser.add_argument("--spec-ckpt", default="/home/weiwentao/workspace/DreaMS/dreams/models/pretrained/ssl_model.ckpt", type=str, action="store")
    parser.add_argument("--mol-ckpt", default=None, type=str, action="store")

    # MabNet-specific for mol encoder
    parser.add_argument("--mabnet-layers", default=6, type=int, action="store")
    parser.add_argument("--mabnet-heads", default=8, type=int, action="store")
    parser.add_argument("--edge-update", default=False, action="store_true")
    parser.add_argument("--magma-dag-folder", default=None, type=str, action="store")

    # Additional parameters from the error message
    parser.add_argument("--lr-decay-rate", default=0.825, action="store", type=float)
    parser.add_argument("--warm-up", default=1000, action="store", type=int)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--mpnn-type", default="GGNN", action="store", type=str, choices=["GGNN", "GINE", "PNA"])
    parser.add_argument("--pool-op", default="avg", action="store", type=str)
    parser.add_argument("--set-layers", default=0, action="store", type=int)
    parser.add_argument("--frag-set-layers", default=3, action="store", type=int)
    parser.add_argument("--mlp-layers", default=1, action="store", type=int)
    parser.add_argument("--gnn-layers", default=6, action="store", type=int)
    parser.add_argument("--loss-fn", default="cosine", action="store", type=str, choices=["mse", "cosine"])
    parser.add_argument("--frag-layers", default=8, action="store", type=int)
    parser.add_argument("--frag-heads", default=8, action="store", type=int)
    parser.add_argument("--embed-adduct", default=False, action="store_true")
    parser.add_argument("--encode-forms", default=False, action="store_true")

    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_align_train_args(parser)
    return parser.parse_args()

def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="align_train.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels =  data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_pickle(labels)
    if kwargs["debug"]:
        df = df[:1000]

    # Spectrum dataset
    in_pth = '/home/weiwentao/workspace/DreaMS/data/msg/msg.hdf5'
    msdata = du.MSData.load(in_pth)
    spec_preproc = du.SpectrumPreprocessor(dformat=dformats.DataFormatA())
    spec_dataset = msdata.to_torch_dataset(spec_preproc)
    
    spec_names = df["spec"].values
    split_file = data_dir / "splits" / kwargs["split_name"]
    train_inds, val_inds, _ = common.get_splits(spec_names, split_file)
    # mol split
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    # spec split
    train_spec_dataset = torch.utils.data.Subset(spec_dataset, train_inds)
    val_spec_dataset = torch.utils.data.Subset(spec_dataset, val_inds)
    
    # Molecule dataset (using pretrain dataset for molecules)
    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}


    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=kwargs['pe_embed_k'],
        root_encode=kwargs['root_encode'],
        binned_targs=kwargs['binned_targs'],
        add_hs=kwargs['add_hs'],
    )

    train_mol_dataset = clip_data.MolDataset(
        train_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )
    val_mol_dataset = clip_data.MolDataset(
        val_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )

    # Create CLIP datasets
    train_dataset = CLIPDataset(train_mol_dataset, train_spec_dataset)
    val_dataset = CLIPDataset(val_mol_dataset, val_spec_dataset)

    persistent_workers = kwargs["num_workers"] > 0

    # Define dataloaders
    # Assuming a custom collate_fn is needed; use mol's collate if applicable, or define one for dict
    def collate_fn(batch):
        mol_data = [item['mol_data'] for item in batch]
        spec_data = [item['spec_data'] for item in batch]
        # Assuming mol_pretrain_data has a collate_fn for mol_data; adjust for spec_data
        mol_coll = train_mol_dataset.get_collate_fn()(mol_data)  # If available
        # For spec_data, assume it's tensor-ready or define collation
        spec_coll = torch.utils.data.dataloader.default_collate(spec_data)  # Placeholder
        return {'mol_data': mol_coll, 'spec_data': spec_coll}

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
    
    # Define model
    model = CLIPModel(
        hidden_size=kwargs["hidden_size"],
        mabnet_heads=kwargs["mabnet_heads"],
        mabnet_layers=kwargs["mabnet_layers"],
        edge_update=kwargs["edge_update"],
        projection_dim=kwargs["projection_dim"],
        temperature=kwargs["temperature"],
        lr=kwargs["learning_rate"],
        spec_ckpt_path=kwargs["spec_ckpt"],
        mol_ckpt_path=kwargs["mol_ckpt"],
        lr_decay_rate=kwargs["lr_decay_rate"],
        warmup=kwargs["warm_up"],
        dropout=kwargs["dropout"],
        weight_decay=kwargs["weight_decay"], 
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
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",  # "{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [earlystop_callback, checkpoint_callback, lr_monitor]

    strategy = "ddp" if kwargs["gpu"] and kwargs["num_gpu"] > 1 else None
    if kwargs["num_gpu"] > 1:
        trainer = pl.Trainer(
            logger=[tb_logger, console_logger],
            accelerator="gpu" if kwargs["gpu"] else "cpu",
            devices=kwargs["num_gpu"] if kwargs["gpu"] else 1,
            strategy=strategy,
            callbacks=callbacks,
            gradient_clip_val=5,
            min_epochs=kwargs["min_epochs"],
            max_epochs=kwargs["max_epochs"],
            gradient_clip_algorithm="value",
            accumulate_grad_batches=kwargs["grad_accumulate"],
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            sync_batchnorm=True if kwargs["gpu"] and kwargs["num_gpu"] > 1 else False,
        )
    else:
        trainer = pl.Trainer(
            logger=[tb_logger, console_logger],
            accelerator="gpu" if kwargs["gpu"] else "cpu",
            gpus=1 if kwargs["gpu"] else 0,
            callbacks=callbacks,
            gradient_clip_val=5,
            min_epochs=kwargs["min_epochs"],
            max_epochs=kwargs["max_epochs"],
            gradient_clip_algorithm="value",
            accumulate_grad_batches=kwargs["grad_accumulate"],
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

        
    logging.info(
        f"Best model with val loss of {best_checkpoint_score}"
    )
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")