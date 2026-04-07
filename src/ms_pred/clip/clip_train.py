"""train_inten.py

Train model to predict emit intensities for each fragment

"""
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.clip import clip_data

from ms_pred.DreaMS.dreams.models.dreams.dreams import DreaMS 
import ms_pred.DreaMS.dreams.utils.data as du
import ms_pred.DreaMS.dreams.utils.dformats as dformats
from ms_pred.clip.clip_data import CLIPDataset
from ms_pred.clip.clip_model import CLIPModel
from ms_pred.clip.decoys_filtering import preprocess_and_save
import shutil
import os
import numpy as np
from ms_pred.clip.utils import run_predict, run_predict_smi
import subprocess
import math


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
    parser.add_argument("--inject-early", default=False, action="store_true")
    parser.add_argument("--encode-forms", default=False, action="store_true")
    
    parser.add_argument("--local-contra", default=False, action="store_true")
    parser.add_argument("--local-weight", default=0.5, type=float, action="store")

    parser.add_argument("--local-threshold", default=0.8, type=float, action="store")
    parser.add_argument("--local-start-epochs", default=0, action="store", type=int)
    
    parser.add_argument("--decoys", default=False, action="store_true")
    parser.add_argument("--decoys-num", default=5, action="store", type=int)
    parser.add_argument("--decoys-path", default="", action="store", type=str)
    
    parser.add_argument("--easy-decoys-ratio", default=0.2, action="store", type=float)
    parser.add_argument("--preprocess-decoys", default=False, action="store_true")
    parser.add_argument("--frags", default=False, action="store_true")
    
    parser.add_argument("--random-split", default=False, action="store_true")
    parser.add_argument("--train-val", default=False, action="store_true")
    parser.add_argument("--debug-test", default=False, action="store_true")
    
    parser.add_argument("--dreams", default=False, action="store_true")
    parser.add_argument("--pooling-strategy", default='cls', action="store", type=str)
    parser.add_argument("--frozen-dreams",  default=False, action="store_true")
    parser.add_argument("--frozen-epochs", default=10, action="store", type=int)
    parser.add_argument("--use-pretrained-dreams", default=True, action="store_true",
                        help="Use pretrained DreaMS weights. Set --no-pretrained-dreams to disable.")
    parser.add_argument("--no-pretrained-dreams", dest="use_pretrained_dreams", action="store_false",
                        help="Reinitialize DreaMS encoder with random weights (no pretraining).")
    
    
    parser.add_argument("--augment",  default=False, action="store_true")
    parser.add_argument("--mz-shift-aug-p", default=0.3, action="store", type=float)
    parser.add_argument("--mz-shift-aug-max", default=50, action="store", type=float)
    
    parser.add_argument("--patience", default=10, action="store", type=float)
    
    parser.add_argument("--embed-ce",  default=False, action="store_true")
    
    parser.add_argument("--frag-supervised",  default=False, action="store_true")
    parser.add_argument("--frag-path", default='cls', action="store", type=str)
    parser.add_argument("--frag-supervised-weight", default=0.5, type=float, action="store")
    
    parser.add_argument("--spec-sim-entropy", default=False, action="store_true")

    parser.add_argument("--wo-global", default=False, action="store_true")
    parser.add_argument("--frag-noise-ratio", default=0.0, type=float, action="store",
                        help="Noise ratio for fragment labels (0.0-1.0). Each element flipped with this probability.")

    parser.add_argument("--clip-ckpt", default=None, action="store", type=str)
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_align_train_args(parser)
    return parser.parse_args()

def cal_total_steps(epochs, batch_size, num_gpus, dataset_size, accumulate_grad_batches: int = 1):
    effective_batch_size = batch_size * num_gpus * accumulate_grad_batches

    if effective_batch_size == 0:
        raise ValueError("Effective batch size cannot be zero.")

    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)

    total_steps = steps_per_epoch * epochs

    return total_steps

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
    df = pd.read_csv(labels, sep='\t')
    # df = pd.read_pickle(labels)
    if kwargs["debug"]:
        df = df[:1000]
        
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
    
    # # mol split
    if kwargs["train_val"]:
        train_inds = np.concatenate([train_inds, val_inds])
        
    if kwargs["debug_test"]:
        train_inds = np.concatenate([train_inds, val_inds, test_inds])
        val_inds = np.concatenate([val_inds, test_inds])
    
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    
    if kwargs["decoys"]:
        decoys_path = kwargs['decoys_path']
        # decoys_path = data_dir / "decoys" / f"decoys_by_mass_processed_with_similarity_{kwargs['split_name'].split('.')[0]}.tsv"
        decoys_df = pd.read_csv(decoys_path, sep='\t')
        if kwargs["debug"]:
            specs = df['spec'].values.tolist()
            decoys_df = decoys_df[decoys_df['spec'].isin(specs)]
        train_specs = train_df['spec'].values.tolist()
        train_decoys_df = decoys_df[decoys_df['spec'].isin(train_specs)]
        val_specs = val_df['spec'].values.tolist()
        val_decoys_df = decoys_df[decoys_df['spec'].isin(val_specs)]
    else:
        train_decoys_df = None
        val_decoys_df = None
        
    # Spectrum dataset
    # in_pth = '/home/weiwentao/workspace/DreaMS/data/msg/msg.hdf5'
    # msdata = du.MSData.load(in_pth)
    # spec_preproc = du.SpectrumPreprocessor(dformat=dformats.DataFormatA())
    # spec_dataset = msdata.to_torch_dataset(spec_preproc)
    
    
    # split decoys
    # decoys_spec_names = decoys_df["spec"].values
    # decoys_train_inds, decoys_val_inds, _ = common.get_splits(decoys_spec_names, split_file)
    # train_decoys_df = decoys_df.iloc[decoys_train_inds]
    # val_decoys_df = decoys_df.iloc[decoys_val_inds]
    

    # spec split
    # train_spec_dataset = torch.utils.data.Subset(spec_dataset, train_inds)
    # val_spec_dataset = torch.utils.data.Subset(spec_dataset, val_inds)
    
    # Molecule dataset (using pretrain dataset for molecules)
    num_workers = kwargs.get("num_workers", 0)
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    print(f"Found {len(name_to_json)} magma json files in {magma_dag_folder}")


    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=kwargs['pe_embed_k'],
        root_encode=kwargs['root_encode'],
        binned_targs=kwargs['binned_targs'],
        add_hs=kwargs['add_hs'],
    )
    
    if kwargs["preprocess_decoys"]:
        results = preprocess_and_save(
            df=df,
            decoys_df=decoys_df,
            tree_processor=tree_processor,
            ion_map=common.ion2onehot_pos,
            output_dir="./filtered_data",
            main_out_name="train_main",
            decoys_out_name="train_decoys",
            save_format="tsv",
            n_processes=64,
        )
        return

    train_dataset = clip_data.CLIPDataset(
        train_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        decoys=kwargs["decoys"],
        decoys_num=kwargs["decoys_num"],
        decoys_df=train_decoys_df,
        easy_decoys_ratio=kwargs["easy_decoys_ratio"],
        frags=kwargs["frags"],
        augment=kwargs["augment"],
        mz_shift_aug_max=kwargs["mz_shift_aug_max"],
        mz_shift_aug_p=kwargs["mz_shift_aug_p"],
        emb_ce=kwargs["embed_ce"],
        frag_supervised=kwargs['frag_supervised'],
        frags_path=kwargs['frag_path'],
        frag_noise_ratio=kwargs['frag_noise_ratio'],
    )
    val_dataset = clip_data.CLIPDataset(
        val_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        decoys=kwargs["decoys"],
        decoys_num=kwargs["decoys_num"],
        decoys_df=val_decoys_df,
        easy_decoys_ratio=kwargs["easy_decoys_ratio"],
        frags=kwargs["frags"],
        emb_ce=kwargs["embed_ce"],
        frag_supervised=kwargs['frag_supervised'],
        frags_path=kwargs['frag_path']
    )
    
    total_steps = cal_total_steps(kwargs["max_epochs"], kwargs["batch_size"], kwargs["num_gpu"], len(train_dataset))
    
    # # Create CLIP datasets
    # train_dataset = CLIPDataset(train_mol_dataset, train_spec_dataset)
    # val_dataset = CLIPDataset(val_mol_dataset, val_spec_dataset)

    persistent_workers = kwargs["num_workers"] > 0

    # Define dataloaders
    # Assuming a custom collate_fn is needed; use mol's collate if applicable, or define one for dict
    # def collate_fn(batch):
    #     mol_data = [item['mol_data'] for item in batch]
    #     spec_data = [item['spec_data'] for item in batch]
    #     # Assuming mol_pretrain_data has a collate_fn for mol_data; adjust for spec_data
    #     mol_coll = train_mol_dataset.get_collate_fn()(mol_data)  # If available
    #     # For spec_data, assume it's tensor-ready or define collation
    #     spec_coll = torch.utils.data.dataloader.default_collate(spec_data)  # Placeholder
    #     return {'mol_data': mol_coll, 'spec_data': spec_coll}
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
        shuffle=False,   # False
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    
    if kwargs["clip_ckpt"] != None:
        print(f"loading clip ckpt from {kwargs['clip_ckpt']}")
        model = CLIPModel.load_from_checkpoint(kwargs["clip_ckpt"], strict=False)
        model.spec_sim_entropy = kwargs['spec_sim_entropy']
        # Override fine-tuning parameters
        model.local_threshold = kwargs['local_threshold']
        model.local_start_epoch = kwargs['local_start_epochs']
        model.local_contra = kwargs['local_contra']
        model.local_flag = kwargs['local_contra']
        model.frag_supervised = kwargs['frag_supervised']
        model.lr = kwargs['learning_rate']
        model.lr_decay_rate = kwargs['lr_decay_rate']
        model.warmup = kwargs['warm_up']
        model.weight_decay = kwargs['weight_decay']
        model.total_steps = total_steps
        model.frozen_dreams = kwargs['frozen_dreams']
        model.unfreeze_epoch = kwargs['frozen_epochs']
        print(f"Overridden: local_threshold={model.local_threshold}, local_start_epoch={model.local_start_epoch}, lr={model.lr}, unfreeze_epoch={model.unfreeze_epoch}")
    else:
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
            pe_embed_k=kwargs["pe_embed_k"],
            emb_adducts=kwargs["embed_adduct"],
            inject_early=kwargs["inject_early"],
            local_contra=kwargs["local_contra"],
            decoys=kwargs["decoys"],
            frags=kwargs["frags"],
            dreams=kwargs["dreams"],
            local_weight=kwargs["local_weight"],
            local_threshold=kwargs["local_threshold"],
            local_start_epoch=kwargs['local_start_epochs'],
            pooling_strategy=kwargs['pooling_strategy'],
            frozen_dreams=kwargs['frozen_dreams'],
            unfreeze_epoch=kwargs['frozen_epochs'],
            embed_ce=kwargs['embed_ce'],
            total_steps=total_steps,
            frag_supervised=kwargs["frag_supervised"],
            spec_sim_entropy=kwargs["spec_sim_entropy"],
            wo_global=kwargs["wo_global"],
            use_pretrained_dreams=kwargs["use_pretrained_dreams"],
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
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    _src_dir = os.path.join(_project_root, 'src', 'ms_pred')
    _cfg_dir = os.path.join(_project_root, 'configs', 'clip')
    if os.path.isdir(_src_dir):
        shutil.copytree(_src_dir, f'{tb_path}/ms_pred')
    if os.path.isdir(_cfg_dir):
        shutil.copytree(_cfg_dir, f'{tb_path}/configs')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor=monitor,
    #     dirpath=tb_path,
    #     filename="best",  # "{epoch}-{val_loss:.2f}",
    #     save_weights_only=False,
    # )
    # earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # callbacks = [earlystop_callback, checkpoint_callback, lr_monitor]
    
    checkpoint_best = ModelCheckpoint(
    monitor=monitor,
    dirpath=tb_path,
    filename="best-{epoch:02d}-{val_loss:.4f}",
    save_top_k=5,
    mode="min",
    save_weights_only=False,
    )

    # checkpoint_every_5steps = ModelCheckpoint(
    #     dirpath=tb_path,
    #     filename="step-{step}",
    #     every_n_train_steps=50000,
    #     save_top_k=-1,
    #     save_weights_only=False,
    # )

    checkpoint_best_single = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",          # 会生成 best.ckpt
        save_top_k=1,
        mode="min",
        save_weights_only=False,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=tb_path,
        filename="last-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=5,
        save_weights_only=False,
    )

    earlystop_callback = EarlyStopping(monitor=monitor, patience=kwargs['patience'])
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [
        earlystop_callback,
        checkpoint_best,
        checkpoint_last,
        lr_monitor,
        checkpoint_best_single,
    ]

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
    
    return best_checkpoint
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    best_checkpoint = train_model()
    end_time = time.time()
    logging.info(f"Training Program finished in: {end_time - start_time} seconds")
    path_obj = Path(best_checkpoint)
    parent_path = str(path_obj.parent)
    # logging.info(f"Testing Programe Start at {parent_path}")
    # run_predict(parent_path)
    
    # run_predict_smi(parent_path)
    # cal_metrics_script = '/home/weiwentao/workspace/ms-pred/run_scripts/clip/cal_retrieval_metrics_v2.py'
    # command = [
    #     'python', 
    #     cal_metrics_script, 
    #     '--basic_path', 
    #     parent_path,
    # ]
    
    # result = subprocess.run(
    #     command, 
    #     check=True, 
    #     text=True, 
    #     # capture_output=True,
    #     encoding='utf-8'
    # )