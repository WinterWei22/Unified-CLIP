"""predict_inten.py

Make intensity predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
import copy
import json
from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
from ms_pred.clip import clip_data, clip_model
from ms_pred.mabnet import dag_pyg_data
from ms_pred.mabnet.utils import dgl_to_pyg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument("--num-workers", default=32, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument(
        "--magma-dag-folder",
        help="Folder to have outputs",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels_confs_instrumentation.pkl")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    
    return parser.parse_args()

def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="inten_pred.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))
    binned_out = kwargs["binned_out"]

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    # df = pd.read_pickle(labels)
        
    best_checkpoint = kwargs["checkpoint_pth"]
    logging.info(f"Loading model with from {best_checkpoint}")
    model = clip_model.CLIPModel.load_from_checkpoint(best_checkpoint)
    emb_ce = model.embed_ce
    

    pe_embed_k = 0
    root_encode = "egt2d"
    add_hs = True
    # magma_dag_folder = Path(kwargs["magma_dag_folder"])
    num_workers = kwargs.get("num_workers", 0)
    # all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    # name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    
    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs
    )
    pred_dataset = clip_data.CLIP_SmiDataset(
        df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        emb_ce=emb_ce
    )
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()

    device = torch.device("cuda") if gpu else torch.device("cpu")
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(pred_loader):
            
            root_reprs = batch["root_reprs"].to(device)
            adducts = batch["adducts"].to(device)
            ces = None
            if emb_ce:
                ces = batch["ces"].to(device)
            spec_names = batch["names"]
            specs = batch["specs"].to(device)
            specs_masks = batch["specs_mask"].to(device)

            spec_global_embs, spec_proj = model.predict_spec(
                specs=specs,
                adducts=adducts,
                specs_mask=specs_masks,
            )
            
            for spec, spec_global_emb in zip(
                    spec_names, spec_global_embs
                ):
                    output_obj = {
                        "spec_name": spec,
                        "spec_global_emb": spec_global_emb,
                    }
                    pred_list.append(output_obj)
                    
            # break
    
    if binned_out:
        # Modification: Create a dictionary {name: embedding}
        output = {
            str(item["spec_name"]): item["spec_global_emb"].cpu().numpy() 
            for item in pred_list
        }
        
        # print(f"Mol embeddings of {kwargs['dataset_labels']} ")
        out_file = Path(kwargs["save_dir"]) / "dreams_embedding.pkl"
        
        with open(out_file, "wb") as fp:
            pickle.dump(output, fp)
            
        print(f"{len(output)} embedding items saved in {out_file}")
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
