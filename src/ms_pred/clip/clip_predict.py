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
        choices=["none", "train_only", "test_only", "debug_special", "filtered_data"],
    )
    
    parser.add_argument("--embed-instrumentation", default=False, action="store_true")
    parser.add_argument("--embed-adducts", default=False, action="store_true")
    parser.add_argument("--inject-early", default=False, action="store_true")
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

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "debug_special":
            names = ["mona_1118"]
            # names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = ["CCMSLIB00001058857"]
            names = ["CCMSLIB00001058185"]
            # names = names[:5]
        elif kwargs["subset_datasets"] == "filtered_data":
            names = splits["spec"].tolist()
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]
        
    best_checkpoint = kwargs["checkpoint_pth"]
    logging.info(f"Loading model with from {best_checkpoint}")
    model = clip_model.CLIPModel.load_from_checkpoint(best_checkpoint)
    # model.emb_adducts = kwargs["embed_adducts"]
    # model.inject_early = kwargs["inject_early"]
    embed_ce = model.embed_ce
    

    pe_embed_k = 0
    root_encode = "egt2d"
    add_hs = True
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    num_workers = kwargs.get("num_workers", 0)
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}

    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs
    )
    pred_dataset = clip_data.CLIPDataset(
        df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
        max_spec_len=64,
        emb_ce=embed_ce
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
            
            specs_scaled = batch["specs_scaled"].to(device)
            specs = batch["specs"].to(device)
            root_reprs = batch["root_reprs"].to(device)
            adducts = batch["adducts"].to(device)
            ces = batch["ces"].to(device)
            specs_masks = batch["specs_mask"].to(device)
            spec_names = batch["names"]
            
            global_sims, local_sims, specs_global_proj, mol_proj, similarity_scores = model.predict(
                specs=specs,
                root_reprs=root_reprs,
                adducts=adducts,
                specs_mask=specs_masks,
                ces=ces,
            )

            for spec, spec_global_emb, mol_emb, global_sim, local_sim, similarity_score in zip(
                spec_names, specs_global_proj, mol_proj, global_sims, local_sims, similarity_scores
            ):
                output_obj = {
                    "spec_name": spec,
                    # "spec_global_emb": spec_global_emb,
                    # "mol_emb": mol_emb,
                    "global_sim": global_sim,
                    "local_sim": local_sim,
                    "similarity_score": similarity_score,
                    # "spec_emb": spec_emb,
                    # "atoms_emb": atoms_emb
                }
                pred_list.append(output_obj)

    def cosine_similarity(a, b):
        """
        Compute cosine similarity between two ndarrays of shape (bs, 1024).
        
        Parameters:
        a (np.ndarray): First array of shape (bs, 1024)
        b (np.ndarray): Second array of shape (bs, 1024)
        
        Returns:
        np.ndarray: Cosine similarities of shape (bs, 1)
        """
        # Compute dot product along the last axis
        dot_product = np.sum(a * b, axis=1)
        
        # Compute norms
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        similarity = dot_product / (norm_a * norm_b + epsilon)
        
        # Reshape to (bs, 1)
        return similarity.reshape(-1, 1)
    
    # Export pred objects
    temperature = model.temperature.item() if hasattr(model.temperature, 'item') else model.temperature
    if binned_out:
        spec_names_ar = [str(i["spec_name"]) for i in pred_list]
        
        # spec_global_emb = np.vstack([i["spec_global_emb"].cpu() for i in pred_list])
        # mol_emb = np.vstack([i["mol_emb"].cpu() for i in pred_list])
        # spec_emb = np.vstack([i["spec_emb"].cpu() for i in pred_list])
        similarity_scores = [i["similarity_score"].cpu() for i in pred_list]
        # atoms_emb = np.vstack([i["atoms_emb"].cpu() for i in pred_list])
        
        local_sim = np.vstack([i["local_sim"].cpu() for i in pred_list])
        # scores = cosine_similarity(spec_global_emb, mol_emb)
        # scores = (spec_emb @ mol_emb.t()) / temperature
        output = {
            "spec_names": spec_names_ar,
            # "spec_emb": spec_emb,
            # "mol_emb": mol_emb,
            # "spec_global_emb": spec_global_emb,
            "similarity_scores": similarity_scores,
            # "atoms_emb": atoms_emb,
            # "cosine_similarity": scores,
            "local_similarity": local_sim, 
        }
        # print(f"Mean cosine similarity: {np.mean(scores, axis=0)}")
        # print(f"Mean cosine similarity: {scores[:10]}")
        print(f"Local_similarity: {local_sim[:10]}")
        out_file = Path(kwargs["save_dir"]) / f"{kwargs['subset_datasets']}_wothreshold_embeddings.pkl"
        with open(out_file, "wb") as fp:
            pickle.dump(output, fp)
        print(f"embeddingfiles saved in {out_file}")
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
