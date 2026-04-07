"""get_mol_embedding.py

Extract molecular embeddings from trained CLIP model for MS data.
Output format: dict[spectrum_name -> numpy array (512,)]
"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
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
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument(
        "--checkpoint-pth",
        help="Path to checkpoint file",
        default="results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/best.ckpt",
    )
    parser.add_argument(
        "--magma-dag-folder",
        help="Folder containing magma DAG JSON files",
        default="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/",
    )
    parser.add_argument("--dataset-name", default="msg")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument(
        "--output-file",
        default="mol_embedding.pkl",
        help="Output file path for embeddings",
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    checkpoint_path = kwargs["checkpoint_pth"]
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    output_file = Path(kwargs["output_file"])

    logging.basicConfig(
        level=logging.INFO if not kwargs["debug"] else logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {checkpoint_path}")
    model = clip_model.CLIPModel.load_from_checkpoint(checkpoint_path)
    emb_ce = model.embed_ce
    logger.info(f"Model embed_ce: {emb_ce}")

    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]

    df = pd.read_csv(labels, sep="\t")
    logger.info(f"Loaded {len(df)} samples from labels")

    magma_json_files = list(magma_dag_folder.glob("*.json"))
    logger.info(f"Found {len(magma_json_files)} magma DAG files")
    name_to_json = {i.stem.replace("pred_", ""): i for i in magma_json_files}

    common_names = set(df["spec"].values) & set(name_to_json.keys())
    logger.info(f"Found {len(common_names)} common names between labels and magma files")
    
    if len(common_names) < len(df):
        logger.warning(f"Only {len(common_names)} out of {len(df)} spectra have magma files")
        df = df[df["spec"].isin(common_names)]

    pe_embed_k = 0
    root_encode = "egt2d"
    add_hs = True

    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs
    )

    pred_dataset = clip_data.CLIP_SmiDataset(
        df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=kwargs["num_workers"],
        tree_processor=tree_processor,
        emb_ce=emb_ce,
    )

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
    logger.info(f"Using device: {device}")

    output_dict = {}

    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Extracting embeddings"):
            root_reprs = batch["root_reprs"].to(device)
            adducts = batch["adducts"].to(device)
            ces = None
            if emb_ce:
                ces = batch["ces"].to(device)
            spec_names = batch["names"]

            mol_proj, atom_proj = model.predict_smi(
                root_reprs=root_reprs,
                adducts=adducts,
                ces=ces,
            )

            for spec_name, mol_emb in zip(spec_names, mol_proj):
                output_dict[str(spec_name)] = mol_emb.cpu().numpy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)

    logger.info(f"Saved {len(output_dict)} embeddings to {output_file}")
    
    sample_keys = list(output_dict.keys())[:3]
    for k in sample_keys:
        logger.info(f"  {k}: shape={output_dict[k].shape}")


if __name__ == "__main__":
    predict()
