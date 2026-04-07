"""extract_mol_embedding.py

Extract molecular embeddings from trained CLIP model using SMILES input.
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
import dgl

import ms_pred.common as common
from ms_pred.clip import clip_data, clip_model
from ms_pred.mabnet.utils import dgl_to_pyg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-pth", type=str, required=True,
                        help="Path to the trained CLIP model checkpoint")
    parser.add_argument("--smiles-file", type=str, required=True,
                        help="Path to the SMILES file (one SMILES per line)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save embeddings")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of workers for data loading")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU if available")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()


class SmilesDataset(torch.utils.data.Dataset):
    """Simple dataset for SMILES strings without spectral data."""
    
    def __init__(self, smiles_list, tree_processor, default_adduct=0, default_ce=0.0):
        self.smiles_list = smiles_list
        self.tree_processor = tree_processor
        self.default_adduct = default_adduct
        self.default_ce = default_ce
        self.adduct_embedder = common.ion2onehot_pos
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            inchi = Chem.MolToInchi(mol)
            root_repr = self.tree_processor.process_mol(inchi)
            
            if not isinstance(root_repr, dgl.DGLGraph):
                raise ValueError(f"Non-graph representation for SMILES: {smiles}")
            
            return {
                "idx": idx,
                "smiles": smiles,
                "root_repr": root_repr,
                "adduct": self.default_adduct,
                "ce": self.default_ce,
            }
        except Exception as e:
            logging.warning(f"Error processing SMILES at index {idx}: {smiles}, Error: {e}")
            return {
                "idx": idx,
                "smiles": smiles,
                "root_repr": None,
                "adduct": self.default_adduct,
                "ce": self.default_ce,
                "error": str(e),
            }
    
    @staticmethod
    def collate_fn(batch):
        valid_items = [item for item in batch if item["root_repr"] is not None]
        invalid_items = [item for item in batch if item["root_repr"] is None]
        
        if not valid_items:
            raise ValueError("All items in the batch failed to process.")
        
        indices = [item["idx"] for item in batch]
        smiles_list = [item["smiles"] for item in batch]
        errors = {item["idx"]: item["error"] for item in invalid_items if "error" in item}
        
        valid_root_reprs = [item["root_repr"] for item in valid_items]
        valid_adducts = torch.tensor([item["adduct"] for item in valid_items])
        valid_ces = torch.tensor([item["ce"] for item in valid_items])
        
        batched_reprs = dgl.batch(valid_root_reprs)
        batched_reprs_pyg = dgl_to_pyg(batched_reprs)
        
        return {
            "indices": indices,
            "smiles": smiles_list,
            "root_reprs": batched_reprs_pyg,
            "adducts": valid_adducts,
            "ces": valid_ces,
            "valid_mask": [item["idx"] in [v["idx"] for v in valid_items] for item in batch],
            "errors": errors,
        }


def extract_embeddings(args):
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    checkpoint_path = Path(args.checkpoint_pth)
    smiles_file = Path(args.smiles_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {checkpoint_path}")
    model = clip_model.CLIPModel.load_from_checkpoint(checkpoint_path)
    
    pe_embed_k = model.pe_embed_k if hasattr(model, 'pe_embed_k') else 0
    root_encode = "egt2d"
    add_hs = True
    
    logger.info(f"Model config: pe_embed_k={pe_embed_k}, emb_adducts={model.emb_adducts}, embed_ce={model.embed_ce}")
    
    logger.info(f"Reading SMILES from {smiles_file}")
    with open(smiles_file, 'r') as f:
        lines = f.readlines()
    
    smiles_data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            smiles = parts[-1] if len(parts) > 1 else parts[0]
            smiles_data.append(smiles)
    
    logger.info(f"Loaded {len(smiles_data)} SMILES molecules")
    
    tree_processor = clip_data.TreeProcessor(
        pe_embed_k=pe_embed_k, 
        root_encode=root_encode, 
        add_hs=add_hs
    )
    
    dataset = SmilesDataset(smiles_data, tree_processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=SmilesDataset.collate_fn,
        shuffle=False,
    )
    
    if args.gpu:
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda:7")
                torch.cuda.set_device(7)
                test_tensor = torch.zeros(1).to(device)
                del test_tensor
                logger.info(f"Using GPU device: cuda:7")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU (error: {e}), falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    all_smiles = []
    all_valid = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            root_reprs = batch["root_reprs"].to(device)
            adducts = batch["adducts"].to(device)
            ces = batch["ces"].to(device)
            
            if model.emb_adducts and model.inject_early:
                mol_proj, atom_proj = model.predict_smi(root_reprs, adducts, ces)
            else:
                mol_proj, atom_proj = model.predict_smi(root_reprs, adducts, ces)
            
            valid_smiles = [batch["smiles"][i] for i, valid in enumerate(batch["valid_mask"]) if valid]
            
            all_embeddings.append(mol_proj.cpu().numpy())
            all_smiles.extend(valid_smiles)
            all_valid.extend(batch["valid_mask"])
    
    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
    else:
        embeddings_array = np.array([])
    
    output_data = {
        "smiles": all_smiles,
        "embeddings": embeddings_array,
        "valid_mask": all_valid,
        "checkpoint": str(checkpoint_path),
    }
    
    output_file = output_dir / "mol_embeddings.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    logger.info(f"Saved {len(all_smiles)} embeddings to {output_file}")
    logger.info(f"Embedding shape: {embeddings_array.shape}")
    
    return output_data


if __name__ == "__main__":
    args = get_args()
    extract_embeddings(args)
