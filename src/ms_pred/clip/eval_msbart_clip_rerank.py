"""eval_msbart_clip_rerank.py

Evaluate whether the CLIP model can re-rank MS-BART generated molecules.

Workflow:
1. Load paper.jsonl (pred SELFIES list, label SELFIES, sample_id)
2. Convert SELFIES to SMILES; build a DataFrame for CLIP_SmiDataset
3. Run CLIP model to get molecule embeddings + spectrum embeddings
4. Compute cosine similarity to re-rank predictions
5. Report TOP-1 and TOP-10 accuracy & tanimoto_sim for both
   original order (frequency-based) and CLIP re-ranked order.
"""

import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from selfies import decoder as selfies_decoder

import ms_pred.common as common
from ms_pred.clip import clip_data, clip_model
from ms_pred.mabnet.utils import dgl_to_pyg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--top-k", default=10, type=int,
                        help="Evaluate TOP-K (default 10)")
    parser.add_argument("--from-saved", default=None, type=str,
                        help="Path to saved details JSON. Skip model inference, "
                             "only re-compute metrics with given --clip-weight")
    parser.add_argument("--checkpoint-pth", default=None,
                        help="Path to CLIP model checkpoint")
    parser.add_argument("--magma-dag-folder", default=None,
                        help="Folder with pred_*.json magma DAG files")
    parser.add_argument("--labels-tsv", default=None,
                        help="Labels TSV with spec/ionization columns")
    parser.add_argument("--paper-jsonl", default=None,
                        help="Path to MS-BART paper.jsonl results")
    parser.add_argument("--clip-weight", default=1.0, type=float,
                        help="Weight for CLIP score in combined ranking. "
                             "0.0 = pure frequency, 1.0 = pure CLIP (default 1.0)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save results (optional)")
    return parser.parse_args()


def selfies_to_canonical_smiles(selfies_str):
    """Convert SELFIES to canonical SMILES. Returns None on failure."""
    try:
        smi = selfies_decoder(selfies_str)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def morgan_fp(mol, radius=2, nbits=2048):
    """Compute Morgan fingerprint."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def compute_tanimoto(smi_a, smi_b):
    """Compute Tanimoto similarity between two SMILES strings."""
    try:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = morgan_fp(mol_a)
        fp_b = morgan_fp(mol_b)
        return TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _eval_one_sample(args):
    """Worker function for parallel evaluate_topk. Returns (is_correct, best_tanimoto)."""
    preds, true_smi, k = args
    topk_preds = [p for p in preds[:k] if p is not None]
    if not topk_preds:
        return 0, 0.0
    is_correct = 1 if true_smi in topk_preds else 0
    # Compute fingerprint for true_smi once, then compare against all candidates
    try:
        true_mol = Chem.MolFromSmiles(true_smi)
        if true_mol is None:
            return is_correct, 0.0
        true_fp = morgan_fp(true_mol)
    except Exception:
        return is_correct, 0.0
    best_tan = 0.0
    for p in topk_preds:
        try:
            mol = Chem.MolFromSmiles(p)
            if mol is None:
                continue
            fp = morgan_fp(mol)
            sim = TanimotoSimilarity(true_fp, fp)
            if sim > best_tan:
                best_tan = sim
        except Exception:
            continue
    return is_correct, best_tan


def evaluate_topk(pred_smiles_list, true_smiles, k, num_workers=20):
    """Evaluate TOP-k accuracy and tanimoto similarity (parallelized)."""
    total = len(true_smiles)
    if total == 0:
        return {"accuracy": 0.0, "tanimoto_sim": 0.0}

    tasks = [(preds, true_smi, k) for preds, true_smi in zip(pred_smiles_list, true_smiles)]
    n_workers = min(num_workers, cpu_count(), total)

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_eval_one_sample, tasks, chunksize=max(1, total // n_workers)),
            total=total, desc=f"Eval TOP-{k}"
        ))

    correct = sum(r[0] for r in results)
    tanimoto_sims = [r[1] for r in results]

    accuracy = correct / total
    avg_tanimoto = float(np.mean(tanimoto_sims))
    return {"accuracy": round(accuracy, 4), "tanimoto_sim": round(avg_tanimoto, 4)}


def rerank_from_saved(saved_results, clip_weight):
    """Re-rank saved results with a new clip_weight.

    The saved details contain orig_smiles (frequency order) and
    reranked_smiles / reranked_cosine_sims (pure CLIP order).
    We reconstruct per-candidate CLIP scores and frequency scores,
    then combine with the given weight.
    """
    all_results = []
    for r in saved_results:
        orig_smiles = r["orig_smiles"]
        true_smiles = r["true_smiles"]

        # Build CLIP score lookup from saved pure-CLIP ranking
        clip_ranked_smiles = r["reranked_smiles"]
        clip_ranked_sims = r.get("reranked_cosine_sims", r.get("reranked_scores", []))

        smi_to_clip = {}
        for smi, sim in zip(clip_ranked_smiles, clip_ranked_sims):
            if smi not in smi_to_clip:
                smi_to_clip[smi] = sim

        # Frequency score: position in orig_smiles (index 0 = highest freq)
        n = len(orig_smiles)
        combined = []
        for idx, smi in enumerate(orig_smiles):
            freq_score = (n - idx) / n
            clip_score = smi_to_clip.get(smi, 0.0)
            score = (1.0 - clip_weight) * freq_score + clip_weight * clip_score
            combined.append((smi, score))

        combined.sort(key=lambda x: -x[1])
        reranked_smiles = [s for s, _ in combined]
        reranked_scores = [sc for _, sc in combined]

        all_results.append({
            "sample_id": r["sample_id"],
            "true_smiles": true_smiles,
            "orig_smiles": orig_smiles,
            "reranked_smiles": reranked_smiles,
            "reranked_scores": reranked_scores,
        })
    return all_results


def main():
    args = get_args()
    kwargs = args.__dict__

    if kwargs["save_dir"]:
        save_dir = Path(kwargs["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    clip_w = kwargs["clip_weight"]

    # ──────────────────────────────────────────────
    # Fast path: load from saved details JSON
    # ──────────────────────────────────────────────
    if kwargs["from_saved"]:
        print(f"Loading saved results from {kwargs['from_saved']} ...")
        with open(kwargs["from_saved"]) as f:
            saved_results = json.load(f)
        print(f"  Loaded {len(saved_results)} samples")

        all_results = rerank_from_saved(saved_results, clip_w)
        print(f"  Re-ranked with clip_weight={clip_w}")

    else:
        # Full inference path — require model args
        assert kwargs["checkpoint_pth"], "--checkpoint-pth required when not using --from-saved"
        assert kwargs["magma_dag_folder"], "--magma-dag-folder required when not using --from-saved"
        assert kwargs["labels_tsv"], "--labels-tsv required when not using --from-saved"
        assert kwargs["paper_jsonl"], "--paper-jsonl required when not using --from-saved"

        # ──────────────────────────────────────────────
        # 1. Load paper.jsonl
        # ──────────────────────────────────────────────
        print("Loading paper.jsonl ...")
        samples = []
        with open(kwargs["paper_jsonl"]) as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"  Loaded {len(samples)} samples")

        # Convert SELFIES -> canonical SMILES
        print("Converting SELFIES to SMILES ...")
        for s in tqdm(samples):
            s["true_smiles"] = selfies_to_canonical_smiles(s["label"])
            s["pred_smiles"] = [selfies_to_canonical_smiles(sf) for sf in s["pred"]]

        # Filter out samples where true_smiles conversion failed
        samples = [s for s in samples if s["true_smiles"] is not None]
        print(f"  {len(samples)} samples with valid ground truth SMILES")

        # ──────────────────────────────────────────────
        # 2. Load labels TSV (for ionization/adduct info)
        # ──────────────────────────────────────────────
        labels_df = pd.read_csv(kwargs["labels_tsv"], sep="\t")
        name_to_ionization = dict(zip(labels_df["spec"].values, labels_df["ionization"].values))
        if "collision_energies" in labels_df.columns:
            name_to_ce = dict(zip(labels_df["spec"].values, labels_df["collision_energies"].values))
        else:
            name_to_ce = {}

        # ──────────────────────────────────────────────
        # 3. Load CLIP model
        # ──────────────────────────────────────────────
        print(f"Loading CLIP model from {kwargs['checkpoint_pth']} ...")
        model = clip_model.CLIPModel.load_from_checkpoint(kwargs["checkpoint_pth"])
        emb_ce = model.embed_ce
        model.eval()

        gpu = kwargs["gpu"]
        device = torch.device("cuda") if gpu else torch.device("cpu")
        if gpu:
            model = model.cuda()

        # ──────────────────────────────────────────────
        # 4. Build magma_map
        # ──────────────────────────────────────────────
        magma_dag_folder = Path(kwargs["magma_dag_folder"])
        all_json_pths = list(magma_dag_folder.glob("*.json"))
        name_to_json = {p.stem.replace("pred_", ""): p for p in all_json_pths}

        # ──────────────────────────────────────────────
        # 5. For each sample, compute spec embedding (once)
        #    and mol embeddings for all candidates
        # ──────────────────────────────────────────────
        pe_embed_k = 0
        root_encode = "egt2d"
        add_hs = True

        tree_processor = clip_data.TreeProcessor(
            pe_embed_k=pe_embed_k, root_encode=root_encode, add_hs=add_hs
        )

        print("Computing embeddings ...")
        all_results = []  # list of dict per sample

        for sample in tqdm(samples, desc="Processing samples"):
            sample_id = sample["sample_id"]
            true_smiles = sample["true_smiles"]
            pred_smiles_list = sample["pred_smiles"]

            if sample_id not in name_to_json:
                print(f"  WARNING: {sample_id} not in magma_dag_folder, skipping")
                continue
            if sample_id not in name_to_ionization:
                print(f"  WARNING: {sample_id} not in labels TSV, skipping")
                continue

            ionization = name_to_ionization[sample_id]
            if ionization not in common.ion2onehot_pos:
                # Try remap
                ionization = common.ion_remap.get(ionization, ionization)
            if ionization not in common.ion2onehot_pos:
                print(f"  WARNING: unknown ionization {ionization} for {sample_id}, skipping")
                continue

            adduct_idx = common.ion2onehot_pos[ionization]

            # Load spectrum from magma JSON
            json_path = name_to_json[sample_id]
            with open(json_path, "r") as f:
                tree_data = json.load(f)
            raw_spec = tree_data["raw_spec"]
            prec_mz = tree_data["prec_mz"]

            # Prepare spectrum tensor (same as CLIP_SmiDataset.__getitem__)
            spec = list(raw_spec)
            spec.insert(0, [prec_mz, 1.1])

            # Collect valid candidate SMILES (including ground truth for reference)
            # We need to get mol graph for each candidate
            candidate_smiles = []  # valid SMILES
            candidate_indices = []  # original index in pred_smiles_list
            candidate_root_reprs = []

            for idx, smi in enumerate(pred_smiles_list):
                if smi is None:
                    continue
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    inchi = Chem.MolToInchi(mol)
                    if inchi is None:
                        continue
                    root_repr = tree_processor.process_mol(inchi)
                    import dgl
                    if not isinstance(root_repr, dgl.DGLGraph):
                        continue
                    candidate_smiles.append(smi)
                    candidate_indices.append(idx)
                    candidate_root_reprs.append(root_repr)
                except Exception as e:
                    continue

            if len(candidate_smiles) == 0:
                continue

            # Build batch data for all candidates (same spectrum, different molecules)
            # Process in mini-batches
            batch_size = kwargs["batch_size"]
            all_mol_embs = []
            spec_global_emb = None

            for start in range(0, len(candidate_smiles), batch_size):
                end = min(start + batch_size, len(candidate_smiles))
                batch_reprs = candidate_root_reprs[start:end]
                bs = len(batch_reprs)

                # Batch graph
                batched_reprs = dgl.batch(batch_reprs)
                batched_reprs = dgl_to_pyg(batched_reprs).to(device)

                # Adducts: same for all candidates
                adducts = torch.FloatTensor([adduct_idx] * bs).to(device)

                # Spectrum: replicate for batch
                max_spec_len = 64
                specs_list = [spec] * bs
                batched_specs, specs_mask = clip_data.pad_tuples_to_tensor(
                    specs_list, max_len=max_spec_len
                )
                batched_specs = batched_specs.to(device)
                specs_mask = specs_mask.to(device)

                ces = None
                if emb_ce:
                    ce_val = name_to_ce.get(sample_id, 0.0)
                    ces = torch.FloatTensor([float(ce_val)] * bs).to(device)

                with torch.no_grad():
                    # Call predict_spec and predict_smi directly to avoid
                    # local similarity mask dimension mismatch in model.predict()
                    spec_global_emb_batch, _ = model.predict_spec(
                        batched_specs, adducts, specs_mask=specs_mask
                    )
                    mol_proj, _ = model.predict_smi(
                        batched_reprs, adducts, ces
                    )

                # Store embeddings
                all_mol_embs.append(mol_proj.cpu().numpy())
                # Spectrum embedding is the same for all candidates; take from first batch
                if spec_global_emb is None:
                    spec_global_emb = spec_global_emb_batch[0].cpu().numpy()

            all_mol_embs = np.vstack(all_mol_embs)  # (n_candidates, dim)

            # Compute cosine similarities
            cos_sims = np.array([
                cosine_similarity(spec_global_emb, all_mol_embs[i])
                for i in range(len(candidate_smiles))
            ])

            # Frequency-based score: higher rank (lower index) = higher score
            # Normalize to [0, 1] range
            n_cands = len(candidate_smiles)
            freq_scores = np.array([
                (n_cands - candidate_indices[i]) / n_cands
                for i in range(n_cands)
            ])

            # Combined score: weighted sum of frequency and CLIP scores
            freq_w = 1.0 - clip_w
            combined_scores = freq_w * freq_scores + clip_w * cos_sims

            # Re-rank by combined score (descending)
            reranked_order = np.argsort(-combined_scores)
            reranked_smiles = [candidate_smiles[i] for i in reranked_order]
            reranked_sims = combined_scores[reranked_order]

            # Original order (frequency-based from MS-BART)
            orig_order = np.argsort(candidate_indices)
            orig_smiles = [candidate_smiles[i] for i in orig_order]

            all_results.append({
                "sample_id": sample_id,
                "true_smiles": true_smiles,
                "orig_smiles": orig_smiles,
                "reranked_smiles": reranked_smiles,
                "reranked_cosine_sims": cos_sims[reranked_order].tolist(),
                "reranked_scores": reranked_sims.tolist(),
            })

        print(f"\nProcessed {len(all_results)} samples successfully")

    # ──────────────────────────────────────────────
    # 6. Compute metrics
    # ──────────────────────────────────────────────
    top_k = kwargs["top_k"]

    orig_preds = [r["orig_smiles"] for r in all_results]
    reranked_preds = [r["reranked_smiles"] for r in all_results]
    true_smiles_list = [r["true_smiles"] for r in all_results]
    n_workers = kwargs["num_workers"]

    clip_w = kwargs["clip_weight"]
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS  (clip_weight={clip_w})")
    print("=" * 60)

    # Compute all metrics at once
    eval_ks = sorted(set([1, top_k]))
    all_metrics = {}

    # Oracle: best possible result from all candidates (upper bound)
    oracle_k = max(len(p) for p in orig_preds)
    oracle_metrics = evaluate_topk(orig_preds, true_smiles_list, k=oracle_k, num_workers=n_workers)
    all_metrics["oracle"] = oracle_metrics
    print(f"\n--- Oracle (best of all candidates) ---")
    print(f"    Accuracy:     {oracle_metrics['accuracy']}")
    print(f"    Tanimoto Sim: {oracle_metrics['tanimoto_sim']}")

    for k in eval_ks:
        orig_metrics = evaluate_topk(orig_preds, true_smiles_list, k, num_workers=n_workers)
        reranked_metrics = evaluate_topk(reranked_preds, true_smiles_list, k, num_workers=n_workers)
        all_metrics[f"top_{k}"] = {"original": orig_metrics, "clip_reranked": reranked_metrics}

        print(f"\n--- TOP-{k} ---")
        print(f"  Original (frequency-based):")
        print(f"    Accuracy:     {orig_metrics['accuracy']}")
        print(f"    Tanimoto Sim: {orig_metrics['tanimoto_sim']}")
        print(f"  Re-ranked (freq_w={1-clip_w:.1f}, clip_w={clip_w:.1f}):")
        print(f"    Accuracy:     {reranked_metrics['accuracy']}")
        print(f"    Tanimoto Sim: {reranked_metrics['tanimoto_sim']}")

    # ──────────────────────────────────────────────
    # 7. Save results
    # ──────────────────────────────────────────────
    if save_dir:
        results_file = save_dir / "eval_msbart_clip_rerank_results.json"
        save_data = {
            "num_samples": len(all_results),
            "top_k": top_k,
            "clip_weight": clip_w,
            "metrics": all_metrics,
        }

        # Also save per-sample details
        per_sample_file = save_dir / "eval_msbart_clip_rerank_details.json"
        with open(per_sample_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPer-sample details saved to {per_sample_file}")

        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Summary results saved to {results_file}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nProgram finished in {end_time - start_time:.1f} seconds")
