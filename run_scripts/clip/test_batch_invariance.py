"""Test if UniMol encoder produces batch-invariant embeddings after the padding fix."""
import sys, os
import numpy as np
import torch
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from ms_pred.clip.clip_model_unimol import CLIPModelUniMol
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features, collate_unimol_features
import ms_pred.common as common

CKPT = "/tmp/best_unimol.ckpt"
CACHE_DIR = "data/spec_datasets/msg/retrieval/unimol_features"

TEST_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)CC(NC(=O)C(CC(=O)O)NC(=O)C(CC(C)C)NC(=O)C)C(=O)O",
    "CCCCCCCCCCCCOC(=O)C",
]


def smiles_hash(smi):
    return hashlib.md5(smi.encode()).hexdigest()


def load_feat(smi):
    cache_path = os.path.join(CACHE_DIR, f"{smiles_hash(smi)}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}
    feat, _ = smiles_to_unimol_features(smi, seed=42)
    return feat


def main():
    os.chdir("/home/weiwentao/workspace/ms-pred")

    model = CLIPModelUniMol.load_from_checkpoint(CKPT, map_location='cpu')
    model.eval()
    model = model.cuda()

    # For each test SMILES, encode alone (batch=1) and in a batch with others
    feats = [load_feat(smi) for smi in TEST_SMILES]

    print("Comparing batch_size=1 vs batch_size=N embeddings:")
    print("=" * 70)

    for i, smi in enumerate(TEST_SMILES):
        # Batch size 1
        batch_1 = collate_unimol_features([feats[i]])
        batch_1_gpu = {k: v.cuda() for k, v in batch_1.items()}
        adduct = torch.FloatTensor([common.ion2onehot_pos['[M+H]+']] ).cuda()
        ce = torch.FloatTensor([20.0]).cuda()

        with torch.no_grad():
            proj_1, _ = model.predict_smi(batch_1_gpu, adduct, ce)

        # Batch with all test SMILES (this molecule is at position i)
        batch_all = collate_unimol_features(feats)
        batch_all_gpu = {k: v.cuda() for k, v in batch_all.items()}
        adducts_all = torch.FloatTensor([common.ion2onehot_pos['[M+H]+']] * len(TEST_SMILES)).cuda()
        ces_all = torch.FloatTensor([20.0] * len(TEST_SMILES)).cuda()

        with torch.no_grad():
            proj_all, _ = model.predict_smi(batch_all_gpu, adducts_all, ces_all)

        emb_1 = proj_1[0].cpu().numpy()
        emb_batch = proj_all[i].cpu().numpy()

        max_diff = np.abs(emb_1 - emb_batch).max()
        cosine = np.dot(emb_1, emb_batch) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_batch) + 1e-10)

        status = "PASS" if max_diff < 1e-4 else "FAIL"
        print(f"  {smi[:50]:50s} max_diff={max_diff:.8f} cosine={cosine:.8f} {status}")

    # Also test: same molecule in two different batches
    print("\nSame molecule in different batch contexts:")
    print("=" * 70)
    smi = TEST_SMILES[0]
    feat = feats[0]

    # Context A: with molecules 1,2
    batch_a = collate_unimol_features([feat, feats[1], feats[2]])
    batch_a_gpu = {k: v.cuda() for k, v in batch_a.items()}
    adducts_a = torch.FloatTensor([common.ion2onehot_pos['[M+H]+']] * 3).cuda()
    ces_a = torch.FloatTensor([20.0] * 3).cuda()

    # Context B: with molecules 3,4
    batch_b = collate_unimol_features([feat, feats[3], feats[4]])
    batch_b_gpu = {k: v.cuda() for k, v in batch_b.items()}
    adducts_b = torch.FloatTensor([common.ion2onehot_pos['[M+H]+']] * 3).cuda()
    ces_b = torch.FloatTensor([20.0] * 3).cuda()

    with torch.no_grad():
        proj_a, _ = model.predict_smi(batch_a_gpu, adducts_a, ces_a)
        proj_b, _ = model.predict_smi(batch_b_gpu, adducts_b, ces_b)

    emb_a = proj_a[0].cpu().numpy()
    emb_b = proj_b[0].cpu().numpy()
    max_diff = np.abs(emb_a - emb_b).max()
    cosine = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
    print(f"  Context A vs B: max_diff={max_diff:.8f} cosine={cosine:.8f}")


if __name__ == "__main__":
    main()
