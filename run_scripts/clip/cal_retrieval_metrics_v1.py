import pickle
import numpy as np
from collections import defaultdict, Counter

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _to_float_array(x, name):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return x

def l2_normalize(x, axis=-1, eps=1e-12):
    x = _to_float_array(x, "embeddings")
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def cosine_similarity(a, b, normalize=True):
    """
    a: (D,) or (N, D)
    b: (M, D)
    return: (M,) if a is (D,), else (N, M)
    If normalize=False, returns dot products, not true cosine similarity.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected a.ndim in {{1,2}} and b.ndim==2, got {a.ndim=}, {b.ndim=}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dim mismatch: a.shape[1]={a.shape[1]} vs b.shape[1]={b.shape[1]}")
    a = _to_float_array(a, "a")
    b = _to_float_array(b, "b")
    if normalize:
        a = l2_normalize(a, axis=-1)
        b = l2_normalize(b, axis=-1)
    return a @ b.T

def build_candidate_index(candidate_dict):
    """
    candidate_dict:
      - spec_names: list[str]
      - mol_emb: np.ndarray (T, D)
    For each spec_name, the first occurrence is target, rest are decoys.
    """
    spec_names = candidate_dict["spec_names"]
    mol_emb = _to_float_array(candidate_dict["mol_emb"], "candidate mol_emb")

    if len(spec_names) != len(mol_emb):
        raise ValueError(f"Length mismatch: len(spec_names)={len(spec_names)} vs mol_emb.shape[0]={mol_emb.shape[0]}")

    index = defaultdict(lambda: {"embs": [], "target_idx": None})
    seen_first = set()

    for i, name in enumerate(spec_names):
        if name not in seen_first:
            index[name]["target_idx"] = len(index[name]["embs"])
            seen_first.add(name)
        index[name]["embs"].append(mol_emb[i])

    for name in list(index.keys()):
        if len(index[name]["embs"]) == 0:
            # Shouldn't happen, but keep guard
            index[name]["embs"] = np.zeros((0, mol_emb.shape[1]), dtype=mol_emb.dtype)
        else:
            index[name]["embs"] = np.stack(index[name]["embs"], axis=0)  # (K, D)

    return dict(index)

def evaluate_topk(label_dict, candidate_index, topk=(1,2,3,4,5,6,7,8,9,10), normalize=True):
    """
    Returns:
      - metrics: dict, top1..top10 accuracy
      - details: per-query details
      - sim_stats: dict with mean target sim and mean decoy sim
    """
    spec_names = label_dict.get("spec_name", None)
    if spec_names is None:
        spec_names = label_dict["spec_names"]
    spec_emb = _to_float_array(label_dict["spec_emb"], "label spec_emb")

    if len(spec_names) != len(spec_emb):
        raise ValueError(f"Length mismatch: len(spec_names)={len(spec_names)} vs spec_emb.shape[0]={spec_emb.shape[0]}")

    correct_at_k = {k: 0 for k in topk}
    total = 0

    details = []

    target_sims = []
    decoy_sims = []

    for i, name in enumerate(spec_names):
        query = spec_emb[i]  # (D,)

        cand = candidate_index.get(name, None)
        if cand is None:
            details.append({
                "spec_name": name,
                "target_rank": None,
                "num_candidates": 0,
                "note": "no candidates"
            })
            continue

        embs = cand["embs"]             # (K, D)
        target_idx = cand["target_idx"] # int

        if embs.shape[0] == 0 or target_idx is None:
            details.append({
                "spec_name": name,
                "target_rank": None,
                "num_candidates": int(embs.shape[0]),
                "note": "invalid candidate set"
            })
            continue

        sims = cosine_similarity(query, embs, normalize=normalize).reshape(-1)  # (K,)
        # Guard NaNs
        if not np.isfinite(sims).all():
            details.append({
                "spec_name": name,
                "target_rank": None,
                "num_candidates": int(embs.shape[0]),
                "note": "non-finite similarity"
            })
            continue

        # Rank via count of strictly greater to be tie-robust
        target_sim = sims[target_idx]
        greater_count = int(np.sum(sims > target_sim))
        target_rank = greater_count + 1

        # Accumulate Top-K
        total += 1
        for k in topk:
            if target_rank <= k:
                correct_at_k[k] += 1

        # Similarity stats
        target_sims.append(float(target_sim))
        if embs.shape[0] > 1:
            mask = np.ones_like(sims, dtype=bool)
            mask[target_idx] = False
            decoy_mean = float(np.mean(sims[mask]))
            decoy_sims.append(decoy_mean)
        else:
            decoy_mean = None

        details.append({
            "spec_name": name,
            "target_rank": target_rank,
            "num_candidates": int(embs.shape[0]),
            "target_sim": float(target_sim),
            "decoy_mean_sim": decoy_mean,
        })

    metrics = {f"top{k}": (correct_at_k[k] / total) if total > 0 else 0.0 for k in topk}

    sim_stats = {
        "mean_target_similarity": float(np.mean(target_sims)) if len(target_sims) > 0 else 0.0,
        "mean_decoy_similarity": float(np.mean(decoy_sims)) if len(decoy_sims) > 0 else 0.0,
        "num_evaluated": total,
        "num_with_decoys": len(decoy_sims),
    }

    return metrics, details, sim_stats

def main(
    label_pkl_path="label.pkl",
    candidate_pkl_path="candidate.pkl",
    topk_max=10,
    verbose=True,
    normalize=True
):
    label_dict = load_pkl(label_pkl_path)
    candidate_dict = load_pkl(candidate_pkl_path)

    candidate_index = build_candidate_index(candidate_dict)

    topk = tuple(range(1, topk_max + 1))
    metrics, details, sim_stats = evaluate_topk(label_dict, candidate_index, topk=topk, normalize=normalize)

    # ==================== 新增统计 ====================
    valid_ranks = []          # 用来算 Mean / Median Rank
    decoy_counts = []         # 每个 query 的 decoy 数量（num_candidates - 1）

    for d in details:
        if d.get("target_rank") is not None:           # 有效评估的 query
            valid_ranks.append(d["target_rank"])
            decoy_counts.append(d["num_candidates"] - 1)

    mean_rank   = np.mean(valid_ranks) if valid_ranks else 0.0
    median_rank = np.median(valid_ranks) if valid_ranks else 0.0
    mean_decoys = np.mean(decoy_counts) if decoy_counts else 0.0

    # Candidate count distribution（保持你原来的）
    cand_count_per_query = [d["num_candidates"] for d in details if d.get("num_candidates") is not None]
    cand_count_dist = Counter(cand_count_per_query)
    # ====================================================

    if verbose:
        print("Evaluation results:")
        for k in topk:
            print(f"Top-{k} accuracy: {metrics[f'top{k}']:.4f}")
        evaluated = len(valid_ranks)
        skipped = len(details) - evaluated
        print(f"Total evaluated queries: {evaluated}")
        if skipped > 0:
            print(f"Skipped queries (no/invalid candidates): {skipped}")

        print("\nSimilarity statistics:")
        print(f"Mean target similarity: {sim_stats['mean_target_similarity']:.6f}")
        print(f"Mean decoy similarity:  {sim_stats['mean_decoy_similarity']:.6f}")
        print(f"Queries counted for mean target similarity: {sim_stats['num_evaluated']}")
        print(f"Queries with decoys (for decoy mean):      {sim_stats['num_with_decoys']}")

        # ==================== 新增输出 ====================
        print("\nRanking statistics:")
        print(f"Mean Rank                : {mean_rank:.2f}")
        print(f"Median Rank              : {median_rank:.1f}")
        print(f"Mean number of decoys    : {mean_decoys:.2f} (≈ average candidates - 1)")
        # ====================================================

        # print("\nCandidate count distribution per query:")
        # total_queries = len(details)
        # for count in sorted(cand_count_dist.keys()):
        #     num_queries = cand_count_dist[count]
        #     pct = (num_queries / total_queries * 100.0) if total_queries > 0 else 0.0
        #     print(f"  {count:>4} candidates : {num_queries:>5} queries  ({pct:5.2f}%)")

    # 如果你想在外部继续使用这些统计量
    extra_stats = {
        "mean_rank": float(mean_rank),
        "median_rank": float(median_rank),
        "mean_decoys": float(mean_decoys),
    }

    return metrics, details, sim_stats, cand_count_dist, extra_stats


if __name__ == "__main__":
    # label_path = '/home/weiwentao/workspace/ms-pred/results/clip_train_canopus/split_1_rnd1/version_9/labels_confs.pkl.pkl'
    # candidate_path = '/home/weiwentao/workspace/ms-pred/results/clip_train_canopus/split_1_rnd1/version_9/cands_df_split_1_50.tsv.pkl'
    
    # label_path = '/home/weiwentao/workspace/ms-pred/results/predict_msg/results/clip_msg_adducts/split_msg_rnd1/version_6/labels_confs.pkl.pkl'
    # candidate_path = '/home/weiwentao/workspace/ms-pred/results/predict_msg/results/clip_msg_adducts/split_msg_rnd1/version_6/cands_df_test_formula_256.tsv.pkl'
    
    label_path = '/home/weiwentao/workspace/ms-pred/results/clip_msg_adducts_formula_decoys/split_msg_rnd1/version_0/labels_confs.pkl.pkl'
    candidate_path = '/home/weiwentao/workspace/ms-pred/results/clip_msg_adducts_formula_decoys/split_msg_rnd1/version_0/cands_df_test_formula_256_new_filtered.tsv.pkl'
    main(label_path, candidate_path, topk_max=20, verbose=True, normalize=False)
