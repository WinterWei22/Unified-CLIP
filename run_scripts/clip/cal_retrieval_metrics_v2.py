# import os
# import pickle
# import numpy as np
# from collections import defaultdict

# def load_pkl(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def l2_normalize(x, axis=-1, eps=1e-12):
#     norm = np.linalg.norm(x, axis=axis, keepdims=True)
#     return x / np.clip(norm, eps, None)

# def cosine_similarity(a, b, normalize=True):
#     """
#     a: (D,) or (N, D)
#     b: (M, D)
#     return: (M,) if a is (D,), else (N, M)
#     """
#     if a.ndim == 1:
#         a = a[None, :]
#     if normalize:
#         a = l2_normalize(a, axis=-1)
#         b = l2_normalize(b, axis=-1)
#     return a @ b.T

# def build_candidate_index(candidate_dict):
#     """
#     candidate_dict:
#       - spec_names: list[str]
#       - mol_emb: np.ndarray (T, D)
#     对于每个新的 spec_name，第一次出现为 target，其余为 decoys
#     """
#     spec_names = candidate_dict["spec_names"]
#     mol_emb = candidate_dict["mol_emb"]

#     index = defaultdict(lambda: {"embs": [], "target_idx": None})
#     seen_first = set()

#     for i, name in enumerate(spec_names):
#         if name not in seen_first:
#             index[name]["target_idx"] = len(index[name]["embs"])
#             seen_first.add(name)
#         index[name]["embs"].append(mol_emb[i])

#     for name in list(index.keys()):
#         index[name]["embs"] = np.stack(index[name]["embs"], axis=0)  # (K, D)

#     return dict(index)

# def evaluate_topk(label_dict, candidate_index, topk=(1,2,3,4,5,6,7,8,9,10), normalize=True):
#     """
#     返回:
#       - metrics: dict, top1..top10 accuracy
#       - details: 每个查询明细
#       - sim_stats: dict, 包含平均 target 相似度与平均 decoy 相似度
#     """
#     # 注意：这里按你最初说明，使用 "spec_name" 而非 "spec_names"
#     # 如果你的 label.pkl 确实是 "spec_names"，改回去即可
#     spec_names = label_dict.get("spec_name", None)
#     if spec_names is None:
#         # 如果文件里确实是 "spec_names"，兼容处理
#         spec_names = label_dict["spec_names"]
#     spec_emb = label_dict["spec_emb"]

#     correct_at_k = {k: 0 for k in topk}
#     total = 0

#     details = []

#     # 新增的统计量
#     target_sims = []
#     decoy_sims = []

#     for i, name in enumerate(spec_names):
#         query = spec_emb[i]  # (D,)

#         cand = candidate_index.get(name, None)
#         if cand is None:
#             details.append({
#                 "spec_name": name,
#                 "target_rank": None,
#                 "num_candidates": 0,
#                 "note": "no candidates"
#             })
#             continue

#         embs = cand["embs"]             # (K, D)
#         target_idx = cand["target_idx"] # int

#         if embs.shape[0] == 0 or target_idx is None:
#             details.append({
#                 "spec_name": name,
#                 "target_rank": None,
#                 "num_candidates": 0,
#                 "note": "invalid candidate set"
#             })
#             continue

#         sims = cosine_similarity(query, embs, normalize=normalize).reshape(-1)  # (K,)
#         order = np.argsort(-sims)
#         target_rank = int(np.where(order == target_idx)[0][0]) + 1

#         # 累计 Top-K
#         total += 1
#         for k in topk:
#             if target_rank <= k:
#                 correct_at_k[k] += 1

#         # 记录相似度统计
#         target_sims.append(float(sims[target_idx]))
#         # decoys: 除 target 外的所有
#         if embs.shape[0] > 1:
#             mask = np.ones_like(sims, dtype=bool)
#             mask[target_idx] = False
#             decoy_mean = float(sims[mask].mean())
#             decoy_sims.append(decoy_mean)

#         details.append({
#             "spec_name": name,
#             "target_rank": target_rank,
#             "num_candidates": int(embs.shape[0]),
#             "target_sim": float(sims[target_idx]),
#             "decoy_mean_sim": float(decoy_mean) if embs.shape[0] > 1 else None,
#         })

#     metrics = {f"top{k}": (correct_at_k[k] / total) if total > 0 else 0.0 for k in topk}

#     # 汇总平均相似度
#     sim_stats = {
#         "mean_target_similarity": float(np.mean(target_sims)) if len(target_sims) > 0 else 0.0,
#         "mean_decoy_similarity": float(np.mean(decoy_sims)) if len(decoy_sims) > 0 else 0.0,
#         "num_evaluated": total,
#         "num_with_decoys": len(decoy_sims),
#     }

#     return metrics, details, sim_stats

# def main(
#     label_pkl_path="label.pkl",
#     candidate_pkl_path="candidate.pkl",
#     topk_max=10,
#     verbose=True,
#     normalize=True
# ):
#     label_dict = load_pkl(label_pkl_path)
#     candidate_dict = load_pkl(candidate_pkl_path)

#     candidate_index = build_candidate_index(candidate_dict)

#     topk = tuple(range(1, topk_max + 1))
#     metrics, details, sim_stats = evaluate_topk(label_dict, candidate_index, topk=topk, normalize=normalize)

#     if verbose:
#         print("Evaluation results:")
#         for k in topk:
#             print(f"Top-{k} accuracy: {metrics[f'top{k}']:.4f}")
#         evaluated = sum(1 for d in details if d.get('target_rank') is not None)
#         skipped = sum(1 for d in details if d.get('target_rank') is None)
#         print(f"Total evaluated queries: {evaluated}")
#         if skipped > 0:
#             print(f"Skipped queries (no/invalid candidates): {skipped}")

#         print("\nSimilarity statistics:")
#         print(f"Mean target similarity: {sim_stats['mean_target_similarity']:.6f}")
#         print(f"Mean decoy similarity:  {sim_stats['mean_decoy_similarity']:.6f}")
#         print(f"Queries counted for mean target similarity: {sim_stats['num_evaluated']}")
#         print(f"Queries with decoys (for decoy mean):      {sim_stats['num_with_decoys']}")

#     return metrics, details, sim_stats


# if __name__ == "__main__":
#     # label_path = '/home/weiwentao/workspace/ms-pred/results/clip_train_canopus/split_1_rnd1/version_9/labels_confs.pkl.pkl'
#     # candidate_path = '/home/weiwentao/workspace/ms-pred/results/clip_train_canopus/split_1_rnd1/version_9/cands_df_split_1_50.tsv.pkl'
    
#     label_path = '/home/weiwentao/workspace/ms-pred/results/predict_msg/results/clip_msg_adducts/split_msg_rnd1/version_6/labels_confs.pkl.pkl'
#     candidate_path = '/home/weiwentao/workspace/ms-pred/results/predict_msg/results/clip_msg_adducts/split_msg_rnd1/version_6/cands_df_test_formula_256.tsv.pkl'
    

#     main(label_path, candidate_path, topk_max=10, verbose=True, normalize=False)
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import argparse

def compute_retrieval_performance(label_path, candidate_path, save_path=None, top_k_max=20):
    """
    Computes retrieval performance metrics and optionally saves them to a text file.
    
    Args:
        label_path (str): Path to the label pickle file.
        candidate_path (str): Path to the candidate pickle file.
        save_path (str, optional): Path to save the results as a .txt file. Defaults to None.
        top_k_max (int, optional): Maximum K for Top-K accuracy. Defaults to 20.
        
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    
    # Load pickle files
    with open(label_path, 'rb') as f:
        label = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        candidate = pickle.load(f)
    
    # === label data ===
    spec_names_label = label['spec_names']
    spec_embs = label['spec_emb']                            # (N, dim)

    # === candidate data ===
    spec_names_cand = candidate['spec_names']
    mol_embs        = candidate.get('mol_emb', None)
    flags           = candidate.get('flags')
    
    # Ensure flags are boolean
    flags = [f == 'True' if isinstance(f, str) else bool(f) for f in flags]
    candidate['flags'] = flags 
    
    if mol_embs is None:
        raise ValueError("candidate.pkl must contain 'mol_emb'.")
    if flags is None or len(flags) != len(spec_names_cand):
        raise ValueError("candidate.pkl must contain 'flags' as a list of length equal to the number of candidates.")

    # Group candidates by spec_name (global index -> list of indices)
    candidate_groups = defaultdict(list)
    for idx, spec_name in enumerate(spec_names_cand):
        candidate_groups[spec_name].append(idx)
    
    # Statistics containers
    valid_N = 0
    top_k_accuracies = {k: 0 for k in range(1, top_k_max + 1)}
    
    total_target_sim = 0.0
    total_decoy_sim  = 0.0
    total_targets = total_decoys = 0
    
    decoys_per_spec   = []
    best_target_ranks = []
    all_target_sims   = []
    all_decoy_sims    = []

    # Use tqdm to wrap the range for the progress bar
    for i in tqdm(range(len(spec_names_label)), desc="Calculating Retrieval Performance"):
        spec_name = spec_names_label[i]
        spec_emb  = spec_embs[i].reshape(1, -1)

        if spec_name not in candidate_groups:
            continue

        cand_global_idxs = candidate_groups[spec_name]  # global indices
        if not cand_global_idxs:
            continue

        valid_N += 1

        # Extract embeddings and flags for this spectrum
        cand_embs = mol_embs[cand_global_idxs]              # (n_cand, dim)
        cand_flags = [flags[global_idx] for global_idx in cand_global_idxs]  # list[bool]

        # Local indices of targets and decoys
        target_local_idxs = [loc for loc, is_target in enumerate(cand_flags) if is_target]
        decoy_local_idxs  = [loc for loc, is_target in enumerate(cand_flags) if not is_target]

        if not target_local_idxs:
            continue  # Skip if no ground truth target exists

        # Cosine similarities
        similarities = cosine_similarity(spec_emb, cand_embs)[0]   # (n_cand,)

        # Sort by descending similarity
        sorted_local_idxs = np.argsort(similarities)[::-1]

        # === Best target rank (smallest rank among all true targets) ===
        target_ranks_this = [np.where(sorted_local_idxs == loc)[0][0] + 1 
                            for loc in target_local_idxs]
        best_rank = min(target_ranks_this)
        best_target_ranks.append(best_rank)

        # === Top-k accuracy: if ANY target appears in top-k ===
        for k in range(1, top_k_max + 1):
            if any(loc in sorted_local_idxs[:k] for loc in target_local_idxs):
                top_k_accuracies[k] += 1

        # === Similarity statistics ===
        target_sims = similarities[target_local_idxs]
        all_target_sims.extend(target_sims)
        total_target_sim += target_sims.sum()
        total_targets += len(target_sims)

        decoys_this = len(decoy_local_idxs)
        decoys_per_spec.append(decoys_this)

        if decoys_this > 0:
            decoy_sims = similarities[decoy_local_idxs]
            all_decoy_sims.extend(decoy_sims)
            total_decoy_sim += decoy_sims.sum()
            total_decoys += decoys_this

    # === Final calculations ===
    for k in range(1, top_k_max + 1):
        top_k_accuracies[k] = top_k_accuracies[k] / valid_N * 100 if valid_N > 0 else 0.0

    avg_target_sim = total_target_sim / total_targets if total_targets > 0 else 0.0
    avg_decoy_sim  = total_decoy_sim  / total_decoys  if total_decoys  > 0 else 0.0

    avg_decoys      = np.mean(decoys_per_spec)   if decoys_per_spec   else 0.0
    avg_best_rank   = np.mean(best_target_ranks) if best_target_ranks else 0.0
    std_target_sim  = np.std(all_target_sims)    if all_target_sims     else 0.0
    std_decoy_sim   = np.std(all_decoy_sims)     if all_decoy_sims      else 0.0

    # === Prepare Output String ===
    output_lines = []
    output_lines.append(f"Total spectra in label file     : {len(spec_names_label)}")
    output_lines.append(f"Valid spectra (with candidates): {valid_N}")
    output_lines.append("\nTop-k Retrieval Accuracies (%):")
    for k in range(1, top_k_max + 1):
        output_lines.append(f"  Top-{k:2d}: {top_k_accuracies[k]:6.2f}%")

    output_lines.append(f"\nAvg spec-target similarity : {avg_target_sim:.4f}")
    output_lines.append(f"Avg spec-decoy  similarity : {avg_decoy_sim:.4f}")
    output_lines.append(f"Avg decoys per spectrum    : {avg_decoys:.2f}")
    output_lines.append(f"Avg best target rank       : {avg_best_rank:.2f}")
    output_lines.append(f"Std spec-target sim        : {std_target_sim:.4f}")
    output_lines.append(f"Std spec-decoy  sim        : {std_decoy_sim:.4f}")
    
    final_output_str = "\n".join(output_lines)

    # === Print results to console ===
    print("\n" + final_output_str)

    # === Save results to TXT if path provided ===
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(final_output_str)
            print(f"\n[INFO] Results successfully saved to: {save_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save results to {save_path}: {e}")

    return {
        "valid_N": valid_N,
        "topk_accuracy": top_k_accuracies,
        "avg_target_sim": avg_target_sim,
        "avg_decoy_sim": avg_decoy_sim,
        "avg_decoys_per_spec": avg_decoys,
        "avg_best_target_rank": avg_best_rank,
        "std_target_sim": std_target_sim,
        "std_decoy_sim": std_decoy_sim,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cal retrieval metrics v2')
    
    parser.add_argument(
        '--basic_path', 
        type=str, 
        required=True,
        help='result path'
    )
    args = parser.parse_args()

    # basic_path = '/home/weiwentao/workspace/ms-pred/results/clip_msg_adducts_frozen_dreams_ce/split_msg_rnd1/version_2'
    basic_path = args.basic_path
    save_path = f'{basic_path}/results.txt'
    label_path = f'{basic_path}/embeddings.pkl'
    candidate_path = f'{basic_path}/cands_df_test_formula_256_new_filtered_filteredagain.tsv.pkl'

    metrics = compute_retrieval_performance(
        label_path=label_path, 
        candidate_path=candidate_path, 
        save_path=save_path,
        top_k_max=100
    )