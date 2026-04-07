"""
Analyze relationship between MAGMa fragment quality and model performance.
- MAGMa is_observed count/ratio vs global cosine similarity
- MAGMa file existence vs global cosine similarity
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# Paths
MAGMA_DIR = Path("/home/weiwentao/workspace/ms-pred/data/spec_datasets/msg/magma_outputs/magma_tree")
EMB_DIR = Path("results/clip_msg_local/split_msg_rnd1/version_16")
DATA_DIR = Path("data/spec_datasets/msg")
SPLIT_FILE = DATA_DIR / "splits" / "split_msg.tsv"
LABELS_FILE = DATA_DIR / "labels.tsv"
OUTPUT_DIR = Path("results/rebuttal_noise_ft/magma_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings(pkl_path):
    """Load per-sample embeddings and compute cosine similarity."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = {}
    for spec_name, (spec_emb, mol_emb) in data.items():
        spec_emb = spec_emb / (np.linalg.norm(spec_emb) + 1e-8)
        mol_emb = mol_emb / (np.linalg.norm(mol_emb) + 1e-8)
        cos_sim = np.dot(spec_emb, mol_emb)
        results[spec_name] = float(cos_sim)
    return results


def load_magma_stats(spec_names):
    """Load MAGMa stats for each spec_name."""
    stats = {}
    for name in spec_names:
        magma_file = MAGMA_DIR / f"{name}.json"
        if not magma_file.exists():
            stats[name] = {
                'has_magma': False,
                'total_frags': 0,
                'observed_frags': 0,
                'observed_ratio': 0.0,
            }
        else:
            with open(magma_file) as f:
                tree = json.load(f)
            frags = tree.get('frags', {})
            total = len(frags)
            observed = sum(1 for v in frags.values() if v.get('is_observed', False))
            stats[name] = {
                'has_magma': True,
                'total_frags': total,
                'observed_frags': observed,
                'observed_ratio': observed / total if total > 0 else 0.0,
            }
    return stats


def get_splits(labels_file, split_file):
    """Get train/val/test split indices."""
    df = pd.read_csv(labels_file, sep='\t')
    spec_names = df['spec'].values
    split_df = pd.read_csv(split_file, sep='\t')

    split_map = {}
    for _, row in split_df.iterrows():
        split_map[row['name']] = row['split']

    train_names = [s for s in spec_names if split_map.get(s) == 'train']
    val_names = [s for s in spec_names if split_map.get(s) == 'val']
    test_names = [s for s in spec_names if split_map.get(s) == 'test']
    return train_names, val_names, test_names


def analyze_and_plot(similarities, magma_stats, split_name, output_dir):
    """Analyze and create plots."""
    # Build DataFrame
    rows = []
    for name in similarities:
        if name in magma_stats:
            row = {
                'spec_name': name,
                'cosine_similarity': similarities[name],
                **magma_stats[name]
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  Analysis for {split_name} set ({len(df)} samples)")
    print(f"{'='*60}")

    # ========== Analysis 1: MAGMa file existence vs similarity ==========
    has_magma = df[df['has_magma'] == True]
    no_magma = df[df['has_magma'] == False]

    print(f"\n--- MAGMa File Existence vs Cosine Similarity ---")
    print(f"  Has MAGMa:  N={len(has_magma):>6}, mean_sim={has_magma['cosine_similarity'].mean():.4f} ± {has_magma['cosine_similarity'].std():.4f}")
    print(f"  No MAGMa:   N={len(no_magma):>6}, mean_sim={no_magma['cosine_similarity'].mean():.4f} ± {no_magma['cosine_similarity'].std():.4f}")

    if len(no_magma) > 0 and len(has_magma) > 0:
        from scipy import stats as sp_stats
        t_stat, p_val = sp_stats.ttest_ind(has_magma['cosine_similarity'], no_magma['cosine_similarity'])
        print(f"  t-test: t={t_stat:.4f}, p={p_val:.2e}")

    # Plot 1: Box plot of similarity by magma existence
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    data_to_plot = [has_magma['cosine_similarity'].values, no_magma['cosine_similarity'].values]
    labels_plot = [f'Has MAGMa\n(n={len(has_magma)})', f'No MAGMa\n(n={len(no_magma)})']
    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][1].set_facecolor('#F44336')
    ax.set_ylabel('Global Cosine Similarity')
    ax.set_title(f'{split_name}: MAGMa File Existence vs Similarity')
    plt.tight_layout()
    plt.savefig(output_dir / f'{split_name}_magma_existence_boxplot.png', dpi=150)
    plt.close()

    # ========== Analysis 2: is_observed count/ratio vs similarity ==========
    magma_df = df[df['has_magma'] == True].copy()

    if len(magma_df) > 0:
        print(f"\n--- is_observed Count vs Cosine Similarity (MAGMa samples only) ---")
        print(f"  Total frags:    mean={magma_df['total_frags'].mean():.1f}, median={magma_df['total_frags'].median():.0f}")
        print(f"  Observed frags: mean={magma_df['observed_frags'].mean():.1f}, median={magma_df['observed_frags'].median():.0f}")
        print(f"  Observed ratio: mean={magma_df['observed_ratio'].mean():.3f}, median={magma_df['observed_ratio'].median():.3f}")

        # Correlation
        corr_count = magma_df['cosine_similarity'].corr(magma_df['observed_frags'])
        corr_ratio = magma_df['cosine_similarity'].corr(magma_df['observed_ratio'])
        corr_total = magma_df['cosine_similarity'].corr(magma_df['total_frags'])
        print(f"\n  Pearson correlation with cosine_similarity:")
        print(f"    observed_count:  r={corr_count:.4f}")
        print(f"    observed_ratio:  r={corr_ratio:.4f}")
        print(f"    total_frags:     r={corr_total:.4f}")

        # Bin by observed_ratio and compute mean similarity
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        magma_df['ratio_bin'] = pd.cut(magma_df['observed_ratio'], bins=bins, right=False)
        bin_stats = magma_df.groupby('ratio_bin').agg(
            count=('cosine_similarity', 'count'),
            mean_sim=('cosine_similarity', 'mean'),
            std_sim=('cosine_similarity', 'std'),
        )
        print(f"\n  Observed Ratio Bins:")
        print(f"  {'Bin':<12} {'Count':>6} {'Mean Sim':>10} {'Std Sim':>10}")
        for idx, row in bin_stats.iterrows():
            print(f"  {str(idx):<12} {row['count']:>6.0f} {row['mean_sim']:>10.4f} {row['std_sim']:>10.4f}")

        # Plot 2: Scatter plot of observed_ratio vs similarity
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 2a: observed_ratio vs similarity
        ax = axes[0]
        ax.scatter(magma_df['observed_ratio'], magma_df['cosine_similarity'],
                   alpha=0.1, s=5, c='steelblue')
        # Add binned means
        bin_centers = [(b1+b2)/2 for b1, b2 in zip(bins[:-1], bins[1:])]
        ax.plot(bin_centers, bin_stats['mean_sim'].values, 'r-o', linewidth=2, markersize=6, label='Bin mean')
        ax.set_xlabel('Observed Fragment Ratio')
        ax.set_ylabel('Global Cosine Similarity')
        ax.set_title(f'{split_name}: Observed Ratio vs Similarity')
        ax.legend()

        # 2b: observed_count vs similarity
        ax = axes[1]
        count_bins = [0, 2, 5, 10, 15, 20, 30, 50, 100]
        magma_df['count_bin'] = pd.cut(magma_df['observed_frags'], bins=count_bins, right=False)
        count_bin_stats = magma_df.groupby('count_bin').agg(
            count=('cosine_similarity', 'count'),
            mean_sim=('cosine_similarity', 'mean'),
        )
        count_centers = [(b1+b2)/2 for b1, b2 in zip(count_bins[:-1], count_bins[1:])]
        valid_mask = count_bin_stats['count'] > 10
        ax.bar(range(len(count_bin_stats)), count_bin_stats['mean_sim'].values,
               color=['steelblue' if v else 'lightgray' for v in valid_mask])
        ax.set_xticks(range(len(count_bin_stats)))
        ax.set_xticklabels([str(idx) for idx in count_bin_stats.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Global Cosine Similarity')
        ax.set_title(f'{split_name}: Observed Count vs Similarity')

        # Add count labels
        for i, (idx, row) in enumerate(count_bin_stats.iterrows()):
            ax.text(i, row['mean_sim'] + 0.005, f'n={int(row["count"])}',
                    ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_observed_vs_similarity.png', dpi=150)
        plt.close()

        # Plot 3: Histogram of observed_ratio colored by similarity quartiles
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        q25 = magma_df['cosine_similarity'].quantile(0.25)
        q75 = magma_df['cosine_similarity'].quantile(0.75)
        low_sim = magma_df[magma_df['cosine_similarity'] < q25]
        high_sim = magma_df[magma_df['cosine_similarity'] > q75]
        ax.hist(low_sim['observed_ratio'], bins=20, alpha=0.6, label=f'Low sim (< {q25:.2f})', color='red', density=True)
        ax.hist(high_sim['observed_ratio'], bins=20, alpha=0.6, label=f'High sim (> {q75:.2f})', color='green', density=True)
        ax.set_xlabel('Observed Fragment Ratio')
        ax.set_ylabel('Density')
        ax.set_title(f'{split_name}: Observed Ratio Distribution by Similarity Quartile')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{split_name}_ratio_by_sim_quartile.png', dpi=150)
        plt.close()

    return df


def main():
    print("Loading splits...")
    train_names, val_names, test_names = get_splits(LABELS_FILE, SPLIT_FILE)
    print(f"  Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # Load embeddings
    print("Loading validation embeddings...")
    val_sims = load_embeddings(EMB_DIR / "valid-spec-mol-embeddings.pkl")
    print(f"  Loaded {len(val_sims)} val embeddings")

    print("Loading training embeddings...")
    train_sims = load_embeddings(EMB_DIR / "train-spec-mol-embeddings.pkl")
    print(f"  Loaded {len(train_sims)} train embeddings")

    # Load MAGMa stats
    all_names = list(set(list(val_sims.keys()) + list(train_sims.keys())))
    print(f"Loading MAGMa stats for {len(all_names)} samples...")
    magma_stats = load_magma_stats(all_names)

    # Analyze
    val_df = analyze_and_plot(val_sims, magma_stats, "Validation", OUTPUT_DIR)
    train_df = analyze_and_plot(train_sims, magma_stats, "Training", OUTPUT_DIR)

    # Save raw data
    val_df.to_csv(OUTPUT_DIR / "val_analysis.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "train_analysis.csv", index=False)

    print(f"\nAll plots and data saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
