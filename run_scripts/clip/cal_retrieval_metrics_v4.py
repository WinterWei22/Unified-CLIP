import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdFMCS import FindMCS
from multiprocessing import Pool, cpu_count
import time # 增加 time 模块用于计时，优化用户体验

# ==================== RDKit MCES/MCS 计算函数 ====================
def get_mces_distance(smiles1: str, smiles2: str) -> float:
    """
    计算两个分子之间的 MCES 距离。
    距离 = 1 - (MCS中的键数 / 两个分子中键数的最大值)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if not mol1 or not mol2:
        return -1.0 # 无法解析，返回错误值

    # 获取键数
    num_bonds1 = mol1.GetNumBonds()
    num_bonds2 = mol2.GetNumBonds()
    max_bonds = max(num_bonds1, num_bonds2)
    
    # 如果任一分子没有键，且它们相同，则距离为 0；否则为 1 (完全不匹配)
    if max_bonds == 0:
        return 0.0 if smiles1 == smiles2 else 1.0
        
    # 计算最大公共子图 (MCS)
    # timeout: 限制计算时间，防止复杂分子计算时间过长
    mcs_result = FindMCS([mol1, mol2], bondCompare=Chem.rdFMCS.BondCompare.CompareAny, timeout=5)
    
    # 获取 MCS 中的键数
    mcs_bonds = mcs_result.numBonds
    
    # 计算距离
    mces_distance = 1.0 - (mcs_bonds / max_bonds)
    
    return mces_distance

# **新增**：并行计算的包装函数
def _mces_worker(smiles_pair):
    """用于多进程池的 worker 函数，接收 (smiles1, smiles2) 元组并返回计算结果。"""
    s1, s2 = smiles_pair
    return get_mces_distance(s1, s2)

# ==================== 数据加载和准备函数 (与原版相同) ====================
def load_and_prepare_data(pkl_path):
    print(f"📂 正在加载数据: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. 获取 spec_names
    spec_names = data["spec_names"]
    
    # 2. 获取 flags 并处理 "True"/"False" 字符串
    raw_flags = data["flags"]
    flags = []
    for f in raw_flags:
        if f == 'True':
            flags.append(1)
        elif f == 'False':
            flags.append(0)
        else:
            try:
                flags.append(int(float(f)))
            except ValueError:
                flags.append(0)

    # 3. 获取相似度 (确保是 1D 数组)
    global_sim = np.array(data["cosine_similarity"]).flatten()
    local_sim = np.array(data["local_similarity"]).flatten()
    
    # 🌟 4. 获取 SMILES 字符串
    smiles_list = data.get("smiles", ["N/A"] * len(spec_names)) 

    # 5. 创建 DataFrame
    df = pd.DataFrame({
        "spec_name": spec_names,
        "flag": flags,
        "global_sim": global_sim,
        "local_sim": local_sim,
        "smiles": smiles_list
    })
    
    print(f"✅ 数据加载完成。总行数: {len(df)}")
    print(f"    Ground Truth (Target) 数量: {sum(flags)}")
    print(f"    Decoy 数量: {len(flags) - sum(flags)}")
    
    return df

# ==================== 指标计算函数 (与原版相同) ====================
def calculate_metrics(df, score_col, label):
    """
    排序并计算 Top-K 和统计指标
    """
    print(f"\n{'='*20} 评估模式: {label} {'='*20}")
    
    # --- 1. 排序 ---
    df['rank'] = df.groupby('spec_name')[score_col].rank(ascending=False, method='min')

    # --- 2. 区分 GT 和 Decoy ---
    gt_df = df[df['flag'] == 1]

    if gt_df.empty:
        print("❌ 错误: 数据中没有 flag 为 'True' (1) 的样本。")
        return None

    num_unique_specs = df['spec_name'].nunique()
    
    # --- 3. 指标 1: Top-K 成功率 ---
    print(f"\n📊 [1] Top-K 成功率 (Total Unique Specs: {num_unique_specs}):")
    best_gt_ranks = gt_df.groupby('spec_name')['rank'].min()
    
    for k in [1, 5, 10, 20]:
        hits = (best_gt_ranks <= k).sum()
        success_rate = hits / num_unique_specs
        print(f"  Top-{k:<2}: {success_rate:.4f}")

    # --- 4. 指标 2: 详细统计 ---
    avg_target_sim = gt_df[score_col].mean()
    std_target_sim = gt_df[score_col].std()
    
    decoy_df = df[df['flag'] == 0]
    avg_decoy_sim = decoy_df[score_col].mean() if not decoy_df.empty else 0.0
    std_decoy_sim = decoy_df[score_col].std() if not decoy_df.empty else 0.0
    
    avg_decoys = decoy_df.groupby('spec_name').size().mean() if not decoy_df.empty else 0.0
    
    avg_rank = best_gt_ranks.mean()
    std_rank = best_gt_ranks.std()

    print(f"\n📈 [2] 统计详情 ({label}):")
    print(f"  Avg spec-target similarity: {avg_target_sim:.4f}")
    print(f"  Std spec-target sim:        {std_target_sim:.4f}")
    print(f"  Avg spec-decoy similarity:  {avg_decoy_sim:.4f}")
    print(f"  Std spec-decoy sim:         {std_decoy_sim:.4f}")
    print(f"  Avg decoys per spectrum:    {avg_decoys:.2f}")
    print(f"  Avg best target rank:       {avg_rank:.2f}")
    print(f"  Std best target rank:       {std_rank:.2f}")
    
    return df

# ==================== MCES 距离计算函数 (并行版本) ====================
def calculate_mces(df, score_col, label):
    """
    使用多进程并行计算 Top-1 候选者和 Ground Truth 之间的平均 MCES 距离。
    """
    print(f"\n{'='*15} 🚀 MCES 评估 (并行 - {label}) {'='*15}")
    
    # --- 1. 数据准备 ---
    top1_df = df[df['rank'] == 1].sort_values(by=score_col, ascending=False).groupby('spec_name').head(1).copy()
    gt_df = df[df['flag'] == 1].drop_duplicates(subset=['spec_name'], keep='first').copy()
    
    top1_df.rename(columns={'smiles': 'top1_smiles'}, inplace=True)
    gt_df.rename(columns={'smiles': 'gt_smiles'}, inplace=True)
    
    mces_df = pd.merge(
        top1_df[['spec_name', 'top1_smiles']], 
        gt_df[['spec_name', 'gt_smiles']], 
        on='spec_name', 
        how='inner'
    )

    if mces_df.empty:
        print("⚠️ 警告: 无法将 Top-1 候选者和 Ground Truth 配对。跳过 MCES 计算。")
        return

    # --- 2. 准备并行任务 ---
    smiles_pairs = list(zip(mces_df['top1_smiles'], mces_df['gt_smiles']))
    num_processes = cpu_count() - 16 # 获取 CPU 核心数
    
    print(f"🔄 正在使用 {num_processes} 个核心并行计算 {len(smiles_pairs)} 个谱图的 MCES 距离...")
    start_time = time.time()
    
    # --- 3. 执行并行计算 ---
    try:
        # 使用 Pool 对象进行多进程并行计算
        with Pool(processes=num_processes) as pool:
            # map 函数将 smiles_pairs 中的每个元素传递给 _mces_worker 函数
            mces_results = pool.map(_mces_worker, smiles_pairs)
        
        mces_df['mces_distance'] = mces_results
        
    except Exception as e:
        print(f"❌ 错误: 并行计算失败 ({e})，尝试使用单进程计算。")
        mces_df['mces_distance'] = mces_df.apply(
            lambda row: get_mces_distance(row['top1_smiles'], row['gt_smiles']), 
            axis=1
        )
        
    end_time = time.time()
    print(f"⏱️ MCES 计算耗时: {end_time - start_time:.2f} 秒")

    # --- 4. 结果汇总 ---
    valid_mces = mces_df[mces_df['mces_distance'] >= 0]
    
    if valid_mces.empty:
        print("❌ 错误: 所有分子 SMILES 字符串均无法解析或计算失败。")
        return

    avg_mces = valid_mces['mces_distance'].mean()
    std_mces = valid_mces['mces_distance'].std()
    
    print(f"\n🧪 [3] 分子编辑距离 (MCES) 详情 ({label}):")
    print(f"  Avg Top-1/GT MCES Distance: {avg_mces:.4f}")
    print(f"  Std Top-1/GT MCES Distance: {std_mces:.4f}")
    print(f"  Total pairs computed:       {len(valid_mces)} / {len(mces_df)}")

# ==================== 主函数 (与原版相同) ====================
def main():
    # 为了在 Windows/macOS 上正确运行多进程，需要将 Pool 创建放在 if __name__ == "__main__": 保护块内
    # 但由于 Pool 位于 calculate_mces 内部，我们只需确保在运行脚本时主函数被调用即可。
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/labels_confs_instrumentation.pkl", 
                        help="Path to the .pkl file")
    parser.add_argument("--weight", type=float, default=0.5, 
                        help="Integration weight for Global Similarity (0.0 - 1.0)")
    args = parser.parse_args()

    pkl_path = Path(args.input)
    
    if not pkl_path.exists():
        candidates = list(Path(".").glob("*.pkl"))
        if candidates:
            print(f"⚠️ 指定文件未找到，将使用当前目录下的: {candidates[0]}")
            pkl_path = candidates[0]
        else:
            print(f"❌ 错误: 找不到文件 {pkl_path}")
            return

    # 1. 加载数据 (包含 SMILES)
    df = load_and_prepare_data(pkl_path)
    
    if df.empty:
        return

    # 2. 评估 全局相似性
    df_global = calculate_metrics(df.copy(), "global_sim", "Global Cosine Similarity")
    if df_global is not None:
        calculate_mces(df_global, "global_sim", "Global Sim")

    # # 3. 评估 局部相似性
    # df_local = calculate_metrics(df.copy(), "local_sim", "Local Similarity")
    # if df_local is not None:
    #     calculate_mces(df_local, "local_sim", "Local Sim")

    # # 4. 评估 集成结果
    # w = args.weight
    # df["integrated_sim"] = (w * df["global_sim"]) + ((1 - w) * df["local_sim"])
    # df_integrated = calculate_metrics(df.copy(), "integrated_sim", f"Integrated (Weight={w})")
    # if df_integrated is not None:
    #     calculate_mces(df_integrated, "integrated_sim", f"Integrated Sim (W={w})")


if __name__ == "__main__":
    # 在主执行块内调用 main()，这是进行多进程计算的 Python 最佳实践
    main()