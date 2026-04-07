import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
from io import StringIO
import contextlib 

# 全局变量，用于存储打开的文件对象
log_file = None 

def tee_print(*args, **kwargs):
    """
    将内容同时打印到标准输出 (sys.stdout) 和日志文件 (log_file)
    """
    text = StringIO()
    with contextlib.redirect_stdout(text):
        print(*args, **kwargs)
    output = text.getvalue()
    sys.stdout.write(output)
    if log_file:
        log_file.write(output)

def load_and_prepare_data(pkl_path):
    """加载相似度得分文件"""
    tee_print(f"📂 正在加载相似度数据: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. 获取 spec_names
    spec_names = data["spec_names"]
    
    # 2. 获取 flags
    raw_flags = data["flags"]
    flags = []
    for f in raw_flags:
        if str(f) == 'True':
            flags.append(1)
        elif str(f) == 'False':
            flags.append(0)
        else:
            try:
                flags.append(int(float(f)))
            except ValueError:
                flags.append(0) 

    # 3. 获取相似度
    global_sim = np.array(data["cosine_similarity"]).flatten()
    local_sim = np.array(data["local_similarity"]).flatten()

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        "spec_name": spec_names,
        "flag": flags,
        "global_sim": global_sim,
        "local_sim": local_sim
    })
    
    tee_print(f"✅ 原始数据加载完成。行数: {len(df)}")
    return df

def map_specs_to_inchikey(df, map_pkl_path):
    """
    加载映射文件并将 spec_name 映射到 inchikey
    """
    tee_print(f"🔗 正在加载映射文件: {map_pkl_path}")
    
    with open(map_pkl_path, "rb") as f:
        # map_data结构: {inchikey: [spec1, spec2, ...]}
        inchikey_to_specs = pickle.load(f)

    # 1. 反转映射关系: spec_name -> inchikey
    spec_to_inchikey = {}
    for inchikey, specs in inchikey_to_specs.items():
        if isinstance(specs, list):
            for spec in specs:
                spec_to_inchikey[spec] = inchikey
        else:
            # 防御性编程：处理非 list 情况
            spec_to_inchikey[specs] = inchikey
            
    tee_print(f"   映射字典构建完成，包含 {len(spec_to_inchikey)} 个谱图映射关系")

    # 2. 将映射应用到 DataFrame
    df['inchikey'] = df['spec_name'].map(spec_to_inchikey)

    # 3. 检查是否有未映射的数据
    missing_count = df['inchikey'].isna().sum()
    if missing_count > 0:
        tee_print(f"⚠️ 警告: 有 {missing_count} 行数据未能找到对应的 Inchikey，将被移除。")
        df = df.dropna(subset=['inchikey'])
    
    tee_print(f"✅ 映射完成。有效数据行数: {len(df)}")
    return df

def calculate_weighted_inchikey_score(df):
    """
    按 Inchikey 分组，计算加权平均分
    公式: Weighted_Score = sum(Score_i * Weight_i) / sum(Weight_i)
    这里 Score_i = global_sim, Weight_i = global_sim
    """
    tee_print(f"\n⚙️ 正在计算分子层面(Inchikey)加权平均分...")

    # 1. 计算加权项
    # 分子部分: sum(sim * sim)
    # 分母部分: sum(sim)
    
    # 预计算权重乘积
    df['weighted_val'] = df['global_sim'] * df['global_sim']

    # 2. 分组聚合
    grouped = df.groupby('inchikey').agg(
        # 只要组内有一个是 GT (flag=1)，这个 inchikey 就是 GT
        flag=('flag', 'max'),
        sum_weighted_val=('weighted_val', 'sum'),
        sum_weight=('global_sim', 'sum')
    ).reset_index()

    # 3. 计算最终得分
    # 避免除以 0
    grouped['inchikey_score'] = np.where(
        grouped['sum_weight'] > 0,
        grouped['sum_weighted_val'] / grouped['sum_weight'],
        0.0
    )

    tee_print(f"✅ 分子打分完成。唯一分子数: {len(grouped)}")
    
    # 返回处理好的 DataFrame，重命名列以适配 calculate_metrics
    result_df = grouped[['inchikey', 'flag', 'inchikey_score']].rename(
        columns={'inchikey': 'spec_name', 'inchikey_score': 'score'}
    )
    return result_df

def calculate_metrics(df, score_col, label):
    """
    排序并计算 Top-K 和统计指标 (通用版)
    """
    tee_print(f"\n{'='*20} 评估模式: {label} {'='*20}")
    
    # --- 1. 排序 ---
    # 对每个 spec_name (这里实际上可能是 inchikey) 内部，根据分数降序排列
    # 注意：如果输入已经是每个 inchikey 一行数据，rank 其实没有意义，
    # 但为了兼容 calculate_metrics 的通用逻辑（通常是 query vs candidates），
    # 这里我们假设 df 已经包含了所有 candidates。
    # ⚠️ 关键修正：在分子检索任务中，通常是对 "一个查询" (Query) 检索 "多个库分子" (Candidates)。
    # 如果您的数据结构是 [Query_Spec, Candidate_Molecule, Score]，那么上面的聚合是对 Candidate_Molecule 进行打分。
    # 如果您的数据结构是 [Query_Inchikey, Candidate_Inchikey, Score]，我们需要明确 "分组列" 是什么。
    
    # 假设：目前的 df 每一行代表一个独立的实体（分子），我们需要对它们进行全局排序或分组排序？
    # 通常检索评估是：对于每个 Query，有多个 Candidates。
    # 如果该脚本是评估 "Identification Rate"（即在这个列表中，正确的分子排在第几位？），
    # 那么通常需要一个 "Query ID"。
    # 由于原始代码是 groupby('spec_name')，这意味着原始数据包含多个 Query，每个 Query 下有多个 Candidates。
    
    # 在 Inchikey 聚合后，spec_name (原 Query ID) 丢失了。
    # 逻辑修正：
    # 如果这是 "谱图检索"：一个谱图 -> 多个候选分子。聚合后 -> 一个谱图 -> 多个候选分子（带加权分）。
    # 如果这是 "分子检索"：一个分子(多谱图) -> 多个候选分子。
    
    # 根据您的描述 "将每一个inchikey对应的spec作为一组...加权平均"，
    # 这听起来像是把多个谱图的信息融合成一个分子的 Fingerprint/Embedding，然后去检索。
    # 或者，仅仅是评估：在此次实验的所有结果中，正确分子的得分排名情况。
    
    # 鉴于原始代码逻辑，我们将 "spec_name" 作为分组键。
    # 在聚合后，如果我们将 inchikey 视为新的 ID，我们需要确保数据中依然保留了 "Query" 的概念。
    # 如果只是单纯对所有 inchikey 混在一起排序，看 GT 排在哪里，那就不需要 groupby。
    
    # 下面保持原始逻辑的兼容性：假设输入 df 包含了多个 Query 的结果。
    # 如果聚合后 df 变成了 [Inchikey1, Score1], [Inchikey2, Score2]... 
    # 且这些全部属于同一个实验，那么我们是对整个数据集进行排序。
    
    # 排序：分数降序
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    # 生成排名 (全局排名)
    df['rank'] = df[score_col].rank(ascending=False, method='min')

    # --- 2. 区分 GT 和 Decoy ---
    gt_df = df[df['flag'] == 1]
    
    if gt_df.empty:
        tee_print("❌ 错误: 数据中没有 flag 为 'True' (1) 的样本。")
        return

    total_candidates = len(df)
    
    # --- 3. 指标 1: Top-K 成功率 ---
    # 这里逻辑变为：在所有候选分子中，正确的分子是否排在前面？
    tee_print(f"\n📊 [1] Top-K 成功率 (Total Candidates: {total_candidates}):")
    
    # 获取所有 GT 的排名
    gt_ranks = gt_df['rank']
    total_gts = len(gt_ranks)
    
    tee_print(f"   Total Ground Truths: {total_gts}")

    for k in range(1, 21):
        hits = (gt_ranks <= k).sum()
        success_rate = hits / total_gts # 注意分母是 GT 的数量
        if k in [1, 5, 10, 20]:
            tee_print(f"  Top-{k:<2}: {success_rate:.4f}")

    # --- 4. 统计 ---
    avg_rank = gt_ranks.mean()
    tee_print(f"\n📈 [2] 统计详情:")
    tee_print(f"  Avg Target Rank: {avg_rank:.2f}")


def main():
    global log_file 
    
    parser = argparse.ArgumentParser(description="评估分子层面(Inchikey)的加权平均得分性能。")
    parser.add_argument("--input", type=str, default="results/labels_confs_instrumentation.pkl", 
                        help="Path to the .pkl file containing similarity scores (original specs).")
    # 新增参数：映射文件
    parser.add_argument("--map", type=str, default="/home/weiwentao/workspace/JESTR1/data/MassSpecGym/inchi_to_id_dict.pkl",
                        help="Path to the .pkl file containing inchikey->list(spec_name) mapping.")
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                        help="Path to the output .txt file.")
    args = parser.parse_args()

    pkl_path = Path(args.input)
    map_path = Path(args.map)
    output_path = pkl_path.parent / args.output # 输出在同级目录
    
    if not pkl_path.exists():
        sys.stderr.write(f"❌ 错误: 找不到输入文件 {pkl_path}\n")
        return
    if not map_path.exists():
        sys.stderr.write(f"❌ 错误: 找不到映射文件 {map_path}\n")
        return

    try:
        log_file = open(output_path, "w", encoding="utf-8")
        sys.stderr.write(f"🚀 Evaluating: {output_path}\n")

        # 1. 加载原始数据
        df = load_and_prepare_data(pkl_path)
        if df.empty: return

        # 2. 映射 Spec -> Inchikey
        df = map_specs_to_inchikey(df, map_path)
        if df.empty: return

        # 3. 计算分子层面加权得分
        # 这会将多行 spec 数据压缩成一行 inchikey 数据
        inchikey_df = calculate_weighted_inchikey_score(df)

        # 4. 评估结果
        calculate_metrics(inchikey_df, score_col="score", label="Weighted Inchikey Score")
            
        sys.stderr.write(f"✅ 评估完成，结果已保存到 {output_path}\n")

    except Exception as e:
        sys.stderr.write(f"发生错误: {e}\n")
        import traceback
        traceback.print_exc() # 打印详细报错方便调试
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()