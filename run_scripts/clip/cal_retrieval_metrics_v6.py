import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
from io import StringIO
import contextlib

log_file = None 

def tee_print(*args, **kwargs):
    text = StringIO()
    with contextlib.redirect_stdout(text):
        print(*args, **kwargs)
    output = text.getvalue()
    sys.stdout.write(output)
    if log_file:
        log_file.write(output)

def load_and_prepare_data(pkl_path):
    tee_print(f"📂 正在加载数据: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 提取必要字段
    spec_names = data["spec_names"]
    # 尝试获取 smiles，如果不存在则填充占位符
    smiles_list = data.get("smiles", ["N/A"] * len(spec_names)) 
    
    # 处理 flags
    raw_flags = data["flags"]
    flags = [1 if str(f).strip().lower() == 'true' else 0 for f in raw_flags]

    # 仅获取全局相似度
    global_sim = np.array(data["cosine_similarity"]).flatten()

    df = pd.DataFrame({
        "spec_name": spec_names,
        "smiles": smiles_list,
        "flag": flags,
        "global_sim": global_sim
    })
    
    tee_print(f"✅ 数据加载完成。总行数: {len(df)}")
    return df

def process_global_top1(df, output_tsv):
    """
    基于 global_sim 排序，计算指标并保存 Top-1 SMILES
    """
    tee_print(f"\n{'='*20} 评估模式: Global Cosine Similarity {'='*20}")
    
    # 排序：按 spec_name 分组，按 global_sim 降序排列
    # method='first' 确保每个 spec 只有一个 Top-1
    df['rank'] = df.groupby('spec_name')['global_sim'].rank(ascending=False, method='first')

    # --- 1. 计算 Top-K 成功率 ---
    gt_df = df[df['flag'] == 1]
    num_unique_specs = df['spec_name'].nunique()
    best_gt_ranks = gt_df.groupby('spec_name')['rank'].min()
    
    tee_print(f"\n📊 Top-K 成功率 (Total Unique Specs: {num_unique_specs}):")
    for k in [1, 5, 10, 20]:
        hits = (best_gt_ranks <= k).sum()
        tee_print(f"  Top-{k:<2}: {hits / num_unique_specs:.4f}")

    # --- 2. 提取并保存 Top-1 SMILES ---
    # 筛选排名第一的行
    top1_df = df[df['rank'] == 1][['spec_name', 'smiles']]
    
    # 保存为 .tsv
    top1_df.to_csv(output_tsv, sep='\t', index=False, header=['spec', 'smiles'])
    tee_print(f"\n💾 Top-1 预测结果已保存至: {output_tsv}")

def main():
    global log_file
    
    parser = argparse.ArgumentParser(description="仅基于 Global Similarity 评估并保存 Top-1 SMILES。")
    parser.add_argument("--input", type=str, default="results/labels_confs_instrumentation.pkl")
    args = parser.parse_args()

    pkl_path = Path(args.input)
    log_path = pkl_path.parent / 'evaluation_global_log.txt'
    tsv_path = pkl_path.parent / f"{pkl_path.stem}_global_top1.tsv"
    
    if not pkl_path.exists():
        sys.stderr.write(f"❌ 错误: 找不到输入文件 {pkl_path}\n")
        return

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        
        # 1. 加载数据
        df = load_and_prepare_data(pkl_path)
        
        if df.empty:
            return

        # 2. 处理全局相似度并导出
        process_global_top1(df, tsv_path)
            
        sys.stderr.write(f"✅ 任务完成。TSV: {tsv_path}\n")

    except Exception as e:
        sys.stderr.write(f"发生错误: {e}\n")
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()