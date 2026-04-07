import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
from io import StringIO
import contextlib # 导入 contextlib 库

# 全局变量，用于存储打开的文件对象，便于在 tee_print 中写入
# ⚠️ 注意: 在实际生产代码中，应避免使用全局变量，但这里为了保持代码结构简洁，采用此方法
# 更好的做法是将文件对象作为参数传递给 calculate_metrics
log_file = None 

def tee_print(*args, **kwargs):
    """
    将内容同时打印到标准输出 (sys.stdout) 和日志文件 (log_file)
    """
    text = StringIO()
    # 捕获 print 的输出到 text 缓冲区
    with contextlib.redirect_stdout(text):
        print(*args, **kwargs)
    
    output = text.getvalue()
    
    # 1. 打印到控制台
    sys.stdout.write(output)
    
    # 2. 写入到文件
    if log_file:
        log_file.write(output)

def load_and_prepare_data(pkl_path):
    tee_print(f"📂 正在加载数据: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. 获取 spec_names
    spec_names = data["spec_names"]
    
    # 2. 获取 flags 并处理 "True"/"False" 字符串
    raw_flags = data["flags"]
    flags = []
    for f in raw_flags:
        # 直接根据字符串值进行转换
        if f == 'True':
            flags.append(1)
        elif f == 'False':
            flags.append(0)
        else:
            # 兼容处理 '1' 或 '0' 形式的字符串，或者直接是数字的情况
            try:
                # 使用 np.float64 或 float() 处理可能的数字字符串
                flags.append(int(float(f)))
            except ValueError:
                flags.append(0) # 无法识别时默认为 Decoy

    # 3. 获取相似度 (确保是 1D 数组)
    global_sim = np.array(data["cosine_similarity"]).flatten()
    local_sim = np.array(data["local_similarity"]).flatten()

    # 4. 创建 DataFrame
    df = pd.DataFrame({
        "spec_name": spec_names,
        "flag": flags,
        "global_sim": global_sim,
        "local_sim": local_sim
    })
    
    tee_print(f"✅ 数据加载完成。总行数: {len(df)}")
    tee_print(f"   Ground Truth (Target) 数量: {sum(flags)}")
    tee_print(f"   Decoy 数量: {len(flags) - sum(flags)}")
    
    return df

def calculate_metrics(df, score_col, label):
    """
    排序并计算 Top-K 和统计指标
    """
    tee_print(f"\n{'='*20} 评估模式: {label} {'='*20}")
    
    # --- 1. 排序 ---
    # 对每个 spec_name 内部，根据分数降序排列
    # method='min': 并列时取最好名次
    df['rank'] = df.groupby('spec_name')[score_col].rank(ascending=False, method='min')

    # --- 2. 区分 GT 和 Decoy ---
    gt_df = df[df['flag'] == 1]
    decoy_df = df[df['flag'] == 0]
    
    if gt_df.empty:
        tee_print("❌ 错误: 数据中没有 flag 为 'True' (1) 的样本。")
        return

    num_unique_specs = df['spec_name'].nunique()
    
    # --- 3. 指标 1: Top-K 成功率 ---
    tee_print(f"\n📊 [1] Top-K 成功率 (Total Unique Specs: {num_unique_specs}):")
    # 计算每个谱图的最佳 GT 排名（防止一个谱图有多个正确答案的情况，虽然通常只有一个）
    best_gt_ranks = gt_df.groupby('spec_name')['rank'].min()
    
    for k in range(1, 21):
        hits = (best_gt_ranks <= k).sum()
        success_rate = hits / num_unique_specs
        # 只打印部分关键 K 值以保持输出整洁
        if k in [1, 5, 10, 20]:
            tee_print(f"  Top-{k:<2}: {success_rate:.4f}")

    # --- 4. 指标 2: 详细统计 ---
    avg_target_sim = gt_df[score_col].mean()
    std_target_sim = gt_df[score_col].std()
    
    avg_decoy_sim = decoy_df[score_col].mean()
    std_decoy_sim = decoy_df[score_col].std()
    
    avg_decoys = decoy_df.groupby('spec_name').size().mean()
    
    avg_rank = best_gt_ranks.mean()
    std_rank = best_gt_ranks.std()

    tee_print(f"\n📈 [2] 统计详情 ({label}):")
    tee_print(f"  Avg spec-target similarity: {avg_target_sim:.4f}")
    tee_print(f"  Std spec-target sim:        {std_target_sim:.4f}")
    tee_print(f"  Avg spec-decoy similarity:  {avg_decoy_sim:.4f}")
    tee_print(f"  Std spec-decoy sim:         {std_decoy_sim:.4f}")
    tee_print(f"  Avg decoys per spectrum:    {avg_decoys:.2f}")
    tee_print(f"  Avg best target rank:       {avg_rank:.2f}")
    tee_print(f"  Std best target rank:       {std_rank:.2f}")

def main():
    global log_file # 声明使用全局变量
    
    parser = argparse.ArgumentParser(description="评估基于相似度得分的匹配性能，同时输出到文件和控制台。")
    parser.add_argument("--input", type=str, default="results/labels_confs_instrumentation.pkl", 
                        help="Path to the .pkl file containing similarity scores and flags.")
    parser.add_argument("--weight", type=float, default=0.5, 
                        help="Integration weight for Global Similarity (0.0 - 1.0).")
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                        help="Path to the output .txt file for results.")
    args = parser.parse_args()

    pkl_path = Path(args.input)
    output_path = pkl_path.parent / 'results.txt'
    
    # 简单的文件存在性检查
    if not pkl_path.exists():
        candidates = list(Path(".").glob("*.pkl"))
        if candidates:
            sys.stderr.write(f"⚠️ 指定输入文件未找到，将使用当前目录下的: {candidates[0]}\n")
            pkl_path = candidates[0]
        else:
            sys.stderr.write(f"❌ 错误: 找不到输入文件 {pkl_path}\n")
            return

    # 打开文件并赋值给全局变量 log_file
    try:
        log_file = open(output_path, "w", encoding="utf-8")
        sys.stderr.write(f"🚀 Evaluating: {output_path}\n")

        # 1. 加载数据
        df = load_and_prepare_data(pkl_path)
        
        # 如果数据为空则退出
        if df.empty:
            return

        # 2. 评估 全局相似性
        calculate_metrics(df, "global_sim", "Global Cosine Similarity")

        # 3. 评估 局部相似性
        calculate_metrics(df, "local_sim", "Local Similarity")

        # 4. 评估 集成结果
        w = args.weight
        df["integrated_sim"] = (w * df["global_sim"]) + ((1 - w) * df["local_sim"])
        calculate_metrics(df, "integrated_sim", f"Integrated (Weight={w})")
            
        sys.stderr.write(f"✅ 评估完成，结果已保存到 {output_path}\n")

    except Exception as e:
        sys.stderr.write(f"发生错误: {e}\n")
    finally:
        # 确保文件关闭
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()