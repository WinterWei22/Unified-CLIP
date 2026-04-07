import os
import json
import logging
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # 可减少 RDKit 控制台噪音

import dgl
# import torch, numpy as np  # 如果不需要可移除

from joblib import Parallel, delayed

# 假定你已有以下对象/模块
# from your_module import TreeProcessor, common
# common.ion2onehot_pos: Dict[str, Any]
# tree_processor: TreeProcessor

def is_valid_molecule(smiles: str,
                      tree_processor,
                      require_graph: bool = True) -> Tuple[bool, Optional[str]]:
    """
    与 Dataset.__getitem__ 逻辑一致的有效性检查：
    1) RDKit 能解析 SMILES
    2) 能生成 InChI
    3) tree_processor.process_mol(inchi) 返回 dgl.DGLGraph（如果 require_graph=True）
    返回: (是否有效, 错误信息或 None)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "MolFromSmiles returned None"

        inchi = Chem.MolToInchi(mol)  # 这里会触发价态类错误
        if inchi is None or len(inchi) == 0:
            return False, "Empty InChI"

        if require_graph:
            root_repr = tree_processor.process_mol(inchi)
            if not isinstance(root_repr, dgl.DGLGraph):
                return False, "process_mol did not return DGLGraph"
        return True, None
    except Exception as e:
        return False, str(e)

def process_main_row(row, tree_processor, ion_map, smiles_col, spec_col, ion_col):
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    import dgl
    # 如果 tree_processor 需要重新创建，请在此处添加代码重新初始化 tree_processor 和 ion_map
    # 例如：
    # from your_module import TreeProcessor, common
    # tree_processor = TreeProcessor(...)
    # ion_map = common.ion2onehot_pos

    RDLogger.DisableLog('rdApp.*')

    smiles = row.get(smiles_col, None)
    spec = row.get(spec_col, None)
    ion = row.get(ion_col, None)

    reject_reason = None

    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        reject_reason = "Empty SMILES"
    elif ion not in ion_map:
        reject_reason = f"Unknown ionization: {ion}"
    else:
        ok, reason = is_valid_molecule(smiles, tree_processor, require_graph=True)
        if not ok:
            reject_reason = reason

    if reject_reason is None:
        return ('valid', row)
    else:
        r = row.copy()
        r["reject_reason"] = reject_reason
        return ('rejected', r)

def filter_main_df(df: pd.DataFrame,
                   tree_processor,
                   ion_map: Dict[str, Any],
                   smiles_col: str = "smiles",
                   spec_col: str = "spec",
                   ion_col: str = "ionization",
                   keep_reason: bool = False,
                   n_processes: Optional[int] = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    过滤主数据 df，确保后续 __getitem__ 不报错。
    额外检查 ionization 能映射到 one-hot 向量。
    支持使用 joblib 进行并行处理。
    n_processes: -1 表示使用所有可用 CPU 核心。
    返回: (df_filtered, df_rejected)
    """
    valid_rows = []
    rejected_rows = []

    rows = df.to_dict(orient='records')  # 转换为字典列表，便于并行

    if n_processes != 1:
        results = Parallel(n_jobs=n_processes, backend='loky')(
            delayed(process_main_row)(row, tree_processor, ion_map, smiles_col, spec_col, ion_col) for row in tqdm(rows, desc="Filtering main df (parallel)")
        )
    else:
        results = []
        for row in tqdm(rows, desc="Filtering main df"):
            results.append(process_main_row(row, tree_processor, ion_map, smiles_col, spec_col, ion_col))

    for res_type, res_row in results:
        if res_type == 'valid':
            valid_rows.append(res_row)
        elif res_type == 'rejected':
            rejected_rows.append(res_row)

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_rej = pd.DataFrame(rejected_rows).reset_index(drop=True)
    if not keep_reason and "reject_reason" in df_rej.columns:
        pass  # 保留原因便于排错；如果不需要可 drop

    return df_valid, df_rej

def process_decoys_row(row, tree_processor, ion_map, smiles_col, spec_col, ion_col):
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    import dgl
    # 如果 tree_processor 需要重新创建，请在此处添加代码重新初始化 tree_processor 和 ion_map
    # 例如：
    # from your_module import TreeProcessor, common
    # tree_processor = TreeProcessor(...)
    # ion_map = common.ion2onehot_pos

    RDLogger.DisableLog('rdApp.*')

    smiles = row.get(smiles_col, None)
    ion = row.get(ion_col, None)

    reject_reason = None

    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        reject_reason = "Empty SMILES"
    elif ion not in ion_map:
        reject_reason = f"Unknown ionization: {ion}"
    else:
        ok, reason = is_valid_molecule(smiles, tree_processor, require_graph=True)
        if not ok:
            reject_reason = reason

    if reject_reason is None:
        return ('valid', row)
    else:
        r = row.copy()
        r["reject_reason"] = reject_reason
        return ('rejected', r)

def filter_decoys_df(decoys_df: pd.DataFrame,
                     tree_processor,
                     ion_map: Dict[str, Any],
                     smiles_col: str = "smiles",
                     spec_col: str = "spec",
                     ion_col: str = "ionization",
                     n_processes: Optional[int] = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    过滤 decoys_df。与 Dataset 中 decoy 处理逻辑一致。
    额外检查 ionization 可映射到 one-hot。
    支持使用 joblib 进行并行处理。
    n_processes: -1 表示使用所有可用 CPU 核心。
    返回: (decoys_df_filtered, decoys_df_rejected)
    """
    valid_rows = []
    rejected_rows = []

    rows = decoys_df.to_dict(orient='records')  # 转换为字典列表，便于并行

    if n_processes != 1:
        results = Parallel(n_jobs=n_processes, backend='loky')(
            delayed(process_decoys_row)(row, tree_processor, ion_map, smiles_col, spec_col, ion_col) for row in tqdm(rows, desc="Filtering decoys df (parallel)")
        )
    else:
        results = []
        for row in tqdm(rows, desc="Filtering decoys df"):
            results.append(process_decoys_row(row, tree_processor, ion_map, smiles_col, spec_col, ion_col))

    for res_type, res_row in results:
        if res_type == 'valid':
            valid_rows.append(res_row)
        elif res_type == 'rejected':
            rejected_rows.append(res_row)

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_rej = pd.DataFrame(rejected_rows).reset_index(drop=True)
    return df_valid, df_rej

def preprocess_and_save(df: pd.DataFrame,
                        decoys_df: Optional[pd.DataFrame],
                        tree_processor,
                        ion_map: Dict[str, Any],
                        output_dir: Optional[str] = None,
                        main_out_name: str = "df_filtered",
                        decoys_out_name: str = "decoys_filtered",
                        save_format: str = "tsv",
                        n_processes: Optional[int] = -1) -> Dict[str, pd.DataFrame]:
    """
    一站式预处理：
      - 过滤主 df 与 decoys_df
      - 可选保存为 CSV/Parquet
    返回字典包含过滤后的与被拒绝的 DataFrame。
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    df_filtered, df_rejected = filter_main_df(df, tree_processor, ion_map, n_processes=n_processes)
    results = {
        "df_filtered": df_filtered,
        "df_rejected": df_rejected
    }

    if decoys_df is not None:
        decoys_filtered, decoys_rejected = filter_decoys_df(decoys_df, tree_processor, ion_map, n_processes=n_processes)
        results["decoys_filtered"] = decoys_filtered
        results["decoys_rejected"] = decoys_rejected
    else:
        decoys_filtered = None
        decoys_rejected = None

    # 保存
    if output_dir:
        def _save(df_obj: Optional[pd.DataFrame], name: str):
            if df_obj is None:
                return
            path = os.path.join(output_dir, f"{name}.{save_format}")
            if save_format.lower() == "tsv":
                df_obj.to_csv(path, sep='\t', index=False)
            elif save_format.lower() in ("parquet", "pq"):
                df_obj.to_parquet(path, index=False)
            else:
                raise ValueError("save_format must be 'csv' or 'parquet'")

        _save(df_filtered, main_out_name)
        _save(df_rejected, f"{main_out_name}_rejected")
        if decoys_df is not None:
            _save(decoys_filtered, decoys_out_name)
            _save(decoys_rejected, f"{decoys_out_name}_rejected")

    # 简要统计
    print(f"[Main] kept: {len(df_filtered)}/{len(df)}; rejected: {len(df_rejected)}")
    if decoys_df is not None:
        print(f"[Decoys] kept: {len(decoys_filtered)}/{len(decoys_df)}; rejected: {len(decoys_rejected)}")

    return results