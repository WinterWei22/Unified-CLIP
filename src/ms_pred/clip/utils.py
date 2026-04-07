import subprocess
import shlex
import sys
import os

def run_predict(new_checkpoint_path):
    """
    运行 run_from_config.py 脚本，并传入自定义的 --checkpoint 参数来覆盖 YAML 文件中的值。
    """
    
    final_command = [
        sys.executable,
        "src/ms_pred/clip/clip_predict.py",
        "--gpu",
        "--num-workers", "16",
        "--batch-size", "256",
        "--dataset-name", "msg",
        "--split-name", "split_msg.tsv",
        "--subset-datasets", "test_only",
        "--dataset-labels", "labels.tsv",
        "--checkpoint", f'{new_checkpoint_path}/best.ckpt', 
        
        "--save-dir", new_checkpoint_path,
        "--binned-out",
        "--embed-adducts",
        "--inject-early",
        "--magma-dag-folder", "/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/",
    ]
    cuda_env = {"CUDA_VISIBLE_DEVICES": "0"}
    command_str = shlex.join(final_command)
    try:
        print(f"--- running predict ---")
        print(f"(CUDA_VISIBLE_DEVICES={cuda_env.get('CUDA_VISIBLE_DEVICES', 'None')}):\n{command_str}\n")
        result = subprocess.run(
                    final_command, 
                    env=cuda_env, 
                    check=True, 
                    text=True, 
                    # capture_output=True,
                    encoding='utf-8'
                )
        print("Successed")
        print("Stdout:\n", result.stdout)
        print("Stderr:\n", result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e.returncode}")
        print("Stdout:\n", e.stdout)
        print("Stderr:\n", e.stderr)
    except FileNotFoundError:
        print("Error")
        
def run_predict_smi(new_checkpoint_path):
    """
    根据 predict_msg_smi.yaml 的配置，构建并运行底层脚本 (clip_predict_smi.py)
    的命令，并用新的 checkpoint 路径覆盖 YAML 文件中的值。
    """

    script_to_run = "src/ms_pred/clip/clip_predict_smi.py"

    final_command = [
        sys.executable,
        script_to_run,
        "--gpu", 
        "--num-workers", "16",
        "--batch-size", "1024",
        "--dataset-name", "msg",
        "--subset-datasets", "test_only",
        "--dataset-labels", "cands_df_test_formula_256_new_filtered_filteredagain.tsv", 
        "--checkpoint", f'{new_checkpoint_path}/best.ckpt', 
        "--save-dir", new_checkpoint_path,
        "--binned-out",
        "--split-name", "split_msg.tsv",
        "--magma-dag-folder", "/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/",
    ]
    
    cuda_env = {"CUDA_VISIBLE_DEVICES": "0"}
    command_str = shlex.join(final_command)
    print(f"--- running predict smi ---")
    print(f"CUDA_VISIBLE_DEVICES={cuda_env}")
    print(f"\n{command_str}\n")
    
    try:
        result = subprocess.run(
            final_command, 
            env=cuda_env, 
            check=True, 
            text=True, 
            encoding='utf-8'
        )
        print("Successed")
        print("-" * 20)
        print("Stdout:\n", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e.returncode}")
        print("-" * 20)
        print("Stdout:\n", e.stdout)
        print("Stderr:\n", e.stderr)
    except FileNotFoundError:
        print("ERRORS")
    finally:
        del os.environ['CUDA_VISIBLE_DEVICES']