import subprocess
import os
import sys


def run_mlm_wwm():
    # 设置基础路径（根据你的项目结构调整）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "../tmp/log")
    output_dir = os.path.join(base_dir, "../tmp/tldbert")

    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令参数列表
    command = [
        "python", "mlm_wwm.py",
        "--model_name_or_path", "bert-base-uncased",
        "--train_file", os.path.join(base_dir, "../data/processed/Jira_issues.csv"),
        "--do_train",
        "--do_eval",
        "--per_device_train_batch_size", "16",
        "--gradient_accumulation_steps", "2",
        "--per_device_eval_batch_size", "16",
        "--output_dir", output_dir,
        "--evaluation_strategy", "steps",
        "--logging_steps", "2000",
        "--save_steps", "2000",
        "--fp16",
        "--warmup_steps", "100",
        "--num_train_epochs", "1",
        "--learning_rate", "3e-5",
        "--weight_decay", "1e-3",
        "--adam_epsilon", "1e-6",
        "--load_best_model_at_end"
    ]

    # 设置日志文件路径
    log_file = os.path.join(log_dir, "tldbert_mlm_wwm_run.log")

    print("Starting training process...")
    print(f"Command: {' '.join(command)}")
    print(f"Logging to: {log_file}")

    try:
        # 执行命令并重定向输出
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                command,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # 合并标准错误到标准输出
                text=True
            )

        print(f"Process started with PID: {process.pid}")
        print("Training is running in the background.")
        print(f"You can monitor progress with: tail -f {log_file}")

        return process.pid

    except Exception as e:
        print(f"Error starting training process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    pid = run_mlm_wwm()
    print(f"Training process started successfully. PID: {pid}")