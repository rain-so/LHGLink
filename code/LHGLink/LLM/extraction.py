import os
import time
import psutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

TARGET_LAYER = 18
COMPRESS_DIM = 512
USE_ATTENTION_POOLING = True

PROJECTS = [ ]

BASE_DIR = "code/LHGLink/data"
MODEL_PATH = "/home/stu/Documents/LLM_CS/Llama/llama3/Meta-Llama-3-8B-Instruct-hf"
MAX_TOKENS = 300
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 系统资源监控函数
def print_system_info():
    """打印当前系统资源使用情况"""
    mem = psutil.virtual_memory()
    print(f"内存使用: {mem.used/1024**3:.1f}G/{mem.total/1024**3:.1f}G ({mem.percent}%)")
    
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: 已分配={alloc:.2f}G, 已保留={reserved:.2f}G, 总量={total:.2f}G")
    else:
        print("没有可用的GPU设备")
    
    load = os.getloadavg()
    print(f"系统负载: {load[0]:.2f} (1min), {load[1]:.2f} (5min), {load[2]:.2f} (15min)")

class FeatureCompressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        return self.projection(x)

def extract_features(batch, model, tokenizer, compressor, attn_layer, device):
    inputs = tokenizer(
        batch,
        max_length=MAX_TOKENS,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type=="cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[TARGET_LAYER]  # [batch, seq_len, hidden_size]

    attention_mask = inputs.attention_mask

    if USE_ATTENTION_POOLING:
        attn_weights = attn_layer(hidden_states).squeeze(-1)
        attn_weights = torch.softmax(attn_weights.masked_fill(attention_mask == 0, -1e9), dim=-1)
        pool_features = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
    else:
        eos_positions = attention_mask.sum(dim=1) - 1
        pool_features = hidden_states[torch.arange(hidden_states.size(0)), eos_positions]

    compressed_features = compressor(pool_features)

    del inputs, outputs, hidden_states, pool_features
    if USE_ATTENTION_POOLING:
        del attn_weights

    compressed_features = compressed_features.detach().cpu()
    if compressed_features.dtype == torch.bfloat16:
        compressed_features = compressed_features.float()
    
    return compressed_features.numpy().astype(np.float32)

def process_project(project_name, model, tokenizer, compressor, attn_layer, device):
    print("\n" + "="*50)
    print(f"开始处理项目: {project_name}")
    print("="*50)

    CSV_FILE = os.path.join(BASE_DIR, project_name, f"{project_name}-enriched-only.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, project_name, f"{project_name}-features-Gptllama.npy")

    print(f"\n读取CSV文件: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE, sep=',')
        print(f"CSV文件的列名: {df.columns.tolist()}")
        texts = df['Augmented_Text'].fillna('').tolist()
        
        print(f"成功处理 {len(texts)} 条记录")
        
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        return False

    total_records = len(texts)
    total_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    features_list = []
    
    print("\n开始特征提取...")
    print(f"总记录数: {total_records}, 总批次数: {total_batches}")
    print_system_info()
    
    try:
        progress_bar = tqdm(total=total_records, desc=f"处理[{project_name}]")
        
        for i in range(0, total_records, BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            features = extract_features(batch, model, tokenizer, compressor, attn_layer, device)
            features_list.append(features)
            processed_count = i + len(batch)
            progress_bar.update(len(batch))
        
        progress_bar.close()

        print("\n处理完成! 保存特征文件...")
        features_array = np.vstack(features_list)
        print(f"特征数组形状: {features_array.shape} (记录数×维度)")
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        np.save(OUTPUT_FILE, features_array)
        print(f"特征已保存至: {os.path.abspath(OUTPUT_FILE)}")
        print_system_info()
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ 处理过程中出错: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("批量项目特征提取工具")
    print(f"开始时间: {time.ctime()}")
    print(f"处理项目数量: {len(PROJECTS)}")
    print(f"使用设备: {DEVICE.upper()}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"目标层: {TARGET_LAYER}")
    print(f"压缩维度: {COMPRESS_DIM}")
    print(f"最大token长度: {MAX_TOKENS}")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"显存碎片整理: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '未设置')}")
    print("="*80)

    print("\n加载本地Llama-3-8B模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            print("已设置pad_token为eos_token")

        if DEVICE == "cuda":
            device = torch.device("cuda")
            dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        print(f" (layer {TARGET_LAYER})...")
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto" if DEVICE == "cuda" else None,
            output_hidden_states=True
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if DEVICE == "cpu":
            model = model.to(device)
        
        model.eval()

        total_layers = model.config.num_hidden_layers
        print(f"模型总层数: {total_layers}")

        if TARGET_LAYER < 0 or TARGET_LAYER >= total_layers:
            print(f"警告: 目标层{TARGET_LAYER}超出范围(0-{total_layers-1})，使用中间层{total_layers//2}")
            target_layer = total_layers // 2
        else:
            target_layer = TARGET_LAYER

        compressor = FeatureCompressor(model.config.hidden_size, COMPRESS_DIM).to(device).to(dtype)
        compressor.eval()

        if USE_ATTENTION_POOLING:
            attn_layer = nn.Linear(model.config.hidden_size, 1).to(device).to(dtype)
            attn_layer.eval()
        else:
            attn_layer = None
        
        print(f"模型加载成功! 目标层: {target_layer}/{total_layers}")
        print(f"模型位置: {next(model.parameters()).device}, 数据类型: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    start_time = time.time()
    success_count = 0
    
    for project in PROJECTS:
        success = process_project(project, model, tokenizer, compressor, attn_layer, device)
        if success:
            success_count += 1
            print(f"\n✅ 项目 {project} 处理完成!")
        else:
            print(f"\n❌ 项目 {project} 处理失败!")
        print("-" * 80)

        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"批量处理完成!")
    print(f"总项目数: {len(PROJECTS)}")
    print(f"成功项目数: {success_count}")
    print(f"失败项目数: {len(PROJECTS) - success_count}")
    print(f"总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"平均每个项目耗时: {elapsed/len(PROJECTS):.2f}秒")
    print("="*80)

if __name__ == "__main__":
    main()
    print("\n 批量特征提取完成!")