import os
import time
import numpy as np
import pandas as pd
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, accuracy_score, recall_score, 
    precision_score, f1_score, precision_recall_curve
)
from tqdm import tqdm

# ===========================
# 1. 全局配置区域
# ===========================
# 项目列表：请在此处添加所有需要跑的项目名称
PROJECT_LIST = [
    # "hbase",
    # "hive",
    # "ignite",
    "cassandra",
    # "hadoop",
    # "solr",
    # "hdfs",
    # "kafka",
    # "impala",
    # "oak",
    # "yarn",
    # "mesos",
    # "derby",
    # "drill",
    # "pdfbox",
    # "mapreduce",
    # "mng",
    # "hdds",
    # "nifi",
    # "sling",
    # "harmony",
    # "spark",
    # "flink"
    ] 

DATA_DIR = "code/AIL/data"     
OUTPUT_DIR = "code/AIL/DCCNN/results" 
REPEAT_TIMES = 3  # 每个项目重复跑的次数

# 超参数配置
EMBEDDING_DIM = 300     
MAX_SEQ_LENGTH = 500    
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.005

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ===========================
# 2. 文本预处理工具 (只需加载一次)
# ===========================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.update(['com', 'org', 'java', 'http', 'https', 'html', 'img', 'src'])
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isna(text) or text == "":
        return []
    tokens = word_tokenize(str(text).lower())
    cleaned_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned_tokens.append(stemmer.stem(word))
    return cleaned_tokens

# ===========================
# 3. 核心功能函数
# ===========================

def load_data_map(project_name):
    """加载特定项目的数据"""
    csv_path = os.path.join(DATA_DIR, f"issues/{project_name}-jira-issue-dataset.csv")
    index_map_path = os.path.join(DATA_DIR, f"{project_name}/issue_index.txt")
    
    print(f"[{project_name}] 正在加载 CSV 数据...")
    try:
        df = pd.read_csv(csv_path, sep=';')
    except:
        df = pd.read_csv(csv_path, sep=',')
        
    df['full_text'] = df['Summary'].fillna('') + " " + df['DescriptionFull'].fillna('')
    key_to_text = pd.Series(df.full_text.values, index=df.Key).to_dict()
    
    print(f"[{project_name}] 正在加载索引映射...")
    index_to_tokens = {}
    with open(index_map_path, 'r', encoding='utf-8') as f:
        # 这里不显示 tqdm 以避免多重进度条混乱
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                key = parts[1]
                raw_text = key_to_text.get(key, "")
                index_to_tokens[idx] = clean_text(raw_text)
    return index_to_tokens

def train_word2vec(tokenized_corpus):
    print(f"正在训练 Word2Vec (Dim={EMBEDDING_DIM})...")
    model = gensim.models.Word2Vec(
        sentences=tokenized_corpus, 
        vector_size=EMBEDDING_DIM, 
        window=5, 
        min_count=1, 
        workers=4,
        sg=1 
    )
    return model

def build_input_matrices(pairs, labels, index_to_tokens, w2v_model):
    num_samples = len(pairs)
    X = np.zeros((num_samples, MAX_SEQ_LENGTH, EMBEDDING_DIM, 2), dtype='float32')
    
    # 同样去掉 tqdm 以保持日志整洁
    for i, (idx1, idx2) in enumerate(pairs):
        tokens_1 = index_to_tokens.get(idx1, [])
        tokens_2 = index_to_tokens.get(idx2, [])
        
        for j, word in enumerate(tokens_1):
            if j >= MAX_SEQ_LENGTH: break
            if word in w2v_model.wv:
                X[i, j, :, 0] = w2v_model.wv[word]
        
        for j, word in enumerate(tokens_2):
            if j >= MAX_SEQ_LENGTH: break
            if word in w2v_model.wv:
                X[i, j, :, 1] = w2v_model.wv[word]
    return X, np.array(labels)

def get_dataset_split(mode, pos_data, neg_data, index_to_tokens, w2v_model):
    pos_pairs = pos_data[mode]
    neg_pairs = neg_data[mode]
    
    pos_labels = [1] * len(pos_pairs)
    neg_labels = [0] * len(neg_pairs)
    
    all_pairs = np.concatenate([pos_pairs, neg_pairs])
    all_labels = np.concatenate([pos_labels, neg_labels])
    
    indices = np.arange(len(all_pairs))
    np.random.shuffle(indices)
    all_pairs = all_pairs[indices]
    all_labels = all_labels[indices]
    
    return build_input_matrices(all_pairs, all_labels, index_to_tokens, w2v_model)

def build_dccnn_model():
    input_layer = layers.Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM, 2))
    
    x1 = layers.Conv2D(100, kernel_size=(1, EMBEDDING_DIM), strides=(1,1), activation='relu')(input_layer)
    x1 = layers.BatchNormalization(axis=-1)(x1)
    x1 = layers.Reshape((MAX_SEQ_LENGTH, 100, 1))(x1)
    
    pool_outputs = []
    for kernel_size in [2, 3, 4]:
        conv = layers.Conv2D(200, kernel_size=(kernel_size, 100), activation='relu')(x1)
        conv = layers.BatchNormalization(axis=-1)(conv)
        pool = layers.MaxPooling2D(pool_size=(MAX_SEQ_LENGTH - kernel_size + 1, 1))(conv)
        flat = layers.Flatten()(pool)
        pool_outputs.append(flat)
    
    merged = layers.Concatenate()(pool_outputs)
    
    dense = layers.Dropout(0.5)(merged)
    dense = layers.Dense(300, activation='relu')(dense)
    dense = layers.BatchNormalization()(dense)
    
    dense = layers.Dropout(0.5)(dense)
    dense = layers.Dense(100, activation='relu')(dense)
    dense = layers.BatchNormalization()(dense)
    
    output = layers.Dense(1, activation='sigmoid')(dense)
    
    model = models.Model(inputs=input_layer, outputs=output, name='DCCNN')
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ===========================
# 4. 单次实验运行逻辑
# ===========================
def run_single_experiment(project_name, run_index):
    """运行指定项目的一次实验"""
    print(f"\n{'='*50}")
    print(f"开始任务: 项目={project_name}, 轮次={run_index+1}/{REPEAT_TIMES}")
    print(f"{'='*50}")
    
    start_time = time.time()
    run_id = f"{project_name}_run{run_index+1}_{int(start_time)}"
    
    # 路径准备
    pos_npz_path = os.path.join(DATA_DIR, f"{project_name}/train_val_test_pos_issue.npz")
    neg_npz_path = os.path.join(DATA_DIR, f"{project_name}/train_val_test_neg_issue.npz")
    
    # 1. 加载数据与 Embedding
    # 注意：为了效率，如果在同一项目循环内，理想情况下应该只加载一次文本和训练一次W2V
    # 但为了保持每次实验的独立性（如随机种子影响），这里每次都重新构建
    index_map = load_data_map(project_name)
    w2v_model = train_word2vec(list(index_map.values()))
    
    # 2. 加载数据集
    pos_data = np.load(pos_npz_path)
    neg_data = np.load(neg_npz_path)
    
    X_train, y_train = get_dataset_split('train', pos_data, neg_data, index_map, w2v_model)
    X_val, y_val = get_dataset_split('val', pos_data, neg_data, index_map, w2v_model)
    X_test, y_test = get_dataset_split('test', pos_data, neg_data, index_map, w2v_model)
    
    # 3. 训练
    model = build_dccnn_model()
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=0)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=2 # 简化日志输出
    )
    
    # 4. 评估 (动态阈值)
    y_pred_prob = model.predict(X_test).flatten()
    
    precision_points, recall_points, thresholds = precision_recall_curve(y_test, y_pred_prob)
    numerator = 2 * precision_points * recall_points
    denominator = precision_points + recall_points
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred_dynamic = (y_pred_prob >= best_threshold).astype(int)
    
    # 指标计算
    metrics = {
        'Project': project_name,
        'Run_Index': run_index + 1,
        'Run_ID': run_id,
        'AUC': roc_auc_score(y_test, y_pred_prob),
        'AP': average_precision_score(y_test, y_pred_prob),
        'Accuracy': accuracy_score(y_test, y_pred_dynamic),
        'Recall': recall_score(y_test, y_pred_dynamic),
        'Precision': precision_score(y_test, y_pred_dynamic),
        'AF1': f1_score(y_test, y_pred_dynamic),
        'Best_Threshold': best_threshold,
        'Time_Taken': time.time() - start_time
    }
    
    print(f"完成: AUC={metrics['AUC']:.4f}, F1={metrics['AF1']:.4f}")
    
    # 5. 清理资源
    keras.backend.clear_session()
    del model, X_train, X_val, X_test, w2v_model
    import gc
    gc.collect()
    
    return metrics

# ===========================
# 5. 批量执行入口
# ===========================
if __name__ == "__main__":
    all_results = []
    
    # 全局结果文件路径
    global_csv_path = os.path.join(OUTPUT_DIR, "all_projects_experiment_results.csv")
    
    for project in PROJECT_LIST:
        print(f"\n>>> 进入项目: {project}")
        
        # 检查文件是否存在，避免报错
        if not os.path.exists(os.path.join(DATA_DIR, f"{project}/issue_index.txt")):
            print(f"跳过项目 {project}: 找不到 issue_index.txt")
            continue
            
        for i in range(REPEAT_TIMES):
            try:
                # 运行单次实验
                result = run_single_experiment(project, i)
                all_results.append(result)
                
                # 实时保存（防止程序中途崩溃数据丢失）
                df_res = pd.DataFrame([result])
                if not os.path.exists(global_csv_path):
                    df_res.to_csv(global_csv_path, mode='w', header=True, index=False)
                else:
                    df_res.to_csv(global_csv_path, mode='a', header=False, index=False)
                    
            except Exception as e:
                print(f"项目 {project} 第 {i+1} 次运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\n所有实验结束！结果已汇总至: {global_csv_path}")