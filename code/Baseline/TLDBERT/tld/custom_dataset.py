import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from sklearn.model_selection import train_test_split
import re
import random
# 导入我们在 linktypes 中定义的配置
from linktypes import LABEL_NAMES, linktype_map


def extract_links_with_types(link_str):
    """
    从链接字符串中提取 (类型, Key) 对
    格式示例: "dependent:CAMEL-1539,Reference:CAMEL-1527"
    """
    if not isinstance(link_str, str) or pd.isna(link_str):
        return []

    # 正则解释:
    # ([^:,]+)  -> 捕获冒号前的非逗号字符作为类型
    # :         -> 分隔符
    # ([A-Za-z]+-\d+) -> 捕获Key (如 PROJ-123)
    raw_links = re.findall(r'([^:,]+):([A-Za-z]+-\d+)', link_str)

    # 清洗数据：去除空白字符
    return [(t.strip(), k.strip()) for t, k in raw_links]


def create_dataset(
        issue_csv: str,
        target: str = "linktype",  # 这里的target参数其实主要用于区分，逻辑里我们直接用多分类
        include_non_links: bool = True,
        random_seed: int = 42,
        non_link_ratio: float = 1.0
):
    # 加载问题数据
    issues_df = pd.read_csv(issue_csv, sep=';', encoding='utf-8-sig')
    if 'ï»¿Project' in issues_df.columns:
        issues_df = issues_df.rename(columns={'ï»¿Project': 'Project'})

    # 预处理：建立 Key 到 Issue 详情的快速查找字典，提高速度
    issues_map = issues_df.set_index('Key').to_dict('index')

    # 定义我们关心的正样本类型 (排除 NoLink，因为它是通过采样生成的)
    target_positive_types = set(LABEL_NAMES) - {'NoLink'}

    # 提取所有存在的链接关系
    # 结构: {(key1, key2): 'type'}
    # 注意：这里我们假设如果有多个链接类型，取其中一个，或者数据中一行只描述一种关系
    positive_samples_dict = {}

    for _, row in issues_df.iterrows():
        source_key = row['Key']

        # 提取 inward 和 outward
        # 注意：这里需要根据实际业务逻辑决定是否区分方向。
        # 你提到"不考虑方向"，但在Jira中 "Duplicate" 和 "Duplicated by" 是不同的字符串。
        # 你需要确保 CSV 中的字符串能精确匹配 linktypes.py 中的 LABEL_NAMES，
        # 或者在这里做一个映射转换 (normalize)。

        raw_links = []
        raw_links.extend(extract_links_with_types(row.get('InwardIssueLinks', '')))
        raw_links.extend(extract_links_with_types(row.get('OutwardIssueLinks', '')))

        for link_type, target_key in raw_links:
            if target_key == source_key: continue  # 跳过自引用

            # 过滤：只保留我们在 LABEL_NAMES 中定义的类型
            if link_type not in target_positive_types:
                continue

            # 为了去重且不考虑方向 (A,B) 和 (B,A) 视为同一个对
            # 我们强制 key1 < key2
            if source_key < target_key:
                k1, k2 = source_key, target_key
            else:
                k1, k2 = target_key, source_key

            # 存储：如果同一对有多个关系，这里简单的逻辑是覆盖（或者你可以选择跳过）
            positive_samples_dict[(k1, k2)] = link_type

    # 构建正样本列表
    positive_samples = []
    positive_pairs_set = set()  # 用于负采样去重

    for (key1, key2), link_type in positive_samples_dict.items():
        if key1 not in issues_map or key2 not in issues_map:
            continue

        issue1 = issues_map[key1]
        issue2 = issues_map[key2]

        positive_samples.append({
            'project': issue1['Project'],
            'key1': key1,
            'key2': key2,
            'title_1': issue1['Summary'],
            'description_1': issue1['DescriptionFull'],
            'title_2': issue2['Summary'],
            'description_2': issue2['DescriptionFull'],
            'link_type_str': link_type,
            'label': linktype_map[link_type]  # 映射到 1, 2, 3...
        })
        positive_pairs_set.add((key1, key2))

    # 构建负样本 (NoLink)
    negative_samples = []
    if include_non_links:
        valid_keys = list(issues_map.keys())
        num_negative = int(len(positive_samples) * non_link_ratio)

        count = 0
        while count < num_negative:
            k1, k2 = random.sample(valid_keys, 2)
            # 排序 key 以匹配 positive_pairs_set 的格式
            if k1 > k2: k1, k2 = k2, k1

            if k1 != k2 and (k1, k2) not in positive_pairs_set:
                issue1 = issues_map[k1]
                issue2 = issues_map[k2]

                negative_samples.append({
                    'project': issue1['Project'],
                    'key1': k1,
                    'key2': k2,
                    'title_1': issue1['Summary'],
                    'description_1': issue1['DescriptionFull'],
                    'title_2': issue2['Summary'],
                    'description_2': issue2['DescriptionFull'],
                    'link_type_str': 'NoLink',
                    'label': linktype_map['NoLink']  # 映射到 0
                })
                # 将生成的负样本加入集合，防止重复生成完全相同的负样本
                positive_pairs_set.add((k1, k2))
                count += 1

    # 合并
    all_samples = positive_samples + negative_samples
    random.Random(random_seed).shuffle(all_samples)

    # 转换为 DataFrame
    all_df = pd.DataFrame(all_samples)

    # 打印一下分布情况，便于调试
    print("Label distribution in full dataset:")
    print(all_df['link_type_str'].value_counts())

    # 分割数据集
    train_df, temp_df = train_test_split(all_df, test_size=0.3, random_state=random_seed, stratify=all_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_seed, stratify=temp_df['label'])

    features = Features({
        'project': Value('string'),
        'key1': Value('string'),
        'key2': Value('string'),
        'title_1': Value('string'),
        'description_1': Value('string'),
        'title_2': Value('string'),
        'description_2': Value('string'),
        'link_type_str': Value('string'),
        # 使用动态定义的类别名称
        'label': ClassLabel(num_classes=len(LABEL_NAMES), names=LABEL_NAMES)
    })

    return DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True), features=features),
        'val': Dataset.from_pandas(val_df.reset_index(drop=True), features=features),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True), features=features)
    })