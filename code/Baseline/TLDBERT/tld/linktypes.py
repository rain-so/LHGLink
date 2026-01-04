from typing import Dict

# # 简化为二分类映射
# linktype_map: Dict[str, int] = {
#     "has_link": 1,      # 表示存在链接
#     "no_link": 0         # 表示不存在链接
# }
#
# # 标签名称映射
# label_names = ["no_link", "has_link"]
#
# # 目标类型保持为字符串
# Target = str

# # 定义类别列表 (顺序很重要，0号通常留给 NoLink)
# LABEL_NAMES = ['NoLink', 'Reference', 'Duplicate', 'Completes', 'Blocker']
#
# # 自动生成名称到ID的映射
# # {'NoLink': 0, 'Reference': 1, 'Duplicate': 2, 'dependent': 3}
# linktype_map: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
#
# # 目标类型定义
# Target = str

# linktypes.py

# 1. 定义核心类型
CORE_TYPES = ['Reference', 'Duplicate', 'Block', 'Dependent']
# CORE_TYPES = ['Reference', 'Duplicate', 'Dependent']
# 2. 定义标签名称列表 (用于 BERT 输出和分类报告)
# 顺序: 0:NoLink, 1:Ref, 2:Dup, 3:Comp, 4:Block, 5:Other
LABEL_NAMES = ['NoLink'] + CORE_TYPES
# LABEL_NAMES = ['NoLink'] + CORE_TYPES + ['Other']
# 3. 建立映射字典
linktype_map = {name: i for i, name in enumerate(LABEL_NAMES)}

Target = str