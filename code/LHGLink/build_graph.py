import pathlib,json
import numpy as np
import scipy.sparse
import scipy.io, os
import pandas as pd
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
import dgl
'''
构建邻接矩阵
'''

######### 构建矩阵
issue_issue = pd.read_csv('data/Index/issue_issue_index.txt', sep=' ', header=None, names=['issue_id', 'issue_id_1', 'issue_link'], keep_default_na=False, encoding='utf-8')
issue_assignee = pd.read_csv('data/Index/issue_assignee_index.txt', sep=' ', header=None, names=['issue_id', 'assignee_id'], keep_default_na=False, encoding='utf-8')
issue_component = pd.read_csv('data/Index/issue_component_index.txt', sep=' ', header=None, names=['issue_id', 'component_id'], keep_default_na=False, encoding='utf-8')

issues = pd.read_csv('data/index/issue_index.txt', sep=' ', header=None, names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
assignees = pd.read_csv('data/index/assignee_index.txt', sep=' ', header=None, names=['assignee_id', 'assignee'], keep_default_na=False, encoding='utf-8')
components = pd.read_csv('data/index/component_index.txt', sep=' ', header=None, names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')

num_issues = len(issues)
num_assignees = len(assignees)
num_components = len(components)


dim = len(issues) + len(assignees) + len(components)

# 保存每个节点的type all nodes (issues, assignees, components) type labels
type_mask = np.zeros((dim), dtype=int)
type_mask[len(issues):len(issues)+len(assignees)] = 1
type_mask[len(issues)+len(assignees):len(issues)+len(assignees)+len(components)] = 2

np.save('data/node_types.npy', type_mask)

issue_id_mapping = {row['issue_id']: i for i, row in issues.iterrows()}
assignee_id_mapping = {row['assignee_id']: i + len(issues) for i, row in assignees.iterrows()}
component_id_mapping = {row['component_id']: i + len(issues)+len(assignees) for i, row in components.iterrows()}


# 获取所有可能的链接类型（例如 ['Related', 'Duplicate', 'Blocked']）
link_types = issue_issue['issue_link'].unique()
link_type_counts = issue_issue['issue_link'].value_counts()
print(link_types)
print(link_type_counts)

####### 构建邻接矩阵
adjM = np.zeros((dim, dim), dtype=int)

for _, row in issue_issue.iterrows():
    idx1 = issue_id_mapping[row['issue_id']]
    idx2 = issue_id_mapping[row['issue_id_1']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

# 填充 Issue-Assignee 边
for _, row in issue_assignee.iterrows():
    idx1 = issue_id_mapping[row['issue_id']]
    idx2 = assignee_id_mapping[row['assignee_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

# 填充 Issue-Component 边
for _, row in issue_component.iterrows():
    idx1 = issue_id_mapping[row['issue_id']]
    idx2 = component_id_mapping[row['component_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

##### 保存邻接矩阵
scipy.sparse.save_npz('data/adjM.npz', scipy.sparse.csr_matrix(adjM))

## 读取邻接矩阵
adjM = scipy.sparse.load_npz('data/adjM.npz')

num_nodes = adjM.shape[0]
num_edges = adjM.nnz

print("节点数:", num_nodes)
print("边数:", num_edges)

adjM = adjM.toarray()

# 统计不同类型的边
src_issue, dst_issue = np.where(adjM[:num_issues, :num_issues] == 1)  # issue-issue
src_assignee, dst_assignee = np.where(adjM[:num_issues, num_issues:num_issues+num_assignees] == 1)  # issue-assignee
src_component, dst_component = np.where(adjM[:num_issues, num_issues+num_assignees:] == 1)  # issue-component

print("\n===== 边类型统计 =====")
print(f"Issue-Issue 边数: {len(src_issue)}")
print(f"Issue-Assignee 边数: {len(src_assignee)}")
print(f"Issue-Component 边数: {len(src_component)}")





