import numpy as np
import pandas as pd
import pickle,pathlib,scipy.sparse

save_path = 'data/'

issue_issue = pd.read_csv('data/index/issue_issue_index.txt',
                          sep=' ',
                          header=None,
                          names=['issue_id', 'issue_id_1', '_'],
                          usecols=[0, 1],
                          dtype={'raw_id1': int, 'raw_id2': int},
                          encoding='utf-8')
issue_assignee = pd.read_csv('data/index/issue_assignee_index.txt', sep=' ', header=None, names=['issue_id', 'assignee_id'], keep_default_na=False, encoding='utf-8')
issue_component = pd.read_csv('data/index/issue_component_index.txt', sep=' ', header=None, names=['issue_id', 'component_id'], keep_default_na=False, encoding='utf-8')


issues = pd.read_csv('data/index/issue_'
                     'index.txt', sep=' ', header=None, names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
assignees = pd.read_csv('data/index/assignee_index.txt', sep=' ', header=None, names=['assignee_id', 'assignee'], keep_default_na=False, encoding='utf-8')
components = pd.read_csv('data/index/component_index.txt', sep=' ', header=None, names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')
num_issues = len(issues)
num_assignees = len(assignees)
num_components = len(components)

adjM = scipy.sparse.load_npz('data/adjM.npz')### 读取
# 检查邻接矩阵的非零元素分布
print("===== adjM 非零元素分布 =====")
print("issues-issues 边数量:", adjM[:num_issues, :num_issues].nnz)
print("issues-assignees 边数量:", adjM[:num_issues, num_issues:num_issues+num_assignees].nnz)
print("issues-components 边数量:", adjM[:num_issues, num_issues+num_assignees:].nnz)


###每个节点的type all nodes (issues, assignees, components)
issue_assignee_list = {
    i: adjM[i, len(issues):len(issues)+len(assignees)].nonzero()[0]
    for i in range(len(issues))
}

issue_component_list = {
    i: adjM[i, len(issues)+len(assignees):].nonzero()[0]  # issues在最后
    for i in range(len(issues))
}

assignee_issue_list = {
    i: adjM[len(issues)+i, :len(issues)].nonzero()[1]
    for i in range(len(assignees))
}

assignee_component_list = {
    i: adjM[len(issues)+i, len(issues)+len(assignees):].nonzero()[0]
    for i in range(len(assignees))
}

component_issue_list = {
    i: adjM[len(issues) + len(assignees) + i, :len(issues)].nonzero()[1]  # 使用列索引（issue的全局索引）
    for i in range(len(components))
}

component_assignee_list = {
    i: adjM[len(issues)+len(assignees)+i, len(issues):len(issues)+len(assignees)].nonzero()[0]
    for i in range(len(components))
}

component_component_list = {
    i: adjM[len(issues)+len(assignees)+i, len(issues)+len(assignees):].nonzero()[0]
    for i in range(len(components))
}

'''
开始创建
'''


## 0-0 issue-issue
i_i = issue_issue.to_numpy(dtype=np.int32)
i_i=np.array(i_i)
sorted_index = sorted(list(range(len(i_i))), key=lambda i : i_i[i].tolist())
i_i = i_i[sorted_index]

# 0-1-0 i_a_i
i_a_i = []
for a, i_list in assignee_issue_list.items():
    if len(i_list) > 0:
        i_a_i.extend([(i1, a, i2) for i1 in i_list for i2 in i_list])
i_a_i = np.array(i_a_i, dtype=np.int32) if i_a_i else np.empty((0, 3), dtype=np.int32)
i_a_i[:, 1] += num_issues
if len(i_a_i) > 0:
    sorted_index = sorted(range(len(i_a_i)), key=lambda idx: i_a_i[idx, [0, 2, 1]].tolist())
    i_a_i = i_a_i[sorted_index]

# 0-2-0 i_c_i
i_c_i = []
for c, i_list in component_issue_list.items():
    if len(i_list) > 0:
        i_c_i.extend([(i1, c, i2) for i1 in i_list for i2 in i_list])
i_c_i = np.array(i_c_i, dtype=np.int32) if i_c_i else np.empty((0, 3), dtype=np.int32)
i_c_i[:, 1] += (num_issues + num_assignees)
if len(i_c_i) > 0:
    sorted_index = sorted(range(len(i_c_i)), key=lambda idx: i_c_i[idx, [0, 2, 1]].tolist())
    i_c_i = i_c_i[sorted_index]

# 0-2-2-0 i_c_c_i
# i_c_c_i = []
# for i, c_list in issue_component_list.items():
#     for c1 in c_list:
#         for c2 in component_component_list[c1]:
#             for i2 in component_issue_list[c2]:
#                 i_c_c_i.extend([(i, c1, c2, i2)])
# i_c_c_i = np.array(i_c_c_i)
# i_c_c_i[:, [0, 3]]+= len(component)+len(component)
# i_c_c_i[:,[1, 2]]+= len(assignee)
# sorted_index = sorted(list(range(len(i_c_c_i))), key=lambda i :i_r_r_i[i, [0, 2, 1, 3]].tolist())
# i_c_c_i = i_c_c_i[sorted_index]


print("\n===== 边列表统计 =====")
print("i_i (issue-issue) 边数量:", len(i_i))
print("i_a_i (issue-assignee-issue) 边数量:", len(i_a_i))
print("i_c_i (issue-component-issue) 边数量:", len(i_c_i))
# print("i_c_c_i (issue-component-component-issue) 边数量:", len(i_c_c_i))
expected_metapaths = [[(0, 0), (0, 1, 0), (0, 2, 0)]]
metapath_indices_mapping = {(0, 0): i_i, (0, 1, 0): i_a_i, (0, 2, 0): i_c_i}

target_idx_lists = [np.arange(num_issues),np.arange(num_assignees),np.arange(num_components)]
offset_list = [0, num_issues , num_assignees+num_issues]
print("\n===== 偏移量 =====")
print("offset_list:", offset_list)
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_path + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

for i, metapaths in enumerate(expected_metapaths):

    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file: ###  每一个路径存一个pickle 文件
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx+ offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        # np.save(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        with open(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:

                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]

                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right


