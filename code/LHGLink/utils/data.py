import random

import networkx as nx
import numpy as np
import scipy
import pickle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_LastFM_data(prefix='./data'):
    issues = pd.read_csv('data/index/issue_index.txt', sep=' ', header=None, names=['issue_id', 'issue'],
                         keep_default_na=False, encoding='utf-8')
    assignees = pd.read_csv('data/index/assignee_index.txt', sep=' ', header=None, names=['assignee_id', 'assignee'],
                            keep_default_na=False, encoding='utf-8')
    components = pd.read_csv('data/index/component_index.txt', sep=' ', header=None,
                             names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')

    num_issues = len(issues)
    num_assignees = len(assignees)
    num_components = len(components)

    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    in_file.close()
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    in_file.close()
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    in_file.close()

    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

        ##### 传入节点特征
    # 使用PCA进行降维
    pca = PCA(n_components=128)

    features = np.load(prefix + '/features-gptllama/solr-features-Gptllama.npy')
    # features = np.load(prefix + '/spark/spark-features-Gptllama.npy')
    features_reduced = pca.fit_transform(features)
    features_0 = torch.tensor(features_reduced, dtype=torch.float32)

    # features_0 = np.random.rand(num_issues, 50)
    features_1 = np.random.rand(num_assignees, 10)
    features_2 = np.random.rand(num_components, 10)
    # features_1 = np.random.normal(0, 1 / np.sqrt(128), (num_assignees, 128))
    # features_2 = np.random.normal(0, 1 / np.sqrt(128), (num_components, 128))

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')

    train_val_test_pos_issue = np.load(prefix + '/train_val_test_pos_issue.npz')
    train_val_test_neg_issue = np.load(prefix + '/train_val_test_neg_issue.npz')

    return [[adjlist00, adjlist01, adjlist02]],\
           [[idx00, idx01, idx02]], \
        [features_0, features_1, features_2],\
           adjM, type_mask, train_val_test_pos_issue, train_val_test_neg_issue




# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs
