import time
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

from utils.pytorchtools import EarlyStopping
from utils.tools import index_generator, parse_minibatch_LastFM
from scipy import sparse
import scipy
import dgl
from GNN import GCN, GAT, GCN_dense
from base_simple_hgn import myGATConv
# Hyper Params
num_ntype = 3
dropout_rate = 0.5
lr = 0.01
weight_decay = 0.000
num_layers = 1
issues = pd.read_csv('data/index/issue_index.txt', sep=' ', header=None, names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
assignees = pd.read_csv('data/index/assignee_index.txt', sep=' ', header=None, names=['assignee_id', 'assignee'], keep_default_na=False, encoding='utf-8')
components = pd.read_csv('data/index/component_index.txt', sep=' ', header=None, names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')

num_issues = len(issues)
num_assignees = len(assignees)
num_components = len(components)

num_user = num_issues
num_artist = num_assignees
num_tag = num_components


class NewGAT(nn.Module):
    def __init__(self, g, num_layers, layer_dim, num_heads, activation,
                 feat_drop, attn_drop, negative_slope, residual,
                 in_dims, type_mask):
        super(NewGAT, self).__init__()
        self.g = g
        self.type_mask = torch.LongTensor(type_mask)
        self.num_ntype = 3
        self.num_layers = num_layers
        self.layer_dim = layer_dim

        # 投影层: 将不同节点类型特征映射到统一维度
        self.proj_list = nn.ModuleList()
        for i in range(len(in_dims)):
            self.proj_list.append(nn.Linear(in_dims[i], layer_dim))

        # 边类型嵌入 (0: 普通边, 1: 自环)
        self.num_etypes = 2

        # 创建GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            gat_layer = myGATConv(
                edge_feats=layer_dim,
                num_etypes=self.num_etypes,
                in_feats=layer_dim,
                out_feats=layer_dim,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual if i > 0 else False,
                activation=activation,
                allow_zero_in_degree=True,
                bias=True,
                alpha=0.1
            )
            self.gat_layers.append(gat_layer)

    def forward(self, features_list):
        device = features_list[0].device
        type_mask = self.type_mask.to(device)

        # 1. 特征投影
        all_feats = []
        for ntype in range(len(features_list)):
            # 只处理存在的节点类型
            if ntype < len(self.proj_list):
                proj_feats = self.proj_list[ntype](features_list[ntype])
                all_feats.append(proj_feats)
            else:
                # 对于没有定义的类型，使用零向量
                num_nodes = (type_mask == ntype).sum().item()
                all_feats.append(torch.zeros(num_nodes, self.layer_dim, device=device))

        # 2. 构建完整特征矩阵 (num_nodes x layer_dim)
        h = torch.zeros(type_mask.shape[0], self.layer_dim, device=device)
        for ntype in range(self.num_ntype):
            mask = (type_mask == ntype)
            h[mask] = all_feats[ntype]

        # 3. 创建边类型特征
        e_feat = torch.zeros(self.g.number_of_edges(), dtype=torch.long, device=device)
        # 标记自环边（假设自环边是最后添加的）
        num_nodes = self.g.number_of_nodes()
        e_feat[-num_nodes:] = 1  # 自环边的类型为1

        # 4. 通过GAT层
        attn_weights = None
        for i, layer in enumerate(self.gat_layers):
            # 注意: myGATConv 返回 (输出特征, 注意力权重)
            h, attn_weights = layer(self.g, h, e_feat, attn_weights)

            # 如果是多头注意力的中间层，对头进行平均
            if i < self.num_layers - 1:
                h = h.mean(dim=1) if len(h.shape) > 2 else h

        return h
def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, model_):
    prefix = './data'
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_issue.npz') #[-1, 2]  [[1,2],[2,3]]
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_issue.npz')
    device = torch.device('cpu')

    pca = PCA(n_components=128)
    features_issue = np.load(prefix + '/features_issue.npy')
    features_reduced = pca.fit_transform(features_issue)
    features_0 = torch.tensor(features_reduced, dtype=torch.float32)
    features_1 = np.random.normal(0, 1/np.sqrt(128), (num_assignees, 128))
    features_2 = np.random.normal(0, 1/np.sqrt(128), (num_components, 128))
    features = [features_0, features_1, features_2]
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    elif feats_type == 2:
        num_all_node = num_user + num_artist + num_tag
        features_list = torch.eye(num_all_node).to(device)
    elif feats_type == 3:
        features_list = [torch.FloatTensor(f).to(device) for f in features]
        in_dims = [features.shape[1] for features in features_list]
    train_pos_user_artist = train_val_test_pos_user_artist['train']
    val_pos_user_artist = train_val_test_pos_user_artist['val']
    test_pos_user_artist = train_val_test_pos_user_artist['test']
    train_neg_user_artist = train_val_test_neg_user_artist['train']
    val_neg_user_artist = train_val_test_neg_user_artist['val']
    test_neg_user_artist = train_val_test_neg_user_artist['test']
    y_true_test = np.array([1] * len(test_pos_user_artist) + [0] * len(test_neg_user_artist))
    print(len(y_true_test))
    train_pos_user_artist[:, 1] += num_user
    val_pos_user_artist[:, 1] += num_user
    test_pos_user_artist[:, 1] += num_user
    train_neg_user_artist[:, 1] += num_user
    val_neg_user_artist[:, 1] += num_user
    test_neg_user_artist[:, 1] += num_user

    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    print(type(g))
    print(f"Total edges: {g.number_of_edges()}, Nodes: {g.number_of_nodes()}")

    auc_list = []
    ap_list = []
    heads = ([num_heads] * num_layers) + [1]
    for _ in range(repeat):
        if model_ == 'gat':  # 修改原GAT为新的NewGAT
            print("Using NewGAT with edge type features")
            net = NewGAT(
                g=g,
                num_layers=num_layers,
                layer_dim=hidden_dim,  # 统一维度
                num_heads=num_heads,
                activation=F.elu,
                feat_drop=dropout_rate,
                attn_drop=dropout_rate,
                negative_slope=0.01,
                residual=False,
                in_dims=in_dims,
                type_mask=type_mask
            ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_artist))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_artist), shuffle=False)
        print('train iteration count of each epoch :%d ,val iteration count of each epoch: %d' % (train_pos_idx_generator.num_iterations(),val_idx_generator.num_iterations()))
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_user_artist_batch = train_pos_user_artist[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_user_artist), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_user_artist_batch = train_neg_user_artist[train_neg_idx_batch].tolist()
                t1 = time.time()
                dur1.append(t1 - t0)
                hid_feature = net(features_list)

                # list transposition
                train_pos_user_artist_batch = list(map(list, zip(*train_pos_user_artist_batch)))
                train_neg_user_artist_batch = list(map(list, zip(*train_neg_user_artist_batch)))
                pos_embedding_user=hid_feature[train_pos_user_artist_batch[0]]
                pos_embedding_artist = hid_feature[train_pos_user_artist_batch[0]]
                neg_embedding_user = hid_feature[train_neg_user_artist_batch[0]]
                neg_embedding_artist = hid_feature[train_neg_user_artist_batch[0]]
                print(len(neg_embedding_user))
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_artist_batch = val_pos_user_artist[val_idx_batch].tolist()
                    val_neg_user_artist_batch = val_neg_user_artist[val_idx_batch].tolist()
                    # list transposition
                    val_pos_user_artist_batch = list(map(list, zip(*val_pos_user_artist_batch)))
                    val_neg_user_artist_batch = list(map(list, zip(*val_neg_user_artist_batch)))

                    hid_feature = net(features_list)
                    pos_embedding_user = hid_feature[val_pos_user_artist_batch[0]]
                    pos_embedding_artist = hid_feature[val_pos_user_artist_batch[0]]
                    neg_embedding_user = hid_feature[val_neg_user_artist_batch[0]]
                    neg_embedding_artist = hid_feature[val_neg_user_artist_batch[0]]

                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_artist), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_user_artist[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_user_artist[test_idx_batch].tolist()
                test_pos_user_artist_batch = list(map(list, zip(*test_pos_user_artist_batch)))
                test_neg_user_artist_batch = list(map(list, zip(*test_neg_user_artist_batch)))

                hid_feature=net(features_list)
                pos_embedding_user = hid_feature[test_pos_user_artist_batch[0]]
                pos_embedding_artist = hid_feature[test_pos_user_artist_batch[0]]
                neg_embedding_user = hid_feature[test_neg_user_artist_batch[0]]
                neg_embedding_artist = hid_feature[test_neg_user_artist_batch[0]]

                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
            print(len(y_proba_test))
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    print('program start at ', time.localtime())
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector.' +
                         '2 - all eye vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=100000, help='Batch size. Default is 100000.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='LastFM', help='Postfix for the saved model and result. Default is LastFM.')
    ap.add_argument('--model', default='gat')
    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix, args.model)
    print('program finished at ', time.localtime())