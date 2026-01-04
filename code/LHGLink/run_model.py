import time
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score, \
    adjusted_rand_score, f1_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_LastFM_data
from utils.tools import index_generator, parse_minibatch_LastFM
from model import MAGNN_lp

# Params
num_ntype = 3
dropout_rate = 0.3
lr = 0.005
weight_decay = 0.001
etypes_lists = [[[0, 0], [0, 1, 0], [0, 2, 0]]]
use_masks = [[True, True, True]]
no_masks = [[False] * 3]
expected_metapaths = [
    [(0, 0), (0, 1, 0), (0,  2, 0)]
]
issues = pd.read_csv('data/index/issue_index.txt', sep=' ', header=None, names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
assignees = pd.read_csv('data/index/assignee_index.txt', sep=' ', header=None, names=['assignee_id', 'assignee'], keep_default_na=False, encoding='utf-8')
components = pd.read_csv('data/index/component_index.txt', sep=' ', header=None, names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')

num_issues = len(issues)
num_assignees = len(assignees)
num_components = len(components)

def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ua, edge_metapath_indices_list_ua, features_list, adjM,  type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist = load_LastFM_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]

    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    elif feats_type == 2:
        for i in range(num_ntype):
            num_nodes = (type_mask == i).sum()
            features_list.append(torch.zeros((num_nodes, features_list[i].shape[1])).to(device))

    train_pos_user_artist = train_val_test_pos_user_artist['train']
    val_pos_user_artist = train_val_test_pos_user_artist['val']
    test_pos_user_artist = train_val_test_pos_user_artist['test']
    train_neg_user_artist = train_val_test_neg_user_artist['train']
    val_neg_user_artist = train_val_test_neg_user_artist['val']
    test_neg_user_artist = train_val_test_neg_user_artist['test']
    y_true_test = np.array([1] * len(test_pos_user_artist) + [0] * len(test_neg_user_artist))

    auc_list = []
    ap_list = []
    af1_list = []
    acc_list = []
    re_list = []
    pr_list = []
    for _ in range(repeat):
        net = MAGNN_lp(
            [3], 8, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_artist))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_artist), shuffle=False)
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

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_artist_batch, device, neighbor_samples,
                    use_masks, num_issues)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_neg_user_artist_batch, device, neighbor_samples,
                    no_masks, num_issues)

                t1 = time.time()
                dur1.append(t1 - t0)

                model_output, _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists,
                     train_pos_idx_batch_mapped_lists))
                pos_embedding_user, pos_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

                model_output, _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists,
                     train_neg_idx_batch_mapped_lists))
                neg_embedding_user, neg_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

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
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_pos_user_artist_batch, device, neighbor_samples,
                        no_masks, num_issues)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_neg_user_artist_batch, device, neighbor_samples,
                        no_masks, num_issues)


                    model_output, _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists,
                         val_pos_idx_batch_mapped_lists))
                    pos_embedding_user, pos_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

                    model_output, _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists,
                         val_neg_idx_batch_mapped_lists))
                    neg_embedding_user, neg_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

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

        val_proba_list = []
        val_true_list = []
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_artist), shuffle=False)
        with torch.no_grad():
            for iteration in range(val_idx_generator.num_iterations()):
                # 正向样本处理
                val_idx_batch = val_idx_generator.next()
                val_pos_batch = val_pos_user_artist[val_idx_batch].tolist()
                val_neg_batch = val_neg_user_artist[val_idx_batch].tolist()

                # 正样本预测
                val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, val_pos_batch, device, neighbor_samples, no_masks,
                    num_issues
                )
                model_output, _ = net(
                    (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                pos_embedding_user = model_output[0]
                pos_embedding_artist = model_output[0]
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                val_proba_list.append(torch.sigmoid(pos_out))
                val_true_list.append(torch.ones(len(val_pos_batch)))

                # 负样本预测
                val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, val_neg_batch, device, neighbor_samples, no_masks,
                    num_issues
                )
                model_output, _ = net(
                    (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                neg_embedding_user = model_output[0]
                neg_embedding_artist = model_output[0]
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                val_proba_list.append(torch.sigmoid(neg_out))
                val_true_list.append(torch.zeros(len(val_neg_batch)))

        val_proba = torch.cat(val_proba_list).cpu().numpy()
        val_true = torch.cat(val_true_list).cpu().numpy()

        # 寻找最佳阈值
        def find_optimal_threshold(y_true, y_proba):
            thresholds = np.linspace(0.1, 0.9, 81)
            best_threshold = 0.5
            best_f1 = 0

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                if len(np.unique(y_pred)) > 1:  # 确保有正负样本
                    f1 = f1_score(y_true, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh

            print(f"Validation optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
            return best_threshold

        optimal_threshold = find_optimal_threshold(val_true, val_proba)
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_user_artist[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_user_artist[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_pos_user_artist_batch, device, neighbor_samples,
                    no_masks, num_issues)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_neg_user_artist_batch, device, neighbor_samples,
                    no_masks, num_issues)

                model_output, _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists,
                     test_pos_idx_batch_mapped_lists))
                pos_embedding_user, pos_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

                model_output, _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists,
                     test_neg_idx_batch_mapped_lists))
                neg_embedding_user, neg_embedding_artist = model_output[0], model_output[0]  # 假设只有一种节点类型

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

        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
        acc = accuracy_score(y_true_test, y_pred_test)
        recall = recall_score(y_true_test, y_pred_test)
        precision = precision_score(y_true_test, y_pred_test)
        af1 = f1_score(y_true_test, y_pred_test, average='binary')
        acc_list.append(acc)
        re_list.append(recall)
        pr_list.append(precision)
        af1_list.append(af1)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}'.format(np.mean(auc_list)))
    print('AP_mean = {}'.format(np.mean(ap_list)))
    print('Acc_mean = {}'.format(np.mean(acc_list)))
    print('Recall_mean = {}'.format(np.mean(re_list)))
    print('Precision_mean = {}'.format(np.mean(pr_list)))
    print('AF1_mean = {},'.format(np.mean(af1_list)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=256, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=16, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='LastFM', help='Postfix for the saved model and result. Default is LastFM.')

    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)