import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score)
import random
import time
import os
from datetime import datetime
import argparse
import math


class HGTLinkPredictor(nn.Module):
    """HGT model for heterogeneous link prediction"""

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers,
                 node_types, edge_types, dropout=0.2):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Create node type to index mapping
        self.node_type_to_idx = {node_type: i for i, node_type in enumerate(node_types)}

        # Input projection for each node type (using torch_geometric Linear for lazy initialization)
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            self.input_projections[node_type] = Linear(-1, hidden_channels)

        # HGT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=(node_types, edge_types),
                num_heads=num_heads,
                group='sum'
            )
            self.convs.append(conv)

        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            layer_norm_dict = nn.ModuleDict()
            for node_type in node_types:
                layer_norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.layer_norms.append(layer_norm_dict)

        # Output projection
        self.output_projection = nn.Linear(hidden_channels, out_channels)

        # Link prediction decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_channels // 2, 1)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters for non-lazy modules
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters for non-lazy modules"""
        for module in self.modules():
            if isinstance(module, nn.Linear):  # Only regular nn.Linear, not torch_geometric Linear
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_dict, edge_index_dict):
        # Input projection for each node type
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.relu(self.input_projections[node_type](x))
            h_dict[node_type] = self.dropout(h_dict[node_type])

        # Apply HGT layers with residual connections
        for i, conv in enumerate(self.convs):
            h_dict_res = h_dict.copy()

            # HGT convolution
            h_dict = conv(h_dict, edge_index_dict)

            # Add residual connection and layer normalization
            for node_type in self.node_types:
                h_dict[node_type] = h_dict[node_type] + h_dict_res[node_type]
                h_dict[node_type] = self.layer_norms[i][node_type](h_dict[node_type])
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = self.dropout(h_dict[node_type])

        # Final output projection
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.output_projection(h)

        return out_dict

    def decode(self, z_dict, edge_label_index, edge_type):
        """Decode edge probabilities for link prediction"""
        src_type, _, dst_type = edge_type

        # Get embeddings for source and destination nodes
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        # Concatenate embeddings
        edge_features = torch.cat([z_src, z_dst], dim=-1)

        # Pass through decoder
        logits = self.decoder(edge_features)
        return logits.squeeze(-1)


class JiraDataLoader:
    """Enhanced data loader for Jira heterogeneous graph data"""

    def __init__(self, data_path='./'):
        self.data_path = data_path

    def load_data(self):
        """Load complete Jira dataset"""
        print("=" * 60)
        print("LOADING JIRA DATASET FOR HGT")
        print("=" * 60)

        # Load issue features
        print("Loading issue features...")
        try:
            # features_issue = np.load(f'{self.data_path}features-gptllama/sling-features-Gptllama.npy')
            features_issue = np.load(f'{self.data_path}spark/spark-features-llama.npy')

            print(f"Issue features shape: {features_issue.shape}")
        except Exception as e:
            print(f"Error loading issue features: {e}")
            raise

        # Load node indices
        print("Loading node indices...")
        try:
            issues_df = pd.read_csv(f'{self.data_path}index/issue_index.txt', sep=' ', header=None,
                                    names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
            assignees_df = pd.read_csv(f'{self.data_path}index/assignee_index.txt', sep=' ', header=None,
                                       names=['assignee_id', 'assignee'], keep_default_na=False, encoding='utf-8')
            components_df = pd.read_csv(f'{self.data_path}index/component_index.txt', sep=' ', header=None,
                                        names=['component_id', 'component'], keep_default_na=False, encoding='utf-8')
        except Exception as e:
            print(f"Error loading node indices: {e}")
            raise

        # Extract numeric IDs and create mappings
        issue_indices = issues_df['issue_id'].values.astype(int)
        assignee_indices = assignees_df['assignee_id'].values.astype(int)
        component_indices = components_df['component_id'].values.astype(int)

        # Count nodes
        num_issues = len(issue_indices)
        num_assignees = len(assignee_indices)
        num_components = len(component_indices)

        print(f"Graph Statistics:")
        print(f"  Issues: {num_issues}")
        print(f"  Assignees: {num_assignees}")
        print(f"  Components: {num_components}")
        print(f"  Total nodes: {num_issues + num_assignees + num_components}")

        # Create node features
        features_issue_tensor = torch.FloatTensor(features_issue)
        features_assignee = torch.randn(num_assignees, 64)
        features_component = torch.randn(num_components, 64)

        # Load edges
        print("Loading edge relationships...")
        edges_data = self._load_all_edges()

        # Create local index mappings
        issue_to_local = {idx: i for i, idx in enumerate(issue_indices)}
        assignee_to_local = {idx: i for i, idx in enumerate(assignee_indices)}
        component_to_local = {idx: i for i, idx in enumerate(component_indices)}

        # Convert to local indices
        local_edges = self._convert_all_edges_to_local(
            edges_data, issue_to_local, assignee_to_local, component_to_local)

        # Load train/val/test splits
        print("Loading train/val/test splits...")
        splits = self._load_splits(issue_indices)

        # Create HeteroData object
        data = HeteroData()

        # Add node features
        data['issue'].x = features_issue_tensor
        data['assignee'].x = features_assignee
        data['component'].x = features_component

        # Add edge indices
        for edge_type, edge_tensor in local_edges.items():
            data[edge_type].edge_index = edge_tensor

        print("Dataset loaded successfully!")
        print(f"Edge types: {list(local_edges.keys())}")
        for edge_type, edge_tensor in local_edges.items():
            print(f"  {edge_type}: {edge_tensor.shape[1]} edges")

        return data, splits

    def _load_all_edges(self):
        """Load all edge types"""
        edges_data = {}

        # Issue-Assignee edges
        try:
            issue_assignee_df = pd.read_csv(f'{self.data_path}index/issue_assignee_index.txt', sep=' ', header=None,
                                            names=['issue_id', 'assignee_id'], keep_default_na=False, encoding='utf-8')
            edges_data['issue_assignee'] = issue_assignee_df.values.astype(int)
            print(f"  Issue-Assignee edges: {len(edges_data['issue_assignee'])}")
        except Exception as e:
            print(f"  Warning: Could not load issue-assignee edges: {e}")
            edges_data['issue_assignee'] = np.array([]).reshape(0, 2)

        # Issue-Component edges
        try:
            issue_component_df = pd.read_csv(f'{self.data_path}index/issue_component_index.txt', sep=' ', header=None,
                                             names=['issue_id', 'component_id'], keep_default_na=False,
                                             encoding='utf-8')
            edges_data['issue_component'] = issue_component_df.values.astype(int)
            print(f"  Issue-Component edges: {len(edges_data['issue_component'])}")
        except Exception as e:
            print(f"  Warning: Could not load issue-component edges: {e}")
            edges_data['issue_component'] = np.array([]).reshape(0, 2)

        # Issue-Issue edges
        try:
            with open(f'{self.data_path}index/issue_issue_index.txt', 'r') as f:
                first_line = f.readline().strip()
                num_cols = len(first_line.split(' '))

            if num_cols == 3:
                issue_issue_df = pd.read_csv(f'{self.data_path}index/issue_issue_index.txt', sep=' ', header=None,
                                             names=['issue_id', 'issue_id_1', 'issue_link'], keep_default_na=False,
                                             encoding='utf-8')
                edges_data['issue_issue'] = issue_issue_df[['issue_id', 'issue_id_1']].values.astype(int)
            else:
                issue_issue_df = pd.read_csv(f'{self.data_path}index/issue_issue_index.txt', sep=' ', header=None,
                                             names=['issue_id', 'issue_id_1'], keep_default_na=False, encoding='utf-8')
                edges_data['issue_issue'] = issue_issue_df.values.astype(int)
            print(f"  Issue-Issue edges: {len(edges_data['issue_issue'])}")
        except Exception as e:
            print(f"  Warning: Could not load issue-issue edges: {e}")
            edges_data['issue_issue'] = np.array([]).reshape(0, 2)

        return edges_data

    def _convert_all_edges_to_local(self, edges_data, issue_to_local, assignee_to_local, component_to_local):
        """Convert all edges to local indices and create PyG format"""
        local_edges = {}

        # Issue-Assignee (bidirectional)
        if edges_data['issue_assignee'].size > 0:
            ia_local = self._convert_edges(edges_data['issue_assignee'], issue_to_local, assignee_to_local)
            if ia_local.size > 0:
                local_edges[('issue', 'assigned_to', 'assignee')] = torch.LongTensor(ia_local.T)
                local_edges[('assignee', 'works_on', 'issue')] = torch.LongTensor(ia_local[:, [1, 0]].T)
            else:
                local_edges[('issue', 'assigned_to', 'assignee')] = torch.LongTensor(2, 0)
                local_edges[('assignee', 'works_on', 'issue')] = torch.LongTensor(2, 0)
        else:
            local_edges[('issue', 'assigned_to', 'assignee')] = torch.LongTensor(2, 0)
            local_edges[('assignee', 'works_on', 'issue')] = torch.LongTensor(2, 0)

        # Issue-Component (bidirectional)
        if edges_data['issue_component'].size > 0:
            ic_local = self._convert_edges(edges_data['issue_component'], issue_to_local, component_to_local)
            if ic_local.size > 0:
                local_edges[('issue', 'belongs_to', 'component')] = torch.LongTensor(ic_local.T)
                local_edges[('component', 'contains', 'issue')] = torch.LongTensor(ic_local[:, [1, 0]].T)
            else:
                local_edges[('issue', 'belongs_to', 'component')] = torch.LongTensor(2, 0)
                local_edges[('component', 'contains', 'issue')] = torch.LongTensor(2, 0)
        else:
            local_edges[('issue', 'belongs_to', 'component')] = torch.LongTensor(2, 0)
            local_edges[('component', 'contains', 'issue')] = torch.LongTensor(2, 0)

        # Issue-Issue
        if edges_data['issue_issue'].size > 0:
            ii_local = self._convert_edges(edges_data['issue_issue'], issue_to_local, issue_to_local)
            if ii_local.size > 0:
                local_edges[('issue', 'links_to', 'issue')] = torch.LongTensor(ii_local.T)
            else:
                local_edges[('issue', 'links_to', 'issue')] = torch.LongTensor(2, 0)
        else:
            local_edges[('issue', 'links_to', 'issue')] = torch.LongTensor(2, 0)

        return local_edges

    def _convert_edges(self, edges, src_mapping, dst_mapping):
        """Convert edge indices using mappings"""
        if edges.size == 0:
            return np.array([]).reshape(0, 2)

        if len(edges.shape) == 1:
            edges = edges.reshape(1, -1)

        if edges.shape[1] > 2:
            edges = edges[:, :2]

        local_edges = []
        for edge in edges:
            if edge[0] in src_mapping and edge[1] in dst_mapping:
                local_edges.append([src_mapping[edge[0]], dst_mapping[edge[1]]])

        return np.array(local_edges) if local_edges else np.array([]).reshape(0, 2)

    def _load_splits(self, issue_indices):
        """Load train/val/test splits with robust error handling"""
        try:
            # Try sparse matrix format first
            train_val_test_pos = sp.load_npz(f'{self.data_path}train_val_test_pos_issue.npz')
            train_val_test_neg = sp.load_npz(f'{self.data_path}train_val_test_neg_issue.npz')
            return self._process_splits_sparse(train_val_test_pos, train_val_test_neg, issue_indices)
        except:
            try:
                # Try numpy array format
                pos_data = np.load(f'{self.data_path}train_val_test_pos_issue.npz')
                neg_data = np.load(f'{self.data_path}train_val_test_neg_issue.npz')
                return self._process_splits_numpy(pos_data, neg_data, issue_indices)
            except:
                # Create default splits
                print("Warning: Could not load splits, creating default ones")
                return self._create_default_splits(issue_indices)

    def _process_splits_sparse(self, pos_matrix, neg_matrix, issue_indices):
        """Process sparse matrix splits"""
        issue_to_local = {idx: i for i, idx in enumerate(issue_indices)}

        pos_edges = np.array(pos_matrix.nonzero()).T
        neg_edges = np.array(neg_matrix.nonzero()).T

        pos_local = self._convert_edges(pos_edges, issue_to_local, issue_to_local)
        neg_local = self._convert_edges(neg_edges, issue_to_local, issue_to_local)

        return self._create_splits_from_edges(pos_local, neg_local)

    def _process_splits_numpy(self, pos_data, neg_data, issue_indices):
        """Process numpy array splits"""
        issue_to_local = {idx: i for i, idx in enumerate(issue_indices)}

        splits = {}
        keys = ['train', 'val', 'test'] if 'train' in pos_data.keys() else sorted(pos_data.keys())

        for i, split_name in enumerate(['train', 'val', 'test']):
            key = keys[i] if i < len(keys) else keys[0]

            pos_edges = pos_data[key] if key in pos_data else np.array([])
            neg_edges = neg_data[key] if key in neg_data else np.array([])

            pos_local = self._convert_edges(pos_edges, issue_to_local, issue_to_local)
            neg_local = self._convert_edges(neg_edges, issue_to_local, issue_to_local)

            pos_tensor = torch.LongTensor(pos_local).T if pos_local.size > 0 else torch.LongTensor(2, 0)
            neg_tensor = torch.LongTensor(neg_local).T if neg_local.size > 0 else torch.LongTensor(2, 0)

            splits[split_name] = {'pos': pos_tensor, 'neg': neg_tensor}
            print(f"  {split_name}: {pos_tensor.shape[1]} pos, {neg_tensor.shape[1]} neg")

        return splits

    def _create_splits_from_edges(self, pos_edges, neg_edges):
        """Create train/val/test splits from edge arrays"""
        pos_tensor = torch.LongTensor(pos_edges).T if pos_edges.size > 0 else torch.LongTensor(2, 0)
        neg_tensor = torch.LongTensor(neg_edges).T if neg_edges.size > 0 else torch.LongTensor(2, 0)

        n_pos = pos_tensor.shape[1]
        n_neg = neg_tensor.shape[1]

        # 70-15-15 split
        train_pos_end = int(0.7 * n_pos)
        val_pos_end = int(0.85 * n_pos)

        train_neg_end = int(0.7 * n_neg)
        val_neg_end = int(0.85 * n_neg)

        splits = {
            'train': {
                'pos': pos_tensor[:, :train_pos_end],
                'neg': neg_tensor[:, :train_neg_end]
            },
            'val': {
                'pos': pos_tensor[:, train_pos_end:val_pos_end],
                'neg': neg_tensor[:, train_neg_end:val_neg_end]
            },
            'test': {
                'pos': pos_tensor[:, val_pos_end:],
                'neg': neg_tensor[:, val_neg_end:]
            }
        }

        for split_name in ['train', 'val', 'test']:
            print(f"  {split_name}: {splits[split_name]['pos'].shape[1]} pos, {splits[split_name]['neg'].shape[1]} neg")

        return splits

    def _create_default_splits(self, issue_indices):
        """Create empty default splits"""
        empty_tensor = torch.LongTensor(2, 0)
        return {
            'train': {'pos': empty_tensor, 'neg': empty_tensor},
            'val': {'pos': empty_tensor, 'neg': empty_tensor},
            'test': {'pos': empty_tensor, 'neg': empty_tensor}
        }


def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_seed(seed_type='fixed', fixed_value=42):
    """
    Get seed value based on seed type

    Args:
        seed_type: 'fixed' or 'random' or 'timestamp'
        fixed_value: value to use when seed_type is 'fixed'

    Returns:
        seed value
    """
    if seed_type == 'fixed':
        return fixed_value
    elif seed_type == 'random':
        return random.randint(1, 10000)
    elif seed_type == 'timestamp':
        return int(time.time())
    else:
        raise ValueError(f"Unknown seed_type: {seed_type}. Use 'fixed', 'random', or 'timestamp'")


def find_optimal_threshold(labels, preds):
    """Find optimal classification threshold based on F1 score"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        pred_binary = (preds >= threshold).astype(int)
        try:
            f1 = f1_score(labels, pred_binary)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except:
            continue

    return best_threshold


def get_next_run_id(save_path='hgt_results.csv'):
    """Get next run ID for experiment tracking"""
    if os.path.exists(save_path):
        try:
            df = pd.read_csv(save_path)
            return df['run_id'].max() + 1 if len(df) > 0 and 'run_id' in df.columns else 1
        except:
            return 1
    return 1


def save_results_to_csv(test_metrics, runtime, config, run_id=None, seed=None, save_path='hgt_results.csv'):
    """Save experimental results to CSV"""
    if run_id is None:
        run_id = get_next_run_id(save_path)

    row = {
        'run_id': run_id,
        'AUC': test_metrics['roc_auc'],
        'AP': test_metrics['ap'],
        'Accuracy': test_metrics['accuracy'],
        'Recall': test_metrics['recall'],
        'Precision': test_metrics['precision'],
        'AF1': test_metrics['f1'],
        'Runtime': runtime,
        'Threshold': test_metrics['threshold'],
        'Seed': seed,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_df = pd.DataFrame([row])
    file_exists = os.path.exists(save_path)
    results_df.to_csv(save_path, mode='a', header=not file_exists, index=False)

    print(f"Results {'appended to' if file_exists else 'saved to'} {save_path} (Run ID: {run_id})")
    return results_df


@torch.no_grad()
def evaluate_model(model, data, splits, split_name, device):
    """Comprehensive model evaluation"""
    model.eval()

    pos_edge_index = splits[split_name]['pos'].to(device)
    neg_edge_index = splits[split_name]['neg'].to(device)

    if pos_edge_index.shape[1] == 0 or neg_edge_index.shape[1] == 0:
        print(f"Warning: No {split_name} edges available")
        return {
            'roc_auc': 0.0, 'ap': 0.0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5
        }

    # Prepare labels
    pos_labels = torch.ones(pos_edge_index.shape[1])
    neg_labels = torch.zeros(neg_edge_index.shape[1])
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()

    # Get predictions
    h_dict = model(data.x_dict, data.edge_index_dict)
    preds = model.decode(h_dict, edge_index, ('issue', 'links_to', 'issue'))
    preds = torch.sigmoid(preds).cpu().numpy()

    # Calculate metrics
    roc_auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)

    optimal_threshold = find_optimal_threshold(labels, preds)
    pred_binary = (preds >= optimal_threshold).astype(int)

    accuracy = accuracy_score(labels, pred_binary)
    precision = precision_score(labels, pred_binary, zero_division=0)
    recall = recall_score(labels, pred_binary, zero_division=0)
    f1 = f1_score(labels, pred_binary, zero_division=0)

    return {
        'roc_auc': roc_auc,
        'ap': ap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': optimal_threshold
    }


def train_one_epoch(model, data, splits, optimizer, device):
    """Train model for one epoch"""
    model.train()

    pos_edge_index = splits['train']['pos'].to(device)
    neg_edge_index = splits['train']['neg'].to(device)

    if pos_edge_index.shape[1] == 0 or neg_edge_index.shape[1] == 0:
        return 0.0

    # Prepare training data
    pos_labels = torch.ones(pos_edge_index.shape[1]).to(device)
    neg_labels = torch.zeros(neg_edge_index.shape[1]).to(device)
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([pos_labels, neg_labels])

    # Shuffle for better training
    perm = torch.randperm(labels.size(0))
    edge_index = edge_index[:, perm]
    labels = labels[perm]

    # Forward pass
    optimizer.zero_grad()
    h_dict = model(data.x_dict, data.edge_index_dict)
    preds = model.decode(h_dict, edge_index, ('issue', 'links_to', 'issue'))

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(preds, labels)

    # Backward pass
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()


def count_parameters_after_init(model):
    """Count parameters after model initialization"""
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params
    except:
        return "Unknown (lazy modules not initialized)"


def run_hgt_experiment(seed_type='fixed', fixed_seed=42, data_path='../AIL/data/'):
    """
    Main experimental pipeline for HGT

    Args:
        seed_type: 'fixed', 'random', or 'timestamp'
        fixed_seed: seed value to use when seed_type is 'fixed'
        data_path: path to data directory
    """

    print("=" * 80)
    print("HGT HETEROGENEOUS LINK PREDICTION EXPERIMENT")
    print("=" * 80)

    # Set random seed for reproducibility
    seed = get_seed(seed_type, fixed_seed)
    set_random_seeds(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Start timing
    start_time = time.time()

    # Load dataset
    print("\n" + "=" * 50)
    print("LOADING DATASET")
    print("=" * 50)

    loader = JiraDataLoader(data_path=data_path)
    data, splits = loader.load_data()
    data = data.to(device)

    # Model configuration
    config = {
        'hidden_channels': 128,
        'out_channels': 64,
        'num_heads': 4,  # HGT-specific parameter
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'epochs': 200,
        'patience': 20,
        'min_improvement': 1e-4,
        'seed_type': seed_type,
        'seed_value': seed
    }

    print(f"\n" + "=" * 50)
    print("MODEL CONFIGURATION")
    print("=" * 50)
    print(f"Architecture: HGT (Heterogeneous Graph Transformer)")
    print(f"Hidden dimensions: {config['hidden_channels']}")
    print(f"Output dimensions: {config['out_channels']}")
    print(f"Number of attention heads: {config['num_heads']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Dropout rate: {config['dropout']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Seed type: {config['seed_type']}")
    print(f"Seed value: {config['seed_value']}")

    # Define graph schema
    node_types = ['issue', 'assignee', 'component']
    edge_types = [
        ('issue', 'assigned_to', 'assignee'),
        ('assignee', 'works_on', 'issue'),
        ('issue', 'belongs_to', 'component'),
        ('component', 'contains', 'issue'),
        ('issue', 'links_to', 'issue')
    ]

    print(f"\nGraph Schema:")
    print(f"Node types: {node_types}")
    print(f"Edge types: {len(edge_types)}")

    # Initialize model
    model = HGTLinkPredictor(
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        node_types=node_types,
        edge_types=edge_types,
        dropout=config['dropout']
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False, min_lr=1e-6
    )

    # Training loop
    print(f"\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0

    # Initialize model with first forward pass and count parameters
    print("Initializing model...")
    val_metrics = evaluate_model(model, data, splits, 'val', device)
    param_count = count_parameters_after_init(model)
    print(f"Model parameters: {param_count:,}" if isinstance(param_count, int) else f"Model parameters: {param_count}")

    for epoch in range(1, config['epochs'] + 1):
        # Train
        epoch_loss = train_one_epoch(model, data, splits, optimizer, device)

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            val_metrics = evaluate_model(model, data, splits, 'val', device)
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | LR: {current_lr:.2e}")

            # Early stopping and model saving
            if val_metrics['roc_auc'] > best_val_auc + config['min_improvement']:
                best_val_auc = val_metrics['roc_auc']
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), 'best_hgt_model.pt')
                print(f"  → New best model saved!")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{config['patience']})")

                if patience_counter >= config['patience']:
                    print(f"  → Early stopping triggered!")
                    break

            # Update learning rate
            scheduler.step(val_metrics['roc_auc'])

    # Final evaluation
    print(f"\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    # Load best model
    model.load_state_dict(torch.load('best_hgt_model.pt'))

    # Evaluate on test set
    test_metrics = evaluate_model(model, data, splits, 'test', device)

    # Calculate total runtime
    total_runtime = time.time() - start_time

    print(f"\nHGT Test Results:")
    print(f"  AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  AP:  {test_metrics['ap']:.4f}")
    print(f"  ACC: {test_metrics['accuracy']:.4f}")
    print(f"  PRE: {test_metrics['precision']:.4f}")
    print(f"  REC: {test_metrics['recall']:.4f}")
    print(f"  F1:  {test_metrics['f1']:.4f}")
    print(f"  Runtime: {total_runtime:.2f}s")
    print(f"  Seed: {seed}")

    # Save results
    save_results_to_csv(
        test_metrics,
        total_runtime,
        config,
        seed=seed,
        save_path='hgt_results.csv'
    )

    print(f"\n" + "=" * 80)
    print("HGT EXPERIMENT COMPLETED!")
    print("=" * 80)

    return test_metrics


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HGT Link Prediction Experiment')
    parser.add_argument('--seed-type', type=str, default='random',
                        choices=['fixed', 'random', 'timestamp'],
                        help='Type of seed to use: fixed, random, or timestamp')
    parser.add_argument('--fixed-seed', type=int, default=42,
                        help='Fixed seed value (used when seed-type is fixed)')
    parser.add_argument('--data-path', type=str, default='../AIL/data/',
                        help='Path to data directory')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of experimental runs (for multiple random seeds)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    print(f"Configuration:")
    print(f"  Seed type: {args.seed_type}")
    if args.seed_type == 'fixed':
        print(f"  Fixed seed: {args.fixed_seed}")
    print(f"  Data path: {args.data_path}")
    print(f"  Number of runs: {args.runs}")
    print()

    if args.runs == 1:
        # Single run
        results = run_hgt_experiment(
            seed_type=args.seed_type,
            fixed_seed=args.fixed_seed,
            data_path=args.data_path
        )
    else:
        # Multiple runs (useful for random seeds to get statistics)
        all_results = []
        for run in range(args.runs):
            print(f"\n{'=' * 20} RUN {run + 1}/{args.runs} {'=' * 20}")
            results = run_hgt_experiment(
                seed_type=args.seed_type,
                fixed_seed=args.fixed_seed,
                data_path=args.data_path
            )
            all_results.append(results)

        # Print summary statistics
        print(f"\n{'=' * 50}")
        print(f"SUMMARY STATISTICS ACROSS {args.runs} RUNS")
        print(f"{'=' * 50}")
        metrics = ['roc_auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            values = [r[metric] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")