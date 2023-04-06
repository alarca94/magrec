import torch
from deepctr_torch.layers import DNN

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MemPooling, ASAPooling, GCN
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor

from utils.constants import *


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.hidden_size = kwargs['hidden_size']
        self.layer_sizes = [self.hidden_size * 4] + kwargs['layer_sizes'] + [1]
        self.enable_BN = kwargs['enable_BN']

        self.key_lin = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.fcn_layers = nn.Sequential(*[self.get_fcn_block(l_ix) for l_ix in range(len(self.layer_sizes)-1)])

    def get_fcn_block(self, l_ix):
        block = nn.Sequential()
        block.add_module('lin', nn.Linear(self.layer_sizes[l_ix], self.layer_sizes[l_ix + 1], bias=False))
        if self.enable_BN:
            block.add_module('bn', nn.BatchNorm1d(self.layer_sizes[l_ix + 1])),
        block.add_module('leakyrelu', nn.LeakyReLU())
        return block

    def forward(self, query, key_value, batch_ixs, return_attention_weights=True):
        att_inputs = self.key_lin(key_value)  # [N, E] --> [N, E]

        if query.shape[0] != key_value.shape[0]:
            query = query.repeat_interleave(torch.unique(batch_ixs, return_counts=True)[1], dim=0)


        last_hidden_nn_layer = torch.cat((att_inputs, query, torch.sub(att_inputs, query),
                                          torch.mul(att_inputs, query)), dim=-1)

        # assert last_hidden_nn_layer.shape[-1] == 4 * self.hidden_size

        # Fully Connected Layers
        att_fn_out = self.fcn_layers(last_hidden_nn_layer)
        att_fn_out = att_fn_out.squeeze(-1)

        # Boolean mask (maybe to attend to what they actually need)
        # Softmax
        att_weights = softmax(att_fn_out, batch_ixs).unsqueeze(1)

        out = key_value * att_weights

        if return_attention_weights:
            return out, att_weights

        return out


class Cluster(nn.Module):
    def __init__(self, **kwargs):
        super(Cluster, self).__init__()
        self.heads = kwargs.get('n_heads', 1)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.add_self_loops = kwargs.get('add_self_loops', True)
        self.k_hop = kwargs.get('k_hop', 2)
        self.ratio = kwargs.get('pool_ratio', 0.8)
        self.dropout = kwargs.get('dropout', 0.0)
        self.reuse_att_w = kwargs.get('reuse_att_w', True)
        att_args = {'hidden_size': self.hidden_size,
                    'layer_sizes': [self.hidden_size * 2, self.hidden_size],
                    'enable_BN': kwargs.get('enable_BN', True)}

        if self.reuse_att_w:
            self.cluster_att = Attention(**att_args)
            self.query_att = Attention(**att_args)
            self.cluster_att2 = Attention(**att_args)
            self.query_att2 = Attention(**att_args)
        else:
            self.cluster_att = nn.ModuleList([Attention(**att_args) for _ in range(self.heads)])
            self.query_att = nn.ModuleList([Attention(**att_args) for _ in range(self.heads)])
            self.cluster_att2 = nn.ModuleList([Attention(**att_args) for _ in range(self.heads)])
            self.query_att2 = nn.ModuleList([Attention(**att_args) for _ in range(self.heads)])

        self.lin = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.heads))

    def forward(self, x, edge_index, edge_weight, target_x, batch_ixs):
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, add_self_loops=True, num_nodes=x.size(0))
        N, L = x.size(0), edge_index.size(-1)
        row, col = edge_index[0], edge_index[1]

        # Cluster embedding
        x_q = x[:]
        for _ in range(self.k_hop):
            x_q = scatter(torch.index_select(x_q, 0, row) * edge_weight.view(-1, 1), col, dim=0, dim_size=N, reduce='sum')

        x_c = torch.zeros_like(x)
        E_g = torch.zeros((L, 1), dtype=torch.float, device=x.device)
        for h in range(self.heads):
            # Cluster and query-aware attention
            _, f_1 = self.cluster_att(x_q, x, batch_ixs, return_attention_weights=True)
            _, f_2 = self.query_att(target_x, x, batch_ixs, return_attention_weights=True)

            # Graph attentive convolution (f_1 and f_2 are the edge weights / edge attention coefficients)
            E = F.leaky_relu(f_1[col] + f_2[row])
            # SURGE performs mask_paddings because they work with dense adjacency matrices
            E = softmax(E, col)
            # Dropout positioning of ASAPooling
            E = F.dropout(E, p=self.dropout, training=self.training)
            # Probably missing the skip connection (but add_self_loops is set to True by default)
            x_c += F.leaky_relu(x + self.lin[h](scatter(x[row] * E, col, dim=0, dim_size=N, reduce='sum')))
            E_g += E
        x_c /= self.heads
        E_g /= self.heads

        # Cluster fitness score
        x_q = x_c[:]
        for _ in range(self.k_hop):
            x_q = scatter(torch.index_select(x_q, 0, row) * edge_weight.view(-1, 1), col, dim=0, dim_size=x_q.size(0), reduce='sum')

        cluster_score = []
        for h in range(self.heads):
            # Cluster and query-aware attention
            _, f_1 = self.cluster_att2(x_q, x_c, batch_ixs, return_attention_weights=True)
            _, f_2 = self.query_att2(target_x, x_c, batch_ixs, return_attention_weights=True)

            cluster_score += [f_1 + f_2]
        cluster_score = softmax(torch.stack(cluster_score).mean(0), batch_ixs).view(-1)

        # Graph pooling
        perm = topk(cluster_score, self.ratio, batch_ixs)
        x = x[perm] * cluster_score[perm].view(-1, 1)
        batch_ixs = batch_ixs[perm]

        # Graph coarsening from ASAPooling
        row, col = edge_index
        A = SparseTensor(row=row, col=col, value=edge_weight.view(-1), sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=E_g.view(-1), sparse_sizes=(N, N))
        S = S[:, perm]

        A = S.t() @ A @ S

        if self.add_self_loops:
            A = A.fill_diag(1.)
        else:
            A = A.remove_diag()

        row, col, edge_weight = A.coo()
        edge_index = torch.stack([row, col], dim=0)

        return x, edge_index, edge_weight, batch_ixs, perm


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_sizes, droprate=0.0, use_bn=False, dnn_activation='relu'):
        super(Tower, self).__init__()
        self.fcs = DNN(input_dim, hidden_sizes, dnn_activation, dropout_rate=droprate, use_bn=use_bn)
        self.cls = nn.Linear(hidden_sizes[-1], 1, bias=False)

    def forward(self, x):
        x = self.fcs(x)
        return self.cls(x)


class RecentAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RecentAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, x, batch, last_ixs):
        sections = torch.bincount(batch)
        v_i = torch.index_select(x, 0, last_ixs)
        v_n_repeat = torch.repeat_interleave(v_i, sections, dim=0)

        alpha = softmax(self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(x))), batch)
        s_g_whole = alpha * x
        s_g = scatter_add(s_g_whole, batch, dim=0)

        return s_g


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, variant=MEMPOOL_VARIANT, **kwargs):
        super(GlobalPooling, self).__init__()

        self.sparse = True  # (variant != MEMPOOL_VARIANT)
        self.aux_loss = 0
        if variant == MEMPOOL_VARIANT:
            self.query_gnn = kwargs['query_gnn']
            self.bn = nn.BatchNorm1d(in_channels)
            self.pool_gnn = nn.ModuleList(
                MemPooling(in_channels, in_channels, heads=kwargs['n_heads'],
                           num_clusters=k, tau=kwargs['tau'])
                for k in kwargs['cluster_ks']
            )
            self.forward_fn = self.forward_memgnn
        elif variant == ASAP_VARIANT:
            self.pool_gnn = nn.ModuleList(
                ASAPooling(in_channels, GNN=GCN, ratio=k, dropout=kwargs['droprate'], improved=True, num_layers=kwargs['gnn_layers'])
                for k in kwargs['cluster_ks']
            )
            self.forward_fn = self.forward_asap
        elif SURGE_VARIANT:
            self.pool_gnn = nn.ModuleList(
                Cluster(hidden_size=in_channels, n_heads=kwargs['n_heads'], pool_ratio=k, dropout=kwargs['droprate'], improved=True,
                        enable_BN=kwargs.get('enable_BN', True))
                for k in kwargs['cluster_ks']
            )
            self.forward_fn = self.forward_surge
        else:
            raise NotImplementedError(f'{variant} variant has not been implemented for GlobalPooling!')

        self.gsl = GSLKernel(in_channels, return_sparse=self.sparse)

    def forward_memgnn(self, x, edge_index, target_x, batch):
        query = self.query_gnn(x, edge_index)
        out = self.bn(query)
        for pool in self.pool_gnn:
            out, S = pool(out, batch)
            # self.aux_loss += pool.kl_loss(S)
        # return global_mean_pool(out, batch)
        return out

    def forward_asap(self, x, edge_index, target_x, batch):
        out = x[:]
        edge_weight = edge_index.new_ones((edge_index.shape[1],), dtype=torch.float)
        p_edge_index, p_batch = edge_index[:], batch[:]
        for pool in self.pool_gnn:
            out, p_edge_index, edge_weight, p_batch, _ = pool(out, p_edge_index, edge_weight, p_batch)
        return out

    def forward_surge(self, x, edge_index, target_x, batch):
        out = x[:]
        edge_weight = edge_index.new_ones((edge_index.shape[1],), dtype=torch.float)
        p_edge_index, p_batch = edge_index[:], batch[:]
        for pool in self.pool_gnn:
            out, p_edge_index, edge_weight, p_batch, _ = pool(out, p_edge_index, edge_weight, target_x, p_batch)
        return out

    def forward(self, x, edge_index, target_x, batch):
        mask = None
        batch_size = batch[-1] + 1
        if not self.sparse:
            x, mask = to_dense_batch(x, batch)
        edge_index = self.gsl(x, batch, mask)
        out = self.forward_fn(x, edge_index, target_x, batch)
        self.aux_loss /= batch_size
        return out


class GSLKernel(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_type='cosine', postprocess_type='relative_ranking', droprate=0.0, n_heads=4,
                 return_sparse=False):
        """
        :param kernel_type: it can be 'inner', 'cosine', 'diffusion' or a combination of the type 'cosine,diffusion'.
        Currently, inner product is the only supported kernel type
        """
        super(GSLKernel, self).__init__()
        self.kernel_type = kernel_type
        self.postprocess_type = postprocess_type
        self.return_sparse = return_sparse
        if out_channels is None:
            out_channels = in_channels
        # self.n_heads = n_heads

        if any([t in self.kernel_type for t in ['cosine', 'inner']]):
            self.lins = nn.ModuleList(nn.Linear(in_channels, out_channels, bias=True) for _ in range(n_heads))
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(droprate)
            if kernel_type == 'inner':
                self.kernel_fn = self.inner_kernel
            elif kernel_type == 'cosine':
                self.kernel_fn = self.cosine_kernel

        if self.postprocess_type == 'relative_ranking':
            self.rel_th = 0.5

    def cosine_kernel(self, Z):
        Z = Z / Z.norm(dim=-1).unsqueeze(-1)
        return abs(torch.bmm(Z, Z.transpose(1, 2)))

    def inner_kernel(self, Z):
        return torch.sigmoid(Z @ Z.transpose(1, 2))

    def metric_kernel(self, x, mask):
        adj = []
        for lin in self.lins:
            Z = self.dropout(self.act(lin(x)))
            adj.append(self.kernel_fn(Z))
        adj = torch.stack(adj).mean(0)

        # Remove padded edges from dense adj. batch
        adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(1)
        return adj

    def relative_ranking(self, adj):
        # Flatten, sort, keep % edges with highest score
        sorted_scores = torch.sort(adj.flatten(1, 2), dim=1, descending=True).values
        n_keep_edges = (torch.count_nonzero(sorted_scores, dim=-1) * self.rel_th).to(torch.int64)
        score_th = torch.gather(sorted_scores, dim=1, index=n_keep_edges.unsqueeze(1))
        return torch.gt(adj, score_th.unsqueeze(-1)).float()

    def forward(self, x, batch, mask):
        adj = None
        if mask is None:
            x, mask = to_dense_batch(x, batch)

        if self.kernel_type in ['inner', 'cosine']:
            adj = self.metric_kernel(x, mask)

        if self.postprocess_type == 'relative_ranking':
            adj = self.relative_ranking(adj)

        # Change dense adjacency to sparse representation again
        offset = batch.new_zeros((x.size(0), ))
        offset[1:] += mask.sum(-1).cumsum(0)[:-1]

        if self.return_sparse:
            nnz = adj.nonzero()
            edge_index = nnz[:, 1:] + torch.index_select(offset, 0, nnz[:, 0]).unsqueeze(-1)
            return edge_index.t()

        return adj


class DenseGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.0, bias=True, concat=True,
                 add_self_loops=True):
        super(DenseGATConv, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.add_self_loops= add_self_loops
        self.concat = concat

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj):
        H, C = self.heads, self.out_channels
        B, L, _ = x.shape

        # Get masked connectivity
        adj_mask = torch.where(torch.gt(adj, 0), 0.0, -torch.inf)
        if self.add_self_loops:
            mask = torch.eye(L, dtype=torch.bool).repeat(B, 1, 1)
            adj_mask[mask] = 0.0

        x = self.dropout(x)
        x_proj = self.lin(x).view(B, L, H, C)

        alpha_src = (x_proj * self.att_src.unsqueeze(0)).sum(dim=-1, keepdims=True)
        alpha_dst = (x_proj * self.att_dst.unsqueeze(0)).sum(dim=-1, keepdims=True)

        alpha = self.leaky_relu(alpha_src.permute(0, 2, 3, 1) + alpha_dst.permute(0, 2, 1, 3))
        alpha = self.softmax(alpha + adj_mask.unsqueeze(1))
        out = torch.matmul(alpha, x_proj.transpose(1, 2))

        out = out.transpose(1, 2)

        if self.concat:
            return out.flatten(2)
        else:
            return out.mean(2)


class DenseGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, heads, dropout=0.0, bias=True, add_skip_connection=True):
        super(DenseGAT, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.droprate = dropout
        self.add_skip_connection = add_skip_connection

        l_heads = [1] + [self.heads] * (num_layers - 1)
        self.gats = nn.ModuleList(
            DenseGATConv(h * in_channels, out_channels, heads=heads, dropout=dropout, bias=bias, concat=(lix+1!=num_layers))
            for lix, h in enumerate(l_heads)
        )

    def forward(self, x, adj):
        out = x[:]
        for gat in self.gats:
            out = gat(out, adj)

        if self.add_skip_connection:
            out = out + x

        return out


