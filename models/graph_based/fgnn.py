"""
Main code extracted from https://github.com/lyj1998/FGNN
"""


import math
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GatedGraphConv, GAT, GATConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter.scatter import scatter_add
from typing import Optional

from models.layers import Tower
from models.graph_based.base import GraphModelBase


class GRUSet2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper
    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})
        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)
        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i
        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,
    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.
    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(GRUSet2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.rnn = nn.GRU(self.out_channels, self.in_channels,
                          num_layers)
        self.linear = nn.Linear(in_channels * 3, in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.orthogonal_(weight.data)
    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        sections = torch.bincount(batch)
        v_i = torch.split(x, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in
                           v_i)  # repeat |V|_i times for the last node embedding
        # x = x * v_n_repeat

        for i in range(self.processing_steps):
            if i == 0:
                q, h = self.rnn(q_star.unsqueeze(0))
            else:
                q, h = self.rnn(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)

            # e = self.linear(torch.cat((x, q[batch], torch.cat(v_n_repeat, dim=0)), dim=-1)).sum(dim=-1, keepdim=True)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class WeightedGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 weighted=True):
        super(WeightedGATConv, self).__init__('add', node_dim=-3)

        self.weighted = weighted
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)

        # gaussian initialization according to paper
        torch.nn.init.normal_(self.weight, 0, 0.1)
        torch.nn.init.normal_(self.att, 0, 0.1)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        edge_index, edge_attr = add_self_loops_partial(edge_index, edge_attr)

        x = torch.mm(x.squeeze(), self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=edge_index.max() + 1, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index, num_nodes, edge_attr):
        # Compute attention coefficients.
        if edge_attr is not None:
            # alpha = ((torch.cat([x_i, x_j], dim=-1) * self.att) * edge_attr.view(-1, 1, 1)).sum(dim=-1)
            alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, x_i.shape[1]).view(-1, x_i.shape[1], 1)],
                               dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])  # num_nodes

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes


def add_self_loops_partial(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    mask = row == col
    masked_weight = edge_weight[mask]
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    loop_weight = torch.full((num_nodes,), fill_value, device=edge_index.device)
    loop_weight[row[mask]] = masked_weight
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    assert edge_index.shape[-1] == edge_weight.shape[0]
    return edge_index, edge_weight


class FGNN(GraphModelBase):
    def __init__(self, n_users, n_items, domains, device, params, variant=''):
        super(FGNN, self).__init__(n_users, n_items, domains, device)
        self.heads = params.att_heads
        self.hidden_size = params.emb_dim
        self.wgat_layers = params.wgat_layers
        self.gru_layers = params.gru_layers
        self.dnn_hidden_units = params.dnn_hidden_units
        self.droprate = params.droprate
        self.dnn_droprate = params.dnn_dropout
        self.use_bn = params.use_bn
        self.dnn_activation = params.dnn_activation

        self.item_embedding = torch.nn.Embedding(self.n_items, embedding_dim=self.hidden_size, padding_idx=0)
        self.wgats = nn.ModuleList(
            WeightedGATConv(in_channels=self.hidden_size,
                            out_channels=self.hidden_size,
                            heads=self.heads,
                            concat=False,
                            negative_slope=0.2,
                            dropout=self.droprate,
                            bias=True,
                            weighted=True)
            for _ in range(self.wgat_layers)
        )
        self.set2set = GRUSet2Set(in_channels=self.hidden_size, processing_steps=self.gru_layers)
        self.linear = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.tower = Tower(2 * self.hidden_size, self.dnn_hidden_units, droprate=self.dnn_droprate, use_bn=self.use_bn,
                           dnn_activation=self.dnn_activation)
        self.lr = params.lr
        self.run_info = f'FGNN'

        self.finish_init('adam', device)

    def forward(self, data):
        x, edge_index, batch, edge_weight, target_x = data.x, data.edge_index, data.batch, data.edge_weight, data.target_x
        x = self.item_embedding(x[:, 1]).squeeze()
        target_emb = self.item_embedding(target_x[:, 1]).squeeze()

        for wgat in self.wgats:
            x = wgat(x, edge_index, edge_weight.flatten().to(torch.long))

        q_star = self.set2set(x, batch)
        tower_x = torch.cat((target_emb, self.linear(q_star)), dim=-1)
        logits = self.tower(tower_x)
        # logits = torch.diag(torch.mm(self.linear(q_star), target_emb.T))
        return torch.sigmoid(logits)
