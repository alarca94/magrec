import torch

# from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch_geometric.nn import GINConv, MemPooling, GATv2Conv, GCNConv, GAT, Sequential
from torch.nn import functional as F
from models.layers import Cluster, Tower
from models.graph_based.base import GraphModelBase


class MemGNN(GraphModelBase):
    def __init__(self, n_users, n_items, device, params, variant=None):
        super(MemGNN, self).__init__(n_users, n_items, device)

        self.emb_dim = params.emb_dim
        self.hidden_size = self.emb_dim * 2
        self.n_query_layers = params.n_query_layers
        self.cluster_ks = params.cluster_ks
        self.tau = params.tau
        self.n_heads = params.n_heads
        self.lr = params.lr
        self.droprate = params.droprate
        self.run_info = f'MemGNN_E{self.emb_dim}_L{self.n_query_layers}_H{self.n_heads}_LR{self.lr}_DR{self.droprate}_' \
                        f'PR{"-".join(list(map(str, self.cluster_ks)))}_Tau{self.tau}'

        self.u_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.i_emb = nn.Embedding(self.n_items, self.emb_dim, padding_idx=0)

        self.add_regularization_weight(self.u_emb.weight, l2=1e-6)
        self.add_regularization_weight(self.i_emb.weight, l2=1e-6)

        self.query_gnn = Sequential('x, edge_index',
            [(GATv2Conv(self.hidden_size, self.hidden_size, heads=self.n_heads, dropout=self.droprate),
              'x, edge_index -> x')] +
            [(GATv2Conv(self.hidden_size * self.n_heads, self.hidden_size, heads=self.n_heads, dropout=self.droprate),
              'x, edge_index -> x')
            for _ in range(self.n_query_layers)]
        )

        self.pool_gnn = nn.ModuleList(
            MemPooling(self.hidden_size * self.n_heads, self.hidden_size * self.n_heads, heads=self.n_heads, num_clusters=k, tau=self.tau)
            for k in self.cluster_ks
        )

        cls_in_channels = self.hidden_size * (self.n_heads + 1)
        self.cls = Tower(cls_in_channels, cls_in_channels // 2, 1, droprate=self.droprate)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cls.named_parameters()), l2=1e-6)

        self.loss_fn = nn.BCELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # self.tensorboard = SummaryWriter(log_dir=self.log_dir + self.run_info)

        self.to(device)

    def forward(self, x, edge_index, batch, target_x):
        h_x = torch.hstack((self.u_emb(x[:, 0]), self.i_emb(x[:, 1])))
        t_x = torch.hstack((self.u_emb(target_x[:, 0]), self.i_emb(target_x[:, 1])))

        query = self.query_gnn(h_x, edge_index)
        xp = query[:]
        for pool in self.pool_gnn:
            xp, _ = pool(xp, batch)

        out = torch.hstack((xp.squeeze(), t_x))
        out = torch.sigmoid(self.cls(out))
        return out
