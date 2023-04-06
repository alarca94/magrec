import math
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GatedGraphConv, GAT
from torch_geometric.utils import to_undirected

from models.layers import Tower, RecentAttention, GlobalPooling, GSLKernel, DenseGAT
from models.graph_based.base import GraphModelBase
from utils.constants import *
from utils.metrics import CKA_Minibatch


class MAGRec(GraphModelBase):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """

    def __init__(self, n_users, n_items, domains, device, params, variant=''):
        super(MAGRec, self).__init__(n_users, n_items, domains, device)
        self.emb_dim = params.emb_dim
        self.domain_dim = params.domain_dim
        self.hidden_size = self.emb_dim
        self.dnn_hidden_units = params.dnn_hidden_units
        self.n_heads = params.n_heads
        self.cluster_ks = params.cluster_ks
        self.tau = params.tau
        self.gated_layers = params.gated_layers
        self.gat_layers = params.gat_layers
        fcn_inputs = 1
        self.use_short_interest = params.use_short_interest
        self.use_long_interest = params.use_long_interest
        self.use_undirected = params.use_undirected
        self.use_bn = params.use_bn
        self.use_domain_info = params.use_domain_info
        self.normalize_domains = params.normalize_domains
        self.d_option = params.d_option
        self.droprate = params.droprate
        self.dnn_droprate = params.dnn_dropout
        self.dnn_activation = params.dnn_activation
        self.norms = [0]

        self.i_emb = nn.Embedding(self.n_items, self.emb_dim, padding_idx=0)
        self.u_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.d_emb = nn.Parameter(torch.FloatTensor(len(self.domains), self.domain_dim))

        # Domain Transform version 1
        if self.use_domain_info:
            self.transform = nn.Linear(self.emb_dim + self.domain_dim, self.emb_dim, bias=True)

        if self.use_short_interest:
            if self.use_domain_info:
                self.source_lin = nn.Linear(self.domain_dim, self.domain_dim, bias=True)
                if self.d_option == 2:
                    self.target_lin = nn.Linear(self.domain_dim, self.domain_dim, bias=True)
                    self.d_transform_fn = self.transform_opt_2
                elif self.d_option == 1:
                    self.d_transform_fn = self.transform_opt_1
                elif self.d_option == 3:
                    self.target_lin = nn.Linear(2 * self.domain_dim, 1, bias=True)
                    self.d_transform_fn = self.transform_opt_3
                else:
                    self.target_lin1 = nn.Linear(self.domain_dim, self.domain_dim, bias=True)
                    self.target_lin2 = nn.Linear(2 * self.domain_dim, 1, bias=True)
                    self.d_transform_fn = self.transform_opt_4
            self.gated = GatedGraphConv(self.hidden_size, num_layers=self.gated_layers)
            # self.gated = GAT(self.hidden_size, self.hidden_size, num_layers=3, heads=4)
            self.last_item_att = RecentAttention(self.hidden_size)
            fcn_inputs += 1
            self.norms.append(0)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gated.named_parameters()), l2=1e-5)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.last_item_att.named_parameters()), l2=1e-5)

        if self.use_long_interest:
            '''
            if self.use_domain_info:
                pool_params = {'domain_dim': domain_dim}
            '''
            pool_params = {}
            if variant == MEMPOOL_VARIANT:
                query_gnn = GAT(self.hidden_size, self.hidden_size, num_layers=self.gat_layers, heads=self.n_heads, dropout=self.droprate)
                # query_gnn = DenseGAT(self.hidden_size, self.hidden_size, num_layers=1, heads=self.n_heads, dropout=0.0,
                #                      add_skip_connection=True)
                pool_params = {'cluster_ks': self.cluster_ks, 'n_heads': self.n_heads, 'tau': self.tau,
                               'query_gnn': query_gnn}
            elif variant == ASAP_VARIANT:
                pool_params = {'cluster_ks': [0.8, 0.5, 1], 'gnn_layers': 1, 'droprate': self.droprate}
            elif variant == SURGE_VARIANT:
                pool_params = {'cluster_ks': [1], 'n_heads': self.n_heads, 'enable_BN': False,
                               'droprate': self.droprate}

            if variant in [MEMPOOL_VARIANT, ASAP_VARIANT, SURGE_VARIANT]:
                self.global_pool = GlobalPooling(self.hidden_size, variant=variant, **pool_params)
                self.user_lin = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
                fcn_inputs += 1
                self.norms.append(0)
                self.add_regularization_weight(
                    filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.global_pool.named_parameters()),
                    l2=1e-5)
                self.add_auxiliary_loss(self.global_pool.aux_loss)

        fcn_in_channels = fcn_inputs * self.hidden_size
        self.dropout = nn.Dropout(self.droprate)
        # self.batchnorm = nn.BatchNorm1d(fcn_in_channels)
        self.tower = Tower(fcn_in_channels, self.dnn_hidden_units, droprate=self.dnn_droprate, use_bn=self.use_bn,
                           dnn_activation=self.dnn_activation)
        self.cka_logger = CKA_Minibatch()
        self.norms = torch.FloatTensor(self.norms)

        self.add_regularization_weight(self.i_emb.weight, l2=1e-5)
        self.add_regularization_weight(self.u_emb.weight, l2=1e-5)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower.named_parameters()), l2=1e-5)

        self.lr = params.lr
        self.run_info = f'MAGRec_E{self.emb_dim}_V-{variant}_LR{self.lr}'

        self.finish_init('adam', device)
        self.reset_parameters()

    def print_cka(self, reset=False):
        # print(self.cka_comp)
        # print(self.norms)
        # cka = (self.cka_comp[0] / (self.cka_comp[1].sqrt() * self.cka_comp[2].sqrt())).item()
        cka = self.cka_logger.compute()
        print(cka)
        if reset:
            # self.cka_comp.zero_()
            self.cka_logger.reset()
            # self.norms.zero_()
        return cka

    def reset_log_info(self):
        self.cka_logger.reset()
        self.norms.zero_()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @staticmethod
    def unpack_data(data):
        return data.x, data.edge_index, data.batch, data.edge_weight, data.edge_attr.to(torch.int32), data.target_x

    def transform_opt_1(self, source_d, target_d, d_emb):
        proj_d = self.source_lin(d_emb)
        source_d_emb = torch.index_select(proj_d, 0, source_d)
        target_d_emb = torch.index_select(proj_d, 0, target_d)
        return (source_d_emb * target_d_emb).sum(1, keepdims=True)

    def transform_opt_2(self, source_d, target_d, d_emb):
        src_proj_d = self.source_lin(d_emb)
        trg_proj_d = self.target_lin(d_emb)
        source_d_emb = torch.index_select(src_proj_d, 0, source_d)
        target_d_emb = torch.index_select(trg_proj_d, 0, target_d)
        return (source_d_emb * target_d_emb).sum(1, keepdims=True)

    def transform_opt_3(self, source_d, target_d, d_emb):
        proj_d = self.source_lin(d_emb)
        edge_emb = torch.cat((torch.index_select(proj_d, 0, source_d), torch.index_select(proj_d, 0, target_d)), dim=1)
        return self.target_lin(edge_emb)

    def transform_opt_4(self, source_d, target_d, d_emb):
        src_proj_d = self.source_lin(d_emb)
        trg_proj_d = self.target_lin1(d_emb)
        edge_emb = torch.cat((torch.index_select(src_proj_d, 0, source_d), torch.index_select(trg_proj_d, 0, target_d)), dim=1)
        return self.target_lin(edge_emb)

    def forward(self, data):
        x, edge_index, batch, edge_weight, edge_attr, target_x = self.unpack_data(data)
        node_seq_ixs, seq_len = data.node_seq_ixs, data.seq_len

        if self.use_undirected:
            orig_edge_index = edge_index[:]
            edge_index, edge_weight = to_undirected(orig_edge_index, edge_weight, num_nodes=x.size(0))
            _, edge_attr = self.to_undirected(orig_edge_index, edge_attr, num_nodes=x.size(0))

        # Get historical embeddings, candidate item embedding and user embedding
        embedding = self.i_emb(x[:, 1]).squeeze()
        target_emb = self.i_emb(target_x[:, 1]).squeeze()
        user_emb = self.u_emb(target_x[:, 0]).squeeze()
        if self.normalize_domains and self.use_domain_info:
            d_emb = F.normalize(self.d_emb, p=2, dim=1)
        else:
            d_emb = self.d_emb

        if self.use_domain_info:
            embedding = self.transform(torch.hstack((embedding, torch.index_select(d_emb, 0, x[:, 2])))).squeeze()
            target_emb = self.transform(torch.hstack((target_emb, torch.index_select(d_emb, 0, target_x[:, 2])))).squeeze()

        fcn_inputs = []
        self.norms[0] += torch.norm(target_emb).item()
        if self.use_long_interest:
            global_x = torch.index_select(user_emb, 0, batch) * embedding
            '''
            if self.use_domain_info:
                global_x = self.global_pool(global_x, edge_index, target_emb, batch, edge_attr).squeeze()
            '''
            global_x = self.global_pool(global_x, edge_index, target_emb, batch).squeeze()
            global_x = self.dropout(self.user_lin(torch.cat((global_x, user_emb), dim=1)))
            fcn_inputs.append(global_x)
            self.norms[-1] += torch.norm(global_x).item()

        if self.use_short_interest:
            si_edge_weight = None
            if self.use_domain_info:
                # OPTION 1: Single projection for source and target, OPTION 2: Source and Target domain projections
                si_edge_weight = self.d_transform_fn(edge_attr[:, 0], edge_attr[:, 1], d_emb)
            hidden = self.dropout(self.gated(embedding, edge_index, si_edge_weight))  # , edge_attr.float())
            last_node_ixs = seq_len.cumsum(0) - 1
            last_node = torch.index_select(node_seq_ixs, 0, last_node_ixs)
            last_node[1:] += torch.bincount(batch).cumsum(0)[:-1]
            short_x = self.last_item_att(hidden, batch, last_node)
            fcn_inputs.append(short_x)
            self.norms[1] += torch.norm(short_x).item()

        tower_x = torch.cat((target_emb, *fcn_inputs), dim=-1)
        # if self.use_bn:
        #     tower_x = self.batchnorm(tower_x)
        if self.use_short_interest and self.use_long_interest:
            # print()
            # print(self.norms[0] / self.norms[1])
            # diff = torch.pow(s_h - aux_embedding, 2).mean(1).sqrt()
            # cos_sim = F.cosine_similarity(s_h, aux_embedding, dim=1)
            # print()
            # print(diff.min().item(), diff.max().item(), diff.mean().item(), cos_sim.min().item(), cos_sim.max().item(),
            #       cos_sim.mean().item())
            self.cka_logger.update(tower_x[:, self.hidden_size:2 * self.hidden_size],
                                   tower_x[:, 2 * self.hidden_size:])
            # self.cka_comp += get_CKA(short_x, global_x, self.n_batches)

        # z_i_hat = torch.mm(s_h, target_item_embedding.transpose(0, 1))
        # return torch.diag(z_i_hat)
        logits = self.tower(tower_x)
        return torch.sigmoid(logits)
