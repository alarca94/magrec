import os
import torch
import pandas as pd
import numpy as np

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from torch_geometric.loader import DataLoader

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import coalesce
from tqdm import tqdm

from utils.constants import *
from utils.data_preparation import split_df, neg_sampling, iterative_kcore_filter
from utils.in_out import colored


def create_dataset(graph_type, domains, dataset_class, **kwargs):
    filter_domain = (graph_type in ['disjoint', 'separate-shared'])
    domain_skip_connect = (graph_type == 'interacting' and dataset_class == PyGDataset)
    return dataset_class(domains=domains, filter_domain=filter_domain, domain_skip_connect=domain_skip_connect,
                         **kwargs)


def get_records(data, min_wsize=5, max_wsize=80, requires_graph=True, undirected=False, filter_domain=False,
                add_domain_skip=False, padding_ix=0):
    def get_graph_record(win_df):
        def add_skip_connections(domains, g_pos, unique_domains):
            edge_index = [[], []]
            for d in unique_domains:
                mask = (domains == d).astype(int)
                diff = mask[1:] - mask[:-1]
                source_nodes = np.argwhere(diff == -1)[:, 0]
                if len(source_nodes) > 0:
                    target_nodes = np.argwhere(diff[source_nodes[0]:] == 1)[:, 0] + source_nodes[0] + 1
                    edge_index[0].extend(source_nodes[:len(target_nodes)])
                    edge_index[1].extend(target_nodes)
            if edge_index[0]:
                return g_pos[np.array(edge_index)]
            else:
                return np.empty((2, 0), dtype=int)

        hist_df = win_df.iloc[:-1]

        y = win_df[TARGET_COL].values[-1]
        target_item = win_df[ITEM_COL].values[-1]
        target_user = win_df[USER_COL].values[-1]
        target_domain = win_df[DOMAIN_COL].values[-1]
        target_timestamp = win_df[TIME_COL].iloc[-1]

        if filter_domain:
            hist_df = hist_df[hist_df[DOMAIN_COL] == target_domain]

        if hist_df.empty:
            x = np.array([[target_user, padding_ix, target_domain]])
            edge_index = np.empty((2, 0))
            edge_weight = np.empty((0, 1))
            edge_attr = np.empty((0, 2))
            seq_ixs = np.array([0])
            seq_len = 1
        else:
            hist_df['g_pos'] = hist_df[ITEM_COL].astype('category').cat.codes
            item_feat_cols = [USER_COL, ITEM_COL, DOMAIN_COL]
            x = hist_df[['g_pos'] + item_feat_cols].sort_values('g_pos')[item_feat_cols].drop_duplicates().values
            source_nodes = hist_df['g_pos'].values[:-1]
            target_nodes = hist_df['g_pos'].values[1:]
            edge_index = np.stack((source_nodes, target_nodes))
            if add_domain_skip:
                unique_domains = np.unique(hist_df[DOMAIN_COL].values)
                if len(unique_domains) > 1:
                    edge_index = np.hstack((edge_index,
                                            add_skip_connections(hist_df[DOMAIN_COL].values, hist_df['g_pos'].values,
                                                                 unique_domains)))
            if undirected:
                edge_index = np.hstack((edge_index, np.stack((target_nodes, source_nodes))))
            edge_weight = np.ones((edge_index.shape[1], 1), dtype=float)
            domain_out = hist_df[DOMAIN_COL].values[edge_index[0, :]]
            domain_in = hist_df[DOMAIN_COL].values[edge_index[1, :]]
            edge_attr = np.stack((domain_out, domain_in)).T
            seq_ixs = hist_df['g_pos'].values
            seq_len = hist_df.shape[0]

        return [target_user, x, target_item, target_domain, target_timestamp, y, edge_index, edge_weight, edge_attr,
                seq_ixs, seq_len]

    def get_deepctr_record(win_df):
        hist_df = win_df.iloc[:-1]

        y = win_df[TARGET_COL].iloc[-1]
        user_id = win_df[USER_COL].values[0]
        item_id = win_df[ITEM_COL].iloc[-1]
        domain_id = win_df[DOMAIN_COL].iloc[-1]
        timestamp = win_df[TIME_COL].iloc[-1]

        if hist_df.empty:
            item_id_list = np.array([[padding_ix]])
            item_attr_list = np.array([[domain_id]])
        else:
            item_id_list = hist_df[ITEM_COL].values
            item_attr_list = hist_df[DOMAIN_COL].values

        return [user_id, item_id_list, item_attr_list, item_id, domain_id, timestamp, y, len(item_id_list)]

    get_record = get_graph_record if requires_graph else get_deepctr_record
    data.sort_values(TIME_COL, inplace=True)
    records = []
    for uid, udata in tqdm(data.groupby(USER_COL)):
        for t in range(min_wsize, udata.shape[0] + 1):
            win_df = udata.iloc[max(0, t - max_wsize):t]
            record = get_record(win_df)
            records.append(record)

    return records


def get_df_subsets(df, split_ixs):
    def get_subset(mode):
        return df.loc[torch.where(split_ixs == MODES[mode])[0].numpy().tolist()]

    return [get_subset(mode) for mode in ['train', 'valid', 'test']]


def read_create_tabular_split(dataset, domains, kcore, common_file_name, ctr_ratio_range, filter_domain, split_type,
                              test_ratio, wsize_range=(5, 80), verbose=1, add_padding=True, add_domain_skip=False,
                              rebuild=False):
    data = []
    ctr_ratios, user_pos_items, domain_items = None, None, None
    tabular_path = os.path.join(DATA_PATH, dataset, TABULAR_DATA_FOLDER)
    if not os.path.exists(tabular_path):
        os.mkdir(tabular_path)

    for dix, d in enumerate(domains):
        filename = f'{d}_5.csv'

        print(f'Reading dataset {d}...')
        ddata = pd.read_csv(os.path.join(DATA_PATH, dataset, 'preprocessed', filename))
        ddata[DOMAIN_COL] = dix
        data.append(ddata)
    data = pd.concat(data, axis=0, ignore_index=True)

    # Iterative kcore
    data = iterative_kcore_filter(data, kcore)

    # Encode user_ids and item_ids
    print(f'Encoding user_id and item_id...')
    data[USER_COL] = data[USER_COL].astype('category').cat.codes
    data[ITEM_COL] = data[ITEM_COL].astype('category').cat.codes + add_padding

    supplement = {
        'n_users': data[USER_COL].nunique(),
        'n_items': data[ITEM_COL].nunique() + add_padding,
        'n_domains': data[DOMAIN_COL].nunique()
    }

    trn_file = os.path.join(tabular_path, f'train_{common_file_name}.pkl')
    val_file = os.path.join(tabular_path, f'valid_{common_file_name}.pkl')
    tst_file = os.path.join(tabular_path, f'test_{common_file_name}.pkl')
    if os.path.isfile(trn_file) and os.path.isfile(val_file) and os.path.isfile(tst_file) and not rebuild:
        if verbose == 1:
            print('Reading pickle splits...')
        trn_df = pd.read_pickle(trn_file)
        val_df = pd.read_pickle(val_file)
        tst_df = pd.read_pickle(tst_file)
    else:
        # If neg_sampling, consider all current interactions as positive
        if ctr_ratio_range is not None:
            data[TARGET_COL] = 1
            user_pos_items = data.groupby([USER_COL, DOMAIN_COL]).apply(lambda g: g[ITEM_COL].unique())
            domain_items = data.groupby(DOMAIN_COL).apply(lambda g: set(g[ITEM_COL].unique().tolist()))

        # Add item_id_list as user history
        print('Extracting graphs...')
        records = get_records(data, wsize_range[0], wsize_range[1], requires_graph=True, filter_domain=filter_domain,
                              add_domain_skip=add_domain_skip)
        data = pd.DataFrame(records, columns=[USER_COL, ITEM_LIST_COL, ITEM_COL, DOMAIN_COL, TIME_COL, TARGET_COL,
                                              EDGE_COL, EDGE_WEIGHT_COL, EDGE_ATTR_COL, NODE_SEQ_COL, SEQ_LEN_COL])

        # Train-test split
        print(f'Splitting data...')
        split_ixs = split_df(data, split_type, test_ratio)

        trn_df, val_df, tst_df = get_df_subsets(data, split_ixs)
        if ctr_ratio_range is not None:
            if supplement['n_domains'] == 1:
                ctr_ratios = [ctr_ratio_range[-1]]
            else:
                ctr_ratios = [round(np.random.uniform(*ctr_ratio_range), 2) for _ in range(supplement['n_domains'])]
            supplement['ctr_ratios'] = ctr_ratios

            int_cols = [USER_COL, DOMAIN_COL, TARGET_COL, SEQ_LEN_COL]
            print('Adding negative sampling to training data...')
            trn_df = pd.concat((trn_df, neg_sampling(trn_df, ctr_ratios, user_pos_items, domain_items)), axis=0,
                               ignore_index=True)
            trn_df[int_cols] = trn_df[int_cols].astype(np.int32)
            print('Adding negative sampling to validation data...')
            val_df = pd.concat((val_df, neg_sampling(val_df, ctr_ratios, user_pos_items, domain_items)), axis=0,
                               ignore_index=True)
            val_df[int_cols] = val_df[int_cols].astype(np.int32)
            print('Adding negative sampling to test data...')
            tst_df = pd.concat((tst_df, neg_sampling(tst_df, ctr_ratios, user_pos_items, domain_items)), axis=0,
                               ignore_index=True)
            tst_df[int_cols] = tst_df[int_cols].astype(np.int32)

        if verbose == 1:
            print('Saving data to pickle')
        trn_df.to_pickle(trn_file)
        val_df.to_pickle(val_file)
        tst_df.to_pickle(tst_file)

    return trn_df, val_df, tst_df, supplement


class DeepCTRDataset:
    def __init__(self, domains, filter_domain=False, min_wsize=5, max_wsize=80, dataset='amazon', emb_dim=64, **kwargs):
        self.min_wsize = min_wsize
        self.max_wsize = max_wsize
        ctr_ratio_range = kwargs.get('ctr_ratio_range', None)
        split_type = kwargs.get('split_type', 'stratified')
        test_ratio = kwargs.get('test_ratio', 0.2)
        rebuild = kwargs.get('rebuild', False)
        kcore = kwargs.get('kcore', 1)
        verbose = kwargs.get('verbose', 1)
        self.domains = domains
        self.filter_domain = filter_domain
        self.add_padding = kwargs.get('add_padding', True)

        suffix = ''
        if ctr_ratio_range is not None:
            suffix += '_ns'
        if self.filter_domain:
            suffix += '_fd'
        common_file_name = f'{split_type}_{"_".join(self.domains)}_{kcore}{suffix}'

        split_tuple = read_create_tabular_split(dataset, self.domains, kcore, common_file_name, ctr_ratio_range,
                                                self.filter_domain, split_type, test_ratio, [min_wsize, max_wsize],
                                                verbose, self.add_padding, rebuild)

        self.trn_df, self.val_df, self.tst_df, self.supplement = split_tuple
        if verbose == 1:
            print('Transforming DataFrames to DeepCTR format...')
        use_cols = [USER_COL, ITEM_LIST_COL, ITEM_COL, DOMAIN_COL, TIME_COL, TARGET_COL, NODE_SEQ_COL, SEQ_LEN_COL]
        self.trn_df, self.val_df, self.tst_df = self.trn_df[use_cols], self.val_df[use_cols], self.tst_df[use_cols]

        # for df in [self.trn_df, self.val_df, self.tst_df]:
        #     n_d = df[ITEM_LIST_COL].apply(lambda v: len(np.unique(v[:, 2])))
        #     assert all(n_d == 1)
        def get_orig_seq(x):
            return x[ITEM_LIST_COL][:, 1][x[NODE_SEQ_COL]]

        self.trn_df[ITEM_LIST_COL] = self.trn_df[[ITEM_LIST_COL, NODE_SEQ_COL]].apply(get_orig_seq, axis=1)
        self.val_df[ITEM_LIST_COL] = self.val_df[[ITEM_LIST_COL, NODE_SEQ_COL]].apply(get_orig_seq, axis=1)
        self.tst_df[ITEM_LIST_COL] = self.tst_df[[ITEM_LIST_COL, NODE_SEQ_COL]].apply(get_orig_seq, axis=1)

        self.max_trn_hlen = min(self.trn_df[SEQ_LEN_COL].max(), self.max_wsize)  # - 1
        if self.val_df.shape[0] > 0:
            self.max_val_hlen = min(self.val_df[SEQ_LEN_COL].max(), self.max_wsize)  # - 1
        if self.tst_df.shape[0] > 0:
            self.max_tst_hlen = min(self.tst_df[SEQ_LEN_COL].max(), self.max_wsize)  # - 1
        self.max_hlen = max(self.max_trn_hlen, self.max_val_hlen, self.max_tst_hlen)

        self.feature_columns = [SparseFeat(USER_COL, self.supplement['n_users'], embedding_dim=emb_dim),
                                SparseFeat(ITEM_COL, self.supplement['n_items'], embedding_dim=emb_dim),
                                VarLenSparseFeat(
                                    SparseFeat(ITEM_LIST_COL, self.supplement['n_items'], embedding_dim=emb_dim,
                                               embedding_name=ITEM_COL),
                                    maxlen=self.max_hlen, length_name='seq_len')]

        self.behavior_feature_list = [ITEM_COL]

    @property
    def n_users(self):
        return self.supplement['n_users']

    @property
    def n_items(self):
        return self.supplement['n_items']

    @property
    def n_domains(self):
        return self.supplement['n_domains']

    def process_subset(self, df, max_hlen):
        # assert np.equal(df[ITEM_LIST_COL].apply(len).values, df[SEQ_LEN_COL].values).all()
        seq_len = df[ITEM_LIST_COL].apply(len).values
        hists = pad_sequences(df[ITEM_LIST_COL].values, maxlen=max_hlen, padding='post')
        c_dict = {USER_COL: df[USER_COL].values,
                  ITEM_COL: df[ITEM_COL].values,
                  ITEM_LIST_COL: hists,
                  'seq_len': seq_len}
        x = {name: c_dict[name] for name in get_feature_names(self.feature_columns)}
        x[DOMAIN_COL] = df[DOMAIN_COL].values
        y = df[TARGET_COL].values
        return x, y

    def get_subset(self, df, domain):
        return df[df[DOMAIN_COL] == domain].reset_index(drop=True)

    def get_dataloader(self, x, y, feature_index, batch_size, shuffle=False, drop_last=False):
        if isinstance(x, dict):
            x = [x[feature] for feature in feature_index] + [x[DOMAIN_COL]]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        return DataLoader(dataset=tensor_data, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)

    def get_dataloaders(self, batch_size, feature_index, domains):
        # n_domains = self.n_domains
        trn_loaders, val_loaders, tst_loaders = [], [], []
        if self.filter_domain:
            for domain in domains:
                d = self.domains.index(domain)
                trn_x, trn_y = self.process_subset(self.get_subset(self.trn_df, domain=d), max_hlen=self.max_hlen)
                trn_loaders.append(self.get_dataloader(trn_x, trn_y, feature_index, batch_size,
                                                       shuffle=True, drop_last=True))
        else:
            trn_x, trn_y = self.process_subset(self.trn_df, max_hlen=self.max_hlen)
            trn_loaders.append(self.get_dataloader(trn_x, trn_y, feature_index, batch_size,
                                                   shuffle=True, drop_last=True))

        for domain in domains:
            d = self.domains.index(domain)
            val_x, val_y = self.process_subset(self.get_subset(self.val_df, domain=d), max_hlen=self.max_hlen)
            val_loaders.append(self.get_dataloader(val_x, val_y, feature_index, batch_size, shuffle=False,
                                                   drop_last=False))
            tst_x, tst_y = self.process_subset(self.get_subset(self.tst_df, domain=d), max_hlen=self.max_hlen)
            tst_loaders.append(self.get_dataloader(tst_x, tst_y, feature_index, batch_size, shuffle=False,
                                                   drop_last=False))

        return trn_loaders, val_loaders, tst_loaders

    def print_split_info(self):
        trn_instances = self.trn_df[DOMAIN_COL].value_counts()
        val_instances = self.val_df[DOMAIN_COL].value_counts()
        tst_instances = self.tst_df[DOMAIN_COL].value_counts()
        for dix, domain in enumerate(self.domains):
            n_trn = trn_instances.get(dix, 0)
            n_val = val_instances.get(dix, 0)
            n_tst = tst_instances.get(dix, 0)
            n_pos = self.trn_df[self.trn_df[DOMAIN_COL] == dix][TARGET_COL].sum()
            total = n_trn + n_val + n_tst
            print(colored(f'Domain: {domain}', Colors.BLUE) +
                  f' --- Training interactions: {n_trn} ({n_trn / total * 100:.2f}%)'
                  f' --- Valid interactions: {n_val} ({n_val / total * 100:.2f}%)'
                  f' --- Test interactions: {n_tst} ({n_tst / total * 100:.2f}%)'
                  f' --- PNR: {n_pos / (total - n_pos):.2f}')
        total = trn_instances.sum() + val_instances.sum() + tst_instances.sum()
        print(colored(f'Overall', Colors.BLUE) +
              f' --- Training interactions: {trn_instances.sum()} ({trn_instances.sum() / total * 100:.2f}%)'
              f' --- Valid interactions: {val_instances.sum()} ({val_instances.sum() / total * 100:.2f}%)'
              f' --- Test interactions: {tst_instances.sum()} ({tst_instances.sum() / total * 100:.2f}%)')


class PyGDataset(InMemoryDataset):
    def __init__(self, domains, min_wsize=5, max_wsize=80, dataset='amazon', **kwargs):
        super(PyGDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.min_wsize = min_wsize
        self.max_wsize = max_wsize

        ctr_ratio_range = kwargs.get('ctr_ratio_range', None)
        split_type = kwargs.get('split_type', 'stratified')
        test_ratio = kwargs.get('test_ratio', 0.2)
        kcore = kwargs.get('kcore', 1)
        rebuild = kwargs.get('rebuild', False)
        verbose = kwargs.get('verbose', 1)
        self.filter_domain = kwargs.get('filter_domain', False)
        self.add_domain_skip = kwargs.get('domain_skip_connect', False)
        self.add_padding = kwargs.get('add_padding', True)

        processed_path = os.path.join(DATA_PATH, dataset, GRAPH_DATA_FOLDER)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path)
        suffix = ''
        if ctr_ratio_range is not None:
            suffix += '_ns'
        if self.filter_domain:
            suffix += '_fd'
        if self.add_domain_skip:
            suffix += '_dsc'
        self.domains = domains
        common_file_name = f'{split_type}_{"_".join(self.domains)}_{kcore}{suffix}'
        processed_file_name = f'{common_file_name}.pt'

        if os.path.isfile(os.path.join(processed_path, processed_file_name)) and not rebuild:
            print(f'Loading processed dataset {processed_file_name}...')
            self.data, self.slices, self.supplement = torch.load(os.path.join(processed_path, processed_file_name))
            # self.supplement['domain_ids'] = torch.LongTensor([0] * len(self.supplement['split_ixs']))
            # torch.save((self.data, self.slices, self.supplement), os.path.join(processed_path, processed_file_name))
        else:
            split_tuple = read_create_tabular_split(dataset, self.domains, kcore, common_file_name, ctr_ratio_range,
                                                    self.filter_domain, split_type, test_ratio, [min_wsize, max_wsize],
                                                    verbose, self.add_padding, self.add_domain_skip, rebuild=False)

            trn_df, val_df, tst_df, self.supplement = split_tuple

            # Test: Filter records without edges
            # trn_df = trn_df[trn_df[EDGE_COL].apply(lambda v: len(v) > 0)]
            # val_df = val_df[val_df[EDGE_COL].apply(lambda v: len(v) > 0)]
            # tst_df = tst_df[tst_df[EDGE_COL].apply(lambda v: len(v) > 0)]

            print('Building graphs...')
            graphs = sum([self.build_graphs(df) for df in [trn_df, val_df, tst_df]], [])
            self.data, self.slices = self.collate(graphs)
            self.supplement['split_ixs'] = torch.LongTensor([MODES['train']] * trn_df.shape[0] +
                                                            [MODES['valid']] * val_df.shape[0] +
                                                            [MODES['test']] * tst_df.shape[0])
            self.supplement['domain_ids'] = torch.LongTensor(np.hstack((trn_df[DOMAIN_COL].values,
                                                                        val_df[DOMAIN_COL].values,
                                                                        tst_df[DOMAIN_COL].values)))
            # graphs = self.extract_graphs(data)

            torch.save((self.data, self.slices, self.supplement), os.path.join(processed_path, processed_file_name))
            print('Dataset created and saved!')

    @property
    def n_users(self):
        return self.supplement['n_users']

    @property
    def n_items(self):
        return self.supplement['n_items']

    @property
    def n_domains(self):
        return self.supplement['n_domains']

    def get_subset(self, mode, domain=None):
        cond_mode = self.supplement['split_ixs'] == MODES[mode]
        if domain is None:
            return Subset(self, torch.where(cond_mode)[0].numpy().tolist())
        else:
            cond_domain = self.supplement['domain_ids'] == domain
            return Subset(self, torch.where(torch.bitwise_and(cond_mode, cond_domain))[0].numpy().tolist())

    def get_dataloaders(self, batch_size, domains):
        # n_domains = self.n_domains
        trn_loaders, val_loaders, tst_loaders = [], [], []
        if self.filter_domain:
            for domain in domains:
                d = self.domains.index(domain)
                trn_loaders.append(
                    DataLoader(self.get_subset('train', domain=d), batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=8))
        else:
            trn_loaders.append(
                DataLoader(self.get_subset('train'), batch_size=batch_size, shuffle=True, drop_last=True,
                           num_workers=8))

        for domain in domains:
            d = self.domains.index(domain)
            val_loaders.append(DataLoader(self.get_subset('valid', domain=d), batch_size=batch_size, shuffle=False,
                                          drop_last=False))
            tst_loaders.append(DataLoader(self.get_subset('test', domain=d), batch_size=batch_size, shuffle=False,
                                          drop_last=False))

        return trn_loaders, val_loaders, tst_loaders

    def build_graphs(self, df, verbose=1):
        def get_graph(row):
            edge_index, edge_weight = coalesce(torch.LongTensor(row[EDGE_COL]), torch.FloatTensor(row[EDGE_WEIGHT_COL]))
            _, edge_attr = coalesce(torch.LongTensor(row[EDGE_COL]), torch.FloatTensor(row[EDGE_ATTR_COL]),
                                    reduce='max')
            # assert edge_attr.shape == edge_index.shape[::-1]
            return Data(x=torch.LongTensor(row[ITEM_LIST_COL]),
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        edge_attr=edge_attr,
                        node_seq_ixs=torch.LongTensor(row[NODE_SEQ_COL]),
                        seq_len=torch.LongTensor([row[SEQ_LEN_COL]]),
                        y=torch.FloatTensor([row[TARGET_COL]]),
                        target_x=torch.LongTensor([row[USER_COL], row[ITEM_COL], row[DOMAIN_COL]]).unsqueeze(0)
                        )

        if verbose == 1:
            tqdm.pandas()
            return df.progress_apply(get_graph, axis=1).tolist()
        else:
            return df.apply(get_graph, axis=1).tolist()

    def print_split_info(self):
        total = np.zeros((3,), dtype=int)
        for dix, domain in enumerate(self.domains):
            dinstances = torch.masked_select(self.supplement['split_ixs'], self.supplement['domain_ids'] == dix)
            n_instances = dinstances.unique(return_counts=True)[1]
            total += n_instances.numpy()
            ratios = n_instances / sum(n_instances) * 100
            print(colored(f'Domain: {domain}', Colors.BLUE) +
                  f' --- Training interactions: {n_instances[0]} ({ratios[0]:.2f}%)'
                  f' --- Valid interactions: {n_instances[1]} ({ratios[1]:.2f}%)'
                  f' --- Test interactions: {n_instances[2]} ({ratios[2]:.2f}%)')
        print(colored(f'Overall', Colors.BLUE) +
              f' --- Training interactions: {total[0]} ({total[0] / total.sum() * 100:.2f}%)'
              f' --- Valid interactions: {total[1]} ({total[1] / total.sum() * 100:.2f}%)'
              f' --- Test interactions: {total[2]} ({total[2] / total.sum() * 100:.2f}%)')
