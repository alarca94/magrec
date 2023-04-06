import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.constants import *


def iterative_kcore_filter(df, kcore, verbose=1):
    def kcore_filter(d, col):
        counts = d[col].value_counts()
        return d[d[col].isin(counts[counts >= kcore].index.values)]
    copy_df = df.copy()
    prev_sz = -1
    if verbose == 1:
        print(f'Starting Iterative K-Core item and user filtering with K = {kcore}...')
        print(f'Initial number of interactions: {df.shape[0]}')
        print(f'Initial number of users: {df[USER_COL].nunique()}')
        print(f'Initial number of items: {df[ITEM_COL].nunique()}')
    while prev_sz != copy_df.shape[0]:
        # Filter by user profile size
        prev_sz = copy_df.shape[0]
        copy_df = kcore_filter(copy_df, USER_COL)
        copy_df = kcore_filter(copy_df, ITEM_COL)
        # copy_df = copy_df.groupby(USER_COL).filter(lambda x: len(x) >= kcore)
        # copy_df = copy_df.groupby(ITEM_COL).filter(lambda x: len(x) >= kcore)

    if verbose == 1:
        print(f'Final number of interactions: {copy_df.shape[0]}')
        print(f'Final number of users: {copy_df[USER_COL].nunique()}')
        print(f'Final number of items: {copy_df[ITEM_COL].nunique()}')
    return copy_df


def split_df(data, split_type, test_ratio, val_ratio=None):
    if val_ratio is None:
        val_ratio = test_ratio

    split_ixs = torch.zeros((data.shape[0]), dtype=torch.long)
    if split_type == 'stratified':
        trn, tst = train_test_split(data, test_size=test_ratio, stratify=data[TARGET_COL].values,
                                    random_state=RNG_SEED)

        trn, val = train_test_split(trn, test_size=val_ratio/(1 - test_ratio), stratify=trn[TARGET_COL].values,
                                    random_state=RNG_SEED)
        split_ixs[val.index.values] = MODES['valid']
        split_ixs[tst.index.values] = MODES['test']
    elif split_type == 'temporal':
        raise NotImplementedError('"temporal" split type has not been implemented yet!')
    elif split_type == 'user_temporal':
        data.sort_values(by=[USER_COL, TIME_COL], inplace=True)
        ucounts = data.groupby(USER_COL).size().values
        uoffsets = ucounts.cumsum()
        if isinstance(test_ratio, float):
            assert isinstance(val_ratio, float)
            assert test_ratio < 1.0
            tst_start_ixs = uoffsets - (ucounts * test_ratio).astype(int)
            val_start_ixs = tst_start_ixs - (ucounts * val_ratio).astype(int)
        elif isinstance(test_ratio, int):
            assert isinstance(val_ratio, int)
            assert all(ucounts > (test_ratio + val_ratio))
            tst_start_ixs = uoffsets - test_ratio
            val_start_ixs = tst_start_ixs - val_ratio
        else:
            raise TypeError('test_ratio is neither int nor float')
        for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
            split_ixs[tix:offset] = MODES['test']
            split_ixs[vix:tix] = MODES['valid']
    else:
        raise NotImplementedError('Specified split type has not been implemented yet!')

    return split_ixs


def neg_sampling(data, ctr_ratios, user_pos_items, domain_items, padding_ix=0):
    neg_df = []

    # items = set(range(n_items))
    # if padding_ix is not None:
    #     items.difference_update({padding_ix})
    n_negs = [int(1 / ctr_ratio) for ctr_ratio in ctr_ratios]

    for did, ddata in data.groupby(DOMAIN_COL):
        items = domain_items[did]
        for uid, udata in tqdm(ddata.groupby(USER_COL)):
            user_neg_items = list(items.difference(user_pos_items.loc[uid, did]))
            n_neg_u = min(n_negs[did], len(user_neg_items))
            for ix, row in udata.iterrows():
                neg_samples = pd.concat([row] * n_neg_u, axis=1).T
                neg_samples[TARGET_COL] = 0
                neg_samples[ITEM_COL] = np.random.choice(user_neg_items, n_neg_u, replace=False)
                neg_df.append(neg_samples)

    return pd.concat(neg_df, ignore_index=True)


