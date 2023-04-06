import argparse
import os
import pandas as pd

from utils.aux_funcs import init_seed, domain_map
from utils.constants import AMAZON_CATEGORIES, DATA_PATH
from utils.datasets import PyGDataset
from utils.in_out import load_config


def decompress_amazon_data(domains=None):
    column_map = {
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    }

    for domain in domains:
        # New data filenames does not start with "reviews_" but ends with "_5"
        file_ext = '.json.gz'
        prefix = ''  # 'reviews_'
        suffix = '_5'
        old_filename = f'{prefix}{domain.replace(" ", "_")}{suffix}'
        filename = f'{prefix}{"".join([d[0] for d in domain.split()])}{suffix}'

        if not os.path.exists(os.path.join(DATA_PATH, dataset, 'preprocessed', filename + '.csv')):
            print(f'Decompressing {dataset}-{domain}...')
            ddata = pd.read_json(os.path.join(DATA_PATH, dataset, 'compressed', old_filename + file_ext),
                                 compression='gzip', lines=True)
            ddata = ddata.loc[:, column_map.keys()]
            ddata.rename(column_map, axis=1, inplace=True)

            # Binarize ratings
            # filename = f'{prefix}{domain.replace(" ", "-")}{suffix}'
            ddata['click'] = (ddata['rating'] > 3).astype(float)
            ddata.drop('rating', axis=1, inplace=True)
            if not os.path.exists(os.path.join(DATA_PATH, dataset, 'preprocessed')):
                if not os.path.exists(os.path.join(DATA_PATH, dataset)):
                    os.mkdir(os.path.join(DATA_PATH, dataset))
                os.mkdir(os.path.join(DATA_PATH, dataset, 'preprocessed'))
            ddata.to_csv(os.path.join(DATA_PATH, dataset, 'preprocessed', filename + '.csv'), index=False)
        else:
            print(f'Dataset {dataset}-{domain} is already decompressed & slightly preprocessed!')


def prepare_amazon_dataset(domains, config, mode=None):
    if mode is None or mode == 'interacting':
        print('\nPreparing Interacting dataset...\n')
        PyGDataset(domains, filter_domain=False, dataset='amazon', domain_skip_connect=True, **config)
    if mode is None or mode == 'flattened':
        print('\nPreparing Flattened dataset...\n')
        PyGDataset(domains, filter_domain=False, dataset='amazon', domain_skip_connect=False, **config)
    if mode is None or mode in ['disjoint', 'separate-shared']:
        print('\nPreparing Disjoint dataset...\n')
        PyGDataset(domains, filter_domain=True, dataset='amazon', domain_skip_connect=False, **config)


if __name__ == '__main__':
    init_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='amazon', help='name of dataset')
    parser.add_argument('--mode', '-m', type=str, default=None, help='mode (interacting, disjoint, flattened)')
    parser.add_argument('--domains', type=lambda s: sorted(s.split(',')), default=None)
    parser.add_argument('--only-decompress', dest='only_decompress', action='store_true')
    parser.set_defaults(only_decompress=False)
    args, _ = parser.parse_known_args()
    dataset = args.dataset.lower()

    dataset_path = os.path.join(DATA_PATH, dataset)
    if dataset == 'amazon':
        if args.domains is None:
            args.domains = AMAZON_CATEGORIES

        domains = {d: domain_map(d) for d in args.domains}

        config = load_config(f'{dataset}.yaml')['dataset']

        # Decompress those domains that are still missing
        decompress_path = os.path.join(dataset_path, 'preprocessed')
        decompress_domains = [d for d, nd in domains.items() if not os.path.isfile(os.path.join(decompress_path, f'{nd}_5.csv'))]
        if decompress_domains:
            print(f'Decompressing {",".join(decompress_domains)} domains...')
            decompress_amazon_data(decompress_domains)

        # Create dataset with specified domain combination. By default create it for all model types (DeepCTR and Graph-based)
        if not args.only_decompress:
            prepare_amazon_dataset([domains[d] for d in args.domains], config, mode=args.mode)

    elif dataset == 'taobao':
        raise Exception('Unfortunately, Taobao does not have interaction timestamps for sequential ordering of clicks')
