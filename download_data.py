import requests
import gzip
import shutil
import os
import pandas as pd
import argparse

from tqdm import tqdm

from utils.constants import AMAZON_CATEGORIES, DATA_PATH


def fetch_or_resume(url, out):
    with open(out, 'ab') as f:
        headers = {}
        pos = f.tell()
        if pos:
            headers['Range'] = f'bytes={pos}-'
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length'))
        for data in tqdm(iterable=response.iter_content(chunk_size=1024), total=total_size//1024, unit='KB'):
            f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='amazon', help='name of dataset')
    args, _ = parser.parse_known_args()

    if args.dataset.lower() == 'amazon':
        # Old URL: wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz
        base_url = 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/'
        prefix = ''
        suffix_url = '_5.json.gz'
        output_dir = os.path.join(DATA_PATH, args.dataset.lower(), 'compressed')

        for cat in AMAZON_CATEGORIES:
            filename = prefix + cat.replace(' ', '_') + suffix_url
            if not os.path.exists(os.path.join(output_dir, filename)):
                print(f'\nDownloading {cat} subset from Amazon dataset...')
                os.system(f'wget {base_url + filename} -P {output_dir} --no-check-certificate')
                # cURL alternative
                # os.system(f'curl {base_url + filename} -o {output_dir}/{filename}')
            # if not os.path.exists(os.path.join(output_dir, filename)):
            #     with gzip.open(os.path.join(output_dir, filename), 'rb') as f_in:
            #         with open(os.path.join(output_dir, cat.replace(' ', '-') + 'pkl'), 'wb') as f_out:
            #             shutil.copyfileobj(f_in, f_out)
            else:
                print(f'\n{cat} subset from Amazon dataset already exists...')

    else:
        raise NotImplementedError('The download of the provided dataset is not supported yet')
