import json
import re
from datetime import datetime

import os
import pandas as pd
import numpy as np

# from ray.tune import ExperimentAnalysis

from utils.constants import DEFAULT_METRICS


def merge_results(results, metrics=DEFAULT_METRICS, prefixes=('', 'val_', 'tst_'), best_epochs=None):
    if 'global' not in results:
        results['global'] = {}
    for prefix in prefixes:
        for metric in metrics:
            curr_metric = f'{prefix}{metric}'
            ind_results = []
            d_sizes = []
            for domain in results:
                if domain != 'global':
                    if best_epochs is not None and isinstance(best_epochs, dict):
                        ind_results.append(results[domain][curr_metric][best_epochs[domain]])
                        d_sizes.append(results[domain][f'{prefix}stats_count'][best_epochs[domain]])
                    else:
                        ind_results.append(results[domain][curr_metric])
                        d_sizes.append(results[domain][f'{prefix}stats_count'])
            results['global'][curr_metric + '_weighted'] = np.multiply(ind_results, d_sizes).sum() / sum(d_sizes)
            results['global'][curr_metric] = np.mean(ind_results)


# def view_ray_results(exp_name):
#     analysis = ExperimentAnalysis(f"~/ray_results/{exp_name}", default_metric='val_auc', default_mode='max')
#     print_exp_analysis(analysis)


def merge_trn_val_tst(res, domains):
    res_out = {}
    for d in domains:
        dresults = {'tst_' + k: v for k, v in res['test'][d].items()}
        dresults.update(res['train_val'][d])
        res_out[d] = dresults

    return res_out


def view_ho_results(model_name=None, dataset=None, domains=None, graph_type=None, how='all', verbose=1,
                    exp_path=None, results_path='./results/', flags=(), from_date=datetime.fromtimestamp(0),
                    to_date=datetime.utcnow()):
    if exp_path is None:
        exp_name = f'{model_name}_{dataset}_{"_".join(domains)}_{graph_type}'
        exp_path = os.path.join(results_path, exp_name)

    if os.path.isdir(exp_path):
        trials = process_path(exp_path, domains, from_date, to_date)
        if trials:
            trials = pd.DataFrame.from_dict(trials).T.sort_values(['val_auc', 'val_logloss'], ascending=[False, True])
            cols = trials.columns
            if 'only_global' in flags:
                cols = [c for c in cols if c.split('_')[-1] not in domains]
            elif 'only_domains' in flags:
                cols = [c for c in cols if c.split('_')[-1] in domains]
            trials = trials[cols]
            trials = filter_trials(trials, how, verbose)

            model_config = get_config(exp_path, trials.iloc[0].name, verbose)

            return trials, model_config

    return pd.DataFrame(), {}


def filter_trials(trials, how, verbose=1):
    f_trials = None
    if how == 'first':
        f_trials = trials.head(1)
    elif how == 'last':
        f_trials = trials.tail(1)
    elif how == 'all':
        f_trials = trials  #.reset_index().rename(columns={'index': 'config'})
    elif isinstance(how, int):
        f_trials = trials.head(how)

    if verbose == 1:
        print(f_trials)
    # print(f'\nBest config: {trials.iloc[0].name}')
    return f_trials


def get_config(exp_path, filename, verbose=1):
    if 'MAMDR' not in filename:
        with open(os.path.join(exp_path, filename), 'r') as fp:
            res = json.load(fp)
        if verbose == 1:
            print(json.dumps(res['model_config'], indent=4))

        return res['model_config']
    else:
        return {}


def process_file(f, res, trials, domains):
    best_epoch = res['best_epoch']
    if best_epoch is None:
        best_epoch = np.argmax(res['train_val']['global']['val_auc'])
    if isinstance(best_epoch, dict):
        trials[f] = {m.replace("weighted", "w"): v for m, v in res['train_val']['global'].items()}
        trials[f].update({f'tst_{m.replace("weighted", "w")}': v for m, v in res['test']['global'].items()})
        for d in domains:
            trials[f].update({f'{m.replace("weighted", "w")}_{d}': v[best_epoch[d]]
                              for m, v in res['train_val'][d].items()
                              if 'val_' in m and 'stats' not in m})
            trials[f].update({f'tst_{m.replace("weighted", "w")}_{d}': v for m, v in res['test'][d].items()
                              if 'stats' not in m})
    else:
        trials[f] = {m.replace("weighted", "w"): v[best_epoch] for m, v in res['train_val']['global'].items() if
                     'stats' not in m}
        trials[f].update(
            {f'tst_{m.replace("weighted", "w")}': v for m, v in res['test']['global'].items() if 'stats' not in m})
        for d in domains:
            trials[f].update({f'{m.replace("weighted", "w")}_{d}': v[best_epoch]
                              for m, v in res['train_val'][d].items()
                              if 'val_' in m and 'stats' not in m and v})
            trials[f].update({f'tst_{m.replace("weighted", "w")}_{d}': v for m, v in res['test'][d].items()
                              if 'stats' not in m and v})
    return trials


def process_path(exp_path, domains, from_date=datetime.fromtimestamp(0), to_date=datetime.utcnow()):
    trials = {}
    for f in os.listdir(exp_path):
        if f != 'best_config.json':
            date = re.findall('\d{4}-\d{2}-\d{2}T\d{2}[:_]\d{2}[:_]\d{2}Z', f)[0].replace('_', ':')
            date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
            if from_date <= date and to_date >= date:
                with open(os.path.join(exp_path, f), 'r') as fp:
                    res = json.load(fp)
                trials = process_file(f, res, trials, domains)
    return trials


def get_avg(dataset, domains, models, graph_types, topk=5, group_by=('model_name', 'graph_type'),
            sort_by='tst_auc', from_date=datetime.fromtimestamp(0), to_date=datetime.utcnow(),
            results_path='./results/', flags=()):
    best_results = []
    model_configs = {}
    for model_name in models:
        for graph_type in graph_types:
            if model_name == 'MAMDR' and graph_type == 'disjoint':
                print(f'{model_name} -- {graph_type}')
            best_res, model_config = view_ho_results(model_name, dataset, domains, graph_type, how='all', verbose=0,
                                                     results_path=results_path, flags=flags, from_date=from_date,
                                                     to_date=to_date)
            if best_res.shape[0] > 0:
                # best_res = pd.DataFrame(best_res.mean()).T
                best_res['graph_type'] = graph_type
                best_res['model_name'] = model_name
                best_results.append(best_res)
                model_configs[f'{model_name}_{graph_type}'] = model_config

    best_results = pd.concat(best_results).sort_values('val_auc', ascending=False)
    drop_cols = [c for c in best_results.columns if 'accuracy' in c]
    if 'only_test' in flags:
        drop_cols += [c for c in best_results.columns if 'tst_' not in c and c not in drop_cols]
    if 'loss' in best_results.columns:
        drop_cols += ['loss']
    best_results.drop(drop_cols, axis=1, inplace=True)
    temp_df = best_results.groupby(list(group_by)).size()
    # print(temp_df[temp_df < topk])
    if isinstance(topk, int):
        best_results = best_results.groupby(list(group_by)).head(topk)
    best_results = best_results.groupby(list(group_by)).agg(['mean', 'std']).reset_index()
    # best_results.astype(str).groupby(level=0, axis=1).transform(lambda x: ' \u00B1 '.join(x))
    # best_results.columns = best_results.columns.map("_".join)
    if isinstance(sort_by, str):
        return best_results.sort_values([(sort_by, 'mean')], ascending=False), model_configs
    elif isinstance(sort_by, list):
        return best_results.sort_values([(c, 'mean') for c in sort_by], ascending=False), model_configs


def get_best(dataset, domains, models, graph_types, group_by=('model_name', 'graph_type'), sort_by='tst_auc',
             results_path='./results/', flags=()):
    best_results = []
    model_configs = {}
    for model_name in models:
        for graph_type in graph_types:
            if model_name == 'MAMDR' and graph_type == 'disjoint':
                print(f'{model_name} -- {graph_type}')
            best_res, model_config = view_ho_results(model_name, dataset, domains, graph_type, how='all', verbose=0,
                                                     results_path=results_path, flags=flags)
            if best_res.shape[0] > 0:
                best_res['graph_type'] = graph_type
                best_res['model_name'] = model_name
                best_results.append(best_res)
                model_configs[f'{model_name}_{graph_type}'] = model_config

    best_results = pd.concat(best_results).sort_values('val_auc', ascending=False)
    drop_cols = [c for c in best_results.columns if 'accuracy' in c]
    if 'only_test' in flags:
        drop_cols += [c for c in best_results.columns if 'tst_' not in c and c not in drop_cols]
    if 'loss' in best_results.columns:
        drop_cols += ['loss']
    best_results.drop(drop_cols, axis=1, inplace=True)
    best_results.drop_duplicates(subset=group_by, keep='first', inplace=True)
    return best_results.sort_values(sort_by, ascending=False), model_configs
