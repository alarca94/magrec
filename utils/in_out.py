import json
import pickle

import os
import torch
import yaml
import numpy as np

from utils.constants import *


def colored(text, color):
    return f'{color}{text}{Colors.ENDC}'


def load_config(config_file, prefix_path=''):
    with open(os.path.join(CONFIG_PATH, prefix_path, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config is None:
        return {}
    return config


def load_best_model(history, val_metric, model_name, dataset, domains, working_dir='.'):
    best_epoch = np.argmax(history.history[val_metric])
    results = {}
    for m, v in history.history.items():
        results[m] = v[best_epoch]
    print(f'Best valid results are at Epoch {best_epoch + 1}:\n{results2str(results)}')

    best_model_file = f'{working_dir}/checkpoints/{model_name}_{dataset}_{"_".join(domains)}_' \
                      f'{best_epoch + 1:02d}_{results[val_metric]:.2f}.hdf5'
    return torch.load(best_model_file), best_epoch


def get_config(args):
    config = load_config(f'{args.dataset}.yaml')
    config[DEFAULT_MODEL_CONF] = load_config(f'global.yaml', prefix_path='model')
    config[DEFAULT_MODEL_CONF].update(load_config(f'{args.model_name}.yaml', prefix_path='model'))
    # if args.use_ray:
    config[MODEL_SPACE] = load_config(f'global.hyper', prefix_path=HYPEROPT_INTERMEDIATE_PATH)
    ho_model_space = load_config(f'{args.model_name}.hyper', prefix_path=HYPEROPT_INTERMEDIATE_PATH)
    if ho_model_space:
        config[MODEL_SPACE].update(ho_model_space)

    config['evaluation']['val_mode'] = 'min' if 'loss' in config['evaluation']['val_metric'] else 'max'
    config['global_args'] = vars(args)

    return config


def print_exp_analysis(analysis):
    config_key = 'config'
    for tid, trial in analysis.results.items():
        if tid == analysis.best_trial.trial_id:
            print(f'{colored("Results:", Colors.BLUE)} {colored(trial[RESULTS_KEY], Colors.GREEN)}, '
                  f'{colored("Config:", Colors.BLUE)} {colored(trial[config_key], Colors.GREEN)}')
        else:
            print(f'{colored("Results:", Colors.BLUE)} {trial[RESULTS_KEY]}, '
                  f'{colored("Config:", Colors.BLUE)} {trial[config_key]}')


def write_best_config(working_dir, exp_name, best_trial):
    results_path = os.path.join(working_dir, 'results', exp_name)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    else:
        os.remove(os.path.join(results_path, '*'))

    results_file = 'best_config.pkl'
    with open(os.path.join(results_path, results_file), 'wb', encoding='utf-8') as f:
        pickle.dump(best_trial, f)


def write_results(results, working_dir, exp_name, filename, model_cfg):
    results_path = os.path.join(working_dir, 'results', exp_name)
    if not os.path.isdir(results_path):
        if not os.path.isdir(os.path.join(working_dir, 'results')):
            os.mkdir(os.path.join(working_dir, 'results'))
        os.mkdir(results_path)

    results['model_config'] = vars(model_cfg)

    with open(os.path.join(results_path, filename), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)


def results2str(results, val_metric='val_auc'):
    out = ''
    for m_name, m_val in results.items():
        if 'stats' not in m_name:
            if m_name == val_metric:
                out += colored(f'{m_name}: {m_val:.4f}', Colors.BLUE) + ' ::: '
            else:
                out += f'{m_name}: {m_val:.4f} ::: '
    return out[:-5]


def print_config(d):
    for k, v in d.items():
        print(colored(f'\t{k}: ', Colors.PURPLE) + colored(f'{v}', Colors.YELLOW))


def print_common_config(config):
    print(colored('GLOBAL CONFIGURATION:', Colors.PURPLE))
    print_config(config['global_args'])
    print(colored('DATA CONFIGURATION:', Colors.PURPLE))
    print_config(config['dataset'])
    print(colored('OPTIMIZATION CONFIGURATION:', Colors.PURPLE))
    print_config(config['optimization'])
    print_config(config['hyperoptimization'])
    print(colored('EVALUATION CONFIGURATION:', Colors.PURPLE))
    print_config(config['evaluation'])


def print_model_config(model_cfg):
    print(colored('MODEL CONFIGURATION:', Colors.PURPLE))
    print_config(vars(model_cfg))


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)
