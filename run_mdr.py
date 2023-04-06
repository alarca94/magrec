import json
import signal
import sys
# import ray
import torch
import argparse
import pandas as pd
import numpy as np

from datetime import datetime
from types import SimpleNamespace
# from deepctr_torch.callbacks import ModelCheckpoint, EarlyStopping
# from ray import tune
# from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import fmin, tpe, Trials

from models.optimization import process_model_config, HyperOptCoordinator, Trainer
from utils.aux_funcs import init_seed, check_model_config, domain_map
from utils.constants import *
from utils.datasets import PyGDataset, DeepCTRDataset, create_dataset
from utils.in_out import print_exp_analysis, get_config, write_results, write_best_config, print_model_config,\
    print_common_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', type=lambda s: sorted(s.split(',')), default='Patio Lawn and Garden,Office Products')  # default='Video Games,Toys and Games')
    parser.add_argument('--graph-type', type=str, default='flattened', help="Possible values: flattened, disjoint, interacting, separate-shared")
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--model-name', type=str, default='MAGRec')
    parser.add_argument('--model-variant', type=str, default='', help='Only used for MAGRec: MemGNN, ASAP, SURGE')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--use-ray', dest='use_ray', action='store_true')
    parser.add_argument('--no-use-hyperopt', dest='use_hyperopt', action='store_false')  # Ignored if use_ray == True
    parser.add_argument('--force-cpu', dest='force_cpu', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-save-results', dest='save_results', action='store_false')
    parser.set_defaults(use_ray=False, use_hyperopt=True, force_cpu=False, test=False, save_results=True)
    args, _ = parser.parse_known_args()
    if 'dummy' not in args.domains[0]:
        args.domains = [domain_map(domain) for domain in args.domains]

    config = get_config(args)
    graph_type = config["global_args"]["graph_type"]
    if args.model_name in ['MAGRec']:
        exp_name = f'{args.model_name}_{args.model_variant}_{args.dataset}_{"_".join(args.domains)}_{graph_type}'
    else:
        exp_name = f'{args.model_name}_{args.dataset}_{"_".join(args.domains)}_{graph_type}'
    config['global_args']['exp_name'] = exp_name

    config['global_args']['device'] = 'cpu'
    if torch.cuda.is_available() and not args.force_cpu:
        config['global_args']['device'] = 'cuda'

    if args.test:
        config['hyperoptimization']['ho_max_evals'] = 2
        config['optimization']['n_epochs'] = 2
        config['global_args']['save_results'] = False

    print_common_config(config)
    model_space, space_size = process_model_config(config.get(MODEL_SPACE, {}))

    run_config(config, args, model_space, space_size, exp_name)


def run_config(config, args, model_space, space_size, exp_name):
    if args.use_ray:
        raise NotImplementedError('Ray version does not currently work due to Signal Termination 15')
        # ray.init(num_gpus=2)
        #
        # resources_per_trial = {'gpu': config['hyperoptimization'].get('gpu_per_trial', 0),
        #                        'cpu': config['hyperoptimization']['cpu_per_trial']}
        # if resources_per_trial['gpu'] == 0 and config['global_args']['device'] == 'cuda':
        #     config['global_args']['device'] = 'cpu'
        #     raise Warning('GPU resources per trial are set to 0 even though there are GPUs available!')
        #
        # hyperopt_search = HyperOptSearch(space=model_space,
        #                                  n_initial_points=1,
        #                                  metric=config['evaluation']['val_metric'],
        #                                  mode=config['evaluation']['val_mode'],
        #                                  random_state_seed=RNG_SEED)
        #
        # analysis = tune.run(
        #     tune.with_parameters(objective_fn, config=config),
        #     local_dir='./ray_results',
        #     name=exp_name,
        #     search_alg=hyperopt_search,
        #     num_samples=min(space_size, config['hyperoptimization']['ho_max_evals']),
        #     metric=config['evaluation']['val_metric'],
        #     mode=config['evaluation']['val_mode'],
        #     verbose=0,
        #     resources_per_trial=resources_per_trial,
        #     max_failures=0)
        #
        # print_exp_analysis(analysis)

    elif args.use_hyperopt:
        ho_coordinator = HyperOptCoordinator(config, objective_fn)
        trials = Trials()
        best_trial = fmin(ho_coordinator.objective,
                          model_space,
                          algo=tpe.suggest,
                          max_evals=min(space_size, config['hyperoptimization']['ho_max_evals']),
                          trials=trials,
                          rstate=np.random.RandomState(RNG_SEED))

        # if config['global_args']['save_results']:
        #     write_best_config('./', exp_name, {'trial_params': best_trial, 'fixed_params': config[DEFAULT_MODEL_CONF],
        #                                        })
    else:
        config['global_args']['verbose'] = 1
        objective_fn(None, config=config)

    print('Run config finished!')


def objective_fn(args, **params):
    init_seed()
    working_dir = os.getcwd()

    config = params['config']
    model_config = config[DEFAULT_MODEL_CONF].copy()
    if args is not None:
        model_config.update(args)

    global_cfg = SimpleNamespace(**config['global_args'])
    data_cfg = SimpleNamespace(**config['dataset'])
    optim_cfg = SimpleNamespace(**config['optimization'])
    eval_cfg = SimpleNamespace(**config['evaluation'])
    model_cfg = SimpleNamespace(**model_config)

    global_cfg.working_dir = working_dir
    global_cfg.start_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    check_model_config(global_cfg.graph_type, model_cfg, global_cfg.model_name)
    print_model_config(model_cfg)

    results = mdr_pipeline(global_cfg, data_cfg, model_cfg, optim_cfg, eval_cfg)

    file_name = f'{global_cfg.model_name}_{global_cfg.start_date}.json'
    if global_cfg.save_results:
        write_results(results, working_dir, global_cfg.exp_name, file_name, model_cfg)
    return results, file_name


def mdr_pipeline(global_cfg, data_cfg, model_cfg, optim_cfg, eval_cfg):
    # Create graph dataset
    deepctr_models = [name for name, m in sys.modules['deepctr_torch.models'].__dict__.items() if isinstance(m, type)]
    if global_cfg.model_name in deepctr_models:
        dataset_class = DeepCTRDataset
    elif global_cfg.model_name in GRAPH_MODELS:
        dataset_class = PyGDataset

    dataset = create_dataset(domains=global_cfg.domains, dataset=global_cfg.dataset, min_wsize=data_cfg.min_win_size,
                             max_wsize=data_cfg.max_win_size, ctr_ratio_range=data_cfg.ctr_ratio_range,
                             split_type=data_cfg.split_type, rebuild=data_cfg.rebuild, kcore=data_cfg.kcore,
                             graph_type=global_cfg.graph_type, verbose=global_cfg.verbose,
                             dataset_class=dataset_class, batch_size=optim_cfg.batch_size, use_seq=data_cfg.use_seq)

    # ASSERT number of edges is the same in edge_attr and edge_index
    # for i in tqdm(range(len(dataset))):
    #     g = dataset.get(i)
    #     assert g.edge_attr.shape[0] == g.edge_index.shape[1]

    if global_cfg.verbose == 1:
        dataset.print_split_info()

    trainer = Trainer(global_cfg, model_cfg, optim_cfg, eval_cfg, dataset)
    results = trainer.fit_predict(dataset)

    print('These are the results')
    # Uncomment for pretty print of the results
    # print(json.dumps(results, indent=4))
    print(results)

    return results


def handler(signum, frame):
    # print('Signal handler called with signal', signum)
    # print('frame:', frame)
    pass

catchable_sigs = [signal.SIGWINCH, signal.SIGTERM]
for sig in catchable_sigs:
    signal.signal(sig, handler)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None

    main()
