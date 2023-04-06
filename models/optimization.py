import importlib
import pickle
import sys
import random
import torch
import inspect
import numpy as np
import torch.nn.functional as F

from tensorflow.python.keras.callbacks import CallbackList
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from torch import optim
from hyperopt import hp, tpe, atpe, mix, rand, anneal, pyll
from hyperopt.base import miscs_update_idxs_vals, STATUS_OK
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error

from utils.constants import *
from utils.in_out import load_best_model, results2str, colored
from utils.datasets import PyGDataset
from utils.metrics import get_gpu_usage
from utils.results import merge_results


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval


def get_hyper_opt_alg(alg):
    hyper_opt_algs = {"tpe": tpe.suggest,
                      "atpe": atpe.suggest,
                      "mix": mix.suggest,
                      "rand": rand.suggest,
                      "anneal": anneal.suggest,
                      "grid": suggest}
    return hyper_opt_algs[alg]


def create_linspace(value):
    start = value['from']
    end = value['to']
    steps = value['in']
    space = np.linspace(start, end, num=steps).astype(value['type'] if 'type' in value else 'float32').tolist()
    return space


def process_model_config(dic):
    config = {}
    valid_functions = ["choice",
                       "range",
                       "no_optimizer",
                       "randint",
                       "uniform",
                       "quniform",
                       "loguniform",
                       "qloguniform",
                       "normal",
                       "qnormal",
                       "lognormal",
                       "qlognormal"
                       ]
    space_size = []
    default_n_sample = 2
    for mod, params in dic.items():
        if isinstance(mod, str) and mod == 'choice':
            for k, v in dic[mod].items():
                config[k] = hp.choice(k, v)
                space_size.append(len(v))
        elif isinstance(mod, str) and mod == 'range':
            for k, v in dic[mod].items():
                config[k] = []
                for vi in v:
                    if isinstance(vi, dict):
                        config[k] += create_linspace(vi)
                space_size.append(len(config[k]))
                config[k] = hp.choice(k, config[k])
        elif isinstance(mod, str) and mod == 'no_optimizer':
            config[mod] = params
        elif isinstance(mod, str) and mod in valid_functions:
            for k, v in dic[mod].items():
                func_ = getattr(hp, mod)
                config[k] = func_(k, *v)
                space_size.append(default_n_sample)
    return config, max(np.prod(space_size), 1)


def get_optim(optimizer_name, model, **params):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), **params)  # lr=params.get('lr', 0.01))
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), **params)  # , lr=params.get('lr', 0.001))
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), **params)  # , lr=params.get('lr', 0.01))
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), **params)  # , lr=params.get('lr', 0.01))
    else:
        raise NotImplementedError


class HyperOptCoordinator:
    def __init__(self, config, objective_fn):
        self.config = config
        self.val_sign = 1 if config['evaluation']['val_mode'] == 'min' else -1
        self.val_metric = config['evaluation']['val_metric']
        self.objective_fn = objective_fn

    def objective(self, args):
        results, results_filename = self.objective_fn(args, config=self.config)
        if isinstance(results['best_epoch'], dict):
            loss = results['train_val']['global'][self.val_metric]
        else:
            loss = results['train_val']['global'][self.val_metric][results['best_epoch']]
        return {
            'loss': loss * self.val_sign,
            'results': results,
            'results_filename': results_filename,
            'status': STATUS_OK
        }


class Trainer:
    def __init__(self, global_cfg, model_cfg, optim_cfg, eval_cfg, dataset, metrics=DEFAULT_METRICS):
        self.model_name = global_cfg.model_name
        self.device = global_cfg.device
        self.dataset = global_cfg.dataset
        self.domains = global_cfg.domains
        self.graph_type = global_cfg.graph_type
        self.working_dir = global_cfg.working_dir
        self.verbose = global_cfg.verbose
        self.model_cfg = model_cfg
        self.n_epochs = optim_cfg.n_epochs
        self.batch_size = optim_cfg.batch_size
        self.early_stopping_patience = optim_cfg.early_stopping_patience
        self.val_metric = eval_cfg.val_metric
        self.val_mode = 'min' if 'loss' in self.val_metric else 'max'
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.loss_func = F.binary_cross_entropy
        self.metrics = self._get_metrics(metrics)

        deepctr_models = [name for name, m in sys.modules['deepctr_torch.models'].__dict__.items() if
                          isinstance(m, type)]
        if self.model_name in deepctr_models:
            self.module = 'deepctr_torch.models'
            self.unpack_func = self.unpack_batch_deepctr
            self.loss_reduction = 'sum'
            if self.model_name in ['DIN', 'DIEN']:
                self.model_specific_params = {
                    'dnn_feature_columns': dataset.feature_columns,
                    'history_feature_list': dataset.behavior_feature_list,
                    'task': 'binary'
                }
            else:
                self.model_specific_params = {
                    'linear_feature_columns': dataset.feature_columns,
                    'dnn_feature_columns': dataset.behavior_feature_list,
                    'task': 'binary'
                }
        else:
            self.module = 'models.graph_based'
            self.unpack_func = self.unpack_batch_pyg
            self.loss_reduction = 'mean'
            self.model_specific_params = {
                'n_users': dataset.n_users,
                'n_items': dataset.n_items,
                'variant': global_cfg.model_variant
            }
        self.model_specific_params['device'] = self.device
        # log_dir = f'tensorboard_log/{self.graph_type}/'
        # run_info = f'{self.model_name}_{global_cfg.start_date}'
        # self.tensorboard = SummaryWriter(log_dir=log_dir + run_info)

        if 'dummy' in self.domains[0]:
            self.batch_size = 32

    def get_callbacks(self, domains):
        if not os.path.isdir(f'{self.working_dir}/checkpoints/'):
            os.mkdir(f'{self.working_dir}/checkpoints/')

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(EarlyStopping(monitor=self.val_metric,
                                           mode=self.val_mode,
                                           patience=self.early_stopping_patience))
        callbacks.append(ModelCheckpoint(filepath=f'{self.working_dir}/checkpoints/' + f'{self.model_name}_' +
                                                  f'{self.dataset}_{"_".join(domains)}' +
                                                  '_{epoch:02d}_{' + f'{self.val_metric}' + ':.2f}.hdf5',
                                         mode=self.val_mode,
                                         monitor=self.val_metric,
                                         save_best_only=True,
                                         save_weights_only=True))
        return callbacks

    @staticmethod
    def get_optim(optimizer, lr, model):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(model.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(model.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(model.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(model.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def fit(self, train_loaders, val_loaders, n_epochs=20, verbose=2, callbacks=None, domains=None):
        model = self.model.train()
        optimizer = self.get_optim(self.model_cfg.optim, self.model_cfg.lr, model)

        # configure callbacks
        callbacks = (callbacks or []) + [model.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self.model)
        callbacks.on_train_begin()
        callbacks.set_model(self.model)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self.model)
        callbacks.model.stop_training = False

        if len(domains) == len(train_loaders):
            train_loaders = list(zip(domains, train_loaders))
        else:
            train_loaders = [('global', train_loaders[0])]
        val_loaders = list(zip(domains, val_loaders))

        results = {domain: {} for domain in ['global'] + domains}
        dhistory = {domain: {} for domain in domains}
        for epoch in range(n_epochs):
            model = model.train()
            callbacks.on_epoch_begin(epoch)
            random.shuffle(train_loaders)
            for domain, trn_loader in train_loaders:
                trn_results = self.train_epoch(trn_loader, epoch, model, optimizer, verbose)
                if domain == 'global':
                    for k, v in trn_results.items():
                        results[k].update(v)
                else:
                    results[domain].update(trn_results['global'])
            str_out = ''
            for domain, val_loader in val_loaders:
                val_results = self.eval_epoch(val_loader, epoch, verbose)
                results[domain].update(val_results['global'])
                if verbose > 0:
                    str_out += f'Domain {domain}: {results2str(results[domain])}\n'
            print(str_out[:-1])
            merge_results(results, self.metrics, prefixes=['', 'val_'])
            if hasattr(self, "epoch_optimizer"):
                self.epoch_optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            self.update_dhistory(dhistory, results)
            callbacks.on_epoch_end(epoch, results['global'])

        # self.tensorboard.flush()
        # self.tensorboard.close()

        return model.history, dhistory

    def update_dhistory(self, dhistory, results):
        for d in dhistory:
            dresults = results[d]
            for m, v in dresults.items():
                dhistory[d][m] = dhistory[d].get(m, []) + [v]

    def unpack_batch_pyg(self, b_data):
        return b_data.to(self.device), b_data.y.to(self.device).float(), b_data.target_x[:, 2].detach().cpu().numpy()

    def unpack_batch_deepctr(self, b_data):
        x = b_data[0][:, :-1].to(self.device).float()
        y = b_data[1].to(self.device).float()
        ds = b_data[0][:, -1].detach().numpy()
        return x, y, ds

    def train_epoch(self, train_loader, epoch, model, optimizer, verbose=2):
        y_preds, ys, d_ixs, total_loss_epoch = [], [], [], 0
        n_batches = len(train_loader)
        if hasattr(model, 'set_n_batches'):
            model.set_n_batches(n_batches)
        iter_data = tqdm(train_loader,
                         ncols=150,
                         desc=colored(f"Train {epoch:>5}", Colors.PURPLE),
                         disable=verbose!=1)
        for b_ix, b_data in enumerate(iter_data):
            x, y, ds = self.unpack_func(b_data)

            y_pred = model(x).squeeze()

            optimizer.zero_grad()
            loss = self.loss_func(y_pred, y.squeeze(), reduction=self.loss_reduction)
            reg_loss = model.get_regularization_loss()
            total_loss = loss + reg_loss + model.aux_loss
            total_loss_epoch += total_loss.item()
            # iter_data.set_postfix_str(
            #     colored(f'Loss: {total_loss_epoch / (b_ix + 1):.4f}', Colors.YELLOW))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            y_preds.append(y_pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            d_ixs.append(ds)

        if hasattr(model, 'reset_log_info'):
            model.reset_log_info()
        y_preds, ys, d_ixs = np.hstack(y_preds), np.hstack(ys), np.hstack(d_ixs)
        if len(np.unique(d_ixs)) == 1:
            d_ixs = None
        results = self.compute_results(y_preds, ys, d_ixs, mode='train')

        if 'global' not in results:
            results['global'] = {}
        results['global']['loss'] = total_loss_epoch / n_batches
        # self.tensorboard.add_scalar('Loss/train', results['global']['loss'], epoch)
        return results

    def predict(self, dataloader):
        model = self.model.eval()

        y_preds, ys, d_ixs, total_loss = [], [], [], 0
        with torch.no_grad():
            for b_ix, b_data in enumerate(dataloader):
                x, y, ds = self.unpack_func(b_data)
                y_pred = model(x)
                y_preds.append(y_pred.detach().cpu().numpy().squeeze())
                ys.append(y.detach().cpu().numpy().squeeze())
                d_ixs.append(ds)

        return np.hstack(y_preds), np.hstack(ys), np.hstack(d_ixs)

    def eval_epoch(self, val_loader, epoch, verbose=2):
        iter_data = tqdm(val_loader,
                         ncols=150,
                         desc=colored(f"Eval {epoch:>6}", Colors.PURPLE),
                         disable=verbose!=1)
        y_preds, ys, _ = self.predict(iter_data)
        if hasattr(self, 'reset_log_info'):
            self.reset_log_info()
        results = self.compute_results(y_preds, ys, None, mode='valid')
        return results

    def evaluate(self, test_loader, verbose=2):
        iter_data = tqdm(test_loader,
                         ncols=150,
                         desc=colored(f"Test", Colors.PURPLE),
                         disable=verbose!=1)
        y_preds, ys, _ = self.predict(iter_data)
        results = self.compute_results(y_preds, ys, None, mode='test')
        if verbose > 0:
            print(f'Test results are:\n{colored(results2str(results), Colors.GREEN)}')
        return results

    def evaluate_multiple(self, test_loaders, domains, verbose=2):
        results = {}
        test_loaders = list(zip(domains, test_loaders))
        for domain, test_loader in test_loaders:
            test_results = self.evaluate(test_loader, verbose=0)
            results[domain] = test_results['global']
            if verbose > 0:
                print(f'Test results for domain {colored(domain, Colors.GREEN)} are: {colored(results2str(results[domain]), Colors.GREEN)}')
        merge_results(results, self.metrics, prefixes=[''])
        if verbose > 0:
            print(f'Global test results are: {colored(results2str(results["global"]), Colors.GREEN)}')
        return results

    def _fit_predict(self, dataset, domains):
        callbacks = self.get_callbacks(domains)

        # Create the model
        if self.verbose == 1:
            print('Creating the model...')
        model_class = getattr(importlib.import_module(self.module), self.model_name)
        if 'deepctr' in self.module:
            model_valid_args = inspect.getfullargspec(model_class.__init__).args
            model_params = {k:v for k, v in vars(self.model_cfg).items() if k in model_valid_args}
            self.model = model_class(**self.model_specific_params, **model_params)
        else:
            self.model = model_class(domains=domains, params=self.model_cfg, **self.model_specific_params)

        if self.verbose == 1:
            print('Creating the data loaders...')
        if isinstance(dataset, PyGDataset):
            loaders = dataset.get_dataloaders(self.batch_size, domains)
        else:
            loaders = dataset.get_dataloaders(self.batch_size, self.model.feature_index, domains)

        if self.verbose == 1:
            print('Training the model...')
        history, dhistory = self.fit(loaders[0], loaders[1], n_epochs=self.n_epochs, verbose=self.verbose,
                                     callbacks=callbacks, domains=domains)

        best_model, best_epoch = load_best_model(history, self.val_metric, self.model_name, self.dataset, domains,
                                                 self.working_dir)
        self.model.load_state_dict(best_model)
        tst_results = self.evaluate_multiple(loaders[2], domains)
        return history.history, tst_results, best_epoch, dhistory

    def fit_predict(self, dataset):
        if self.graph_type in ['separate-shared', 'separate']:
            history, tst_results, best_epoch = {}, {}, {}
            for dix, domain in enumerate(self.domains):
                _, d_tst_r, d_best_e, d_h = self._fit_predict(dataset, [domain])

                # Merge separate domain results for a single report
                history.update(d_h)
                tst_results[domain] = d_tst_r[domain]
                best_epoch[domain] = d_best_e
            merge_results(history, prefixes=['', 'val_'], best_epochs=best_epoch)
            merge_results(tst_results, prefixes=[''])
        else:
            aux, tst_results, best_epoch, dhistory = self._fit_predict(dataset, self.domains)
            history = {'global': aux}
            history.update(dhistory)

        return {
            'train_val': history,
            'test': tst_results,
            'best_epoch': best_epoch
        }

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps=eps,
                        normalize=normalize,
                        sample_weight=sample_weight,
                        labels=labels)

    def _get_metrics(self, metrics, set_eps=True):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                if metric == 'pcoc':
                    metrics_[metric] = lambda y_true, y_pred: sum(y_pred) / sum(y_true)
        return metrics_

    def compute_metrics(self, y, y_pred, prefix=''):
        return {f'{prefix}{m_name}': m_func(y, y_pred) for m_name, m_func in self.metrics.items()}

    def compute_results(self, y_pred, y, domain_ixs=None, mode=None):
        prefix = ''
        if 'val' in mode:
            prefix = 'val_'

        results = {}
        if domain_ixs is not None:
            for dix, domain in enumerate(self.domains):
                results[domain] = {}
                dmask = domain_ixs == dix
                results[domain][f'{prefix}stats_count'] = dmask.sum()
                # results[domain][f'{prefix}stats']['ratio'] = results[domain][f'{prefix}stats']['count'] / len(dmask)
                if dmask.any():
                    results[domain].update(self.compute_metrics(y[dmask], y_pred[dmask], prefix))
        else:
            results['global'] = {f'{prefix}stats_count': len(y)}
            results['global'].update(self.compute_metrics(y, y_pred, prefix))

        return results