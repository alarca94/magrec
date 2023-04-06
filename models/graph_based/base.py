import numpy as np
import torch
import random
import torch.nn.functional as F

from tensorflow.python.keras.callbacks import CallbackList, History
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import coalesce
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from utils.constants import Colors, DEFAULT_METRICS
from utils.in_out import colored, results2str
from utils.results import merge_results


class GraphModelBase(nn.Module):
    def __init__(self, n_users, n_items, domains, device, metrics=DEFAULT_METRICS):
        super(GraphModelBase, self).__init__()

        self.regularization_weight = []
        self.auxiliary_loss = []
        self.n_users = n_users
        self.n_items = n_items
        self.domains = sorted(domains)
        self.device = device
        self.metrics = metrics
        self.aux_loss = 0

        self.log_dir = 'tensorboard_log/graph_based/'
        self.history = History()
        self.grad_history = []

    def finish_init(self, optim, device):
        self.loss_func = F.binary_cross_entropy
        self.optim = self._get_optim(optim, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.1)

        # self.tensorboard = SummaryWriter(log_dir=self.log_dir + self.run_info)

        self.to(device)
        print(self)

    def set_n_batches(self, n_batches):
        self.n_batches = n_batches

    @staticmethod
    def to_undirected(edge_index, edge_attr, num_nodes, reduce: str = "max"):
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        if edge_attr is not None:
            back_edge_attr = torch.index_select(edge_attr, 1,
                                                torch.tensor([1,0,*range(2, edge_attr.shape[1])], device=edge_attr.device))
            edge_attr = torch.cat([edge_attr, back_edge_attr], dim=0)

        return coalesce(edge_index, edge_attr, num_nodes, reduce)

    def fit(self, train_loaders, val_loaders, n_epochs=20, verbose=2, callbacks=None):
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        if len(self.domains) == len(train_loaders):
            train_loaders = list(zip(self.domains, train_loaders))
        else:
            train_loaders = [('global', train_loaders[0])]
        val_loaders = list(zip(self.domains, val_loaders))

        results = {domain: {} for domain in ['global'] + self.domains}
        dhistory = {domain: {} for domain in self.domains}
        for epoch in range(n_epochs):
            model = model.train()
            callbacks.on_epoch_begin(epoch)
            random.shuffle(train_loaders)
            for domain, trn_loader in train_loaders:
                trn_results = self.train_epoch(trn_loader, epoch, model, loss_func, optim, verbose)
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
            if hasattr(self, "epoch_optim"):
                self.epoch_optim.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            self.update_dhistory(dhistory, results)
            callbacks.on_epoch_end(epoch, results['global'])

        # self.tensorboard.flush()
        # self.tensorboard.close()

        return self.history, dhistory

    def update_dhistory(self, dhistory, results):
        for d in dhistory:
            dresults = results[d]
            for m, v in dresults.items():
                dhistory[d][m] = dhistory[d].get(m, []) + [v]

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

    def evaluate_multiple(self, test_loaders, verbose=2):
        results = {}
        test_loaders = list(zip(self.domains, test_loaders))
        for domain, test_loader in test_loaders:
            test_results = self.evaluate(test_loader, verbose=0)
            results[domain] = test_results['global']
            if verbose > 0:
                print(f'Test results for domain {colored(domain, Colors.GREEN)} are: {colored(results2str(results[domain]), Colors.GREEN)}')
        merge_results(results, self.metrics, prefixes=[''])
        if verbose > 0:
            print(f'Global test results are: {colored(results2str(results["global"]), Colors.GREEN)}')
        return results

    def train_epoch(self, train_loader, epoch, model, loss_func, optim, verbose=2):
        y_preds, ys, d_ixs, total_loss_epoch = [], [], [], 0
        n_batches = len(train_loader)
        model.set_n_batches(n_batches)
        iter_data = tqdm(train_loader,
                         ncols=150,
                         desc=colored(f"Train {epoch:>5}", Colors.PURPLE),
                         disable=verbose!=1)
        max_y = 0
        for b_ix, b_data in enumerate(iter_data):
            b_data = b_data.to(self.device)
            y = b_data.y.float()

            y_pred = model(b_data).squeeze()

            optim.zero_grad()
            loss = loss_func(y_pred, y.squeeze(), reduction='mean')
            reg_loss = self.get_regularization_loss()
            total_loss = loss + reg_loss
            total_loss_epoch += total_loss.item()
            max_y = max(max_y, y_pred.max().item())
            if not hasattr(self, 'norms'):
                iter_data.set_postfix_str(
                    colored(f'Loss: {total_loss_epoch / (b_ix + 1):.4f}', Colors.YELLOW))
            else:
                norm_str = ', '.join([f'{n:.2f}' for n in model.norms.detach().cpu().numpy().tolist()])
                iter_data.set_postfix_str(colored(f'Loss: {total_loss_epoch / (b_ix + 1):.4f}, Norms: {norm_str}',
                                                  Colors.YELLOW))
            # iter_data.set_postfix_str(colored(f'Loss: {total_loss_epoch/(b_ix + 1):.4f}', Colors.YELLOW))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            # self.autoclip_gradient()
            optim.step()

            y_preds.append(y_pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            d_ixs.append(b_data.target_x[:, 2].detach().cpu().numpy())

        # model.print_cka(reset=False)
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
        # self.tensorboard.add_scalar('CKA/train', cka, epoch)
        return results

    def predict(self, dataloader):
        model = self.eval()

        y_preds, ys, d_ixs, total_loss = [], [], [], 0
        with torch.no_grad():
            for b_data in dataloader:
                b_data = b_data.to(self.device)
                y = b_data.y.float()
                y_pred = model(b_data)
                y_preds.append(y_pred.detach().cpu().numpy().squeeze())
                ys.append(y.detach().cpu().numpy().squeeze())
                d_ixs.append(b_data.target_x[:, 2].detach().cpu().numpy().squeeze())

        return np.hstack(y_preds), np.hstack(ys), np.hstack(d_ixs)

    def eval_epoch(self, val_loader, epoch, verbose=2):
        iter_data = tqdm(val_loader,
                         ncols=150,
                         desc=colored(f"Eval {epoch:>6}", Colors.PURPLE),
                         disable=verbose!=1)
        y_preds, ys, _ = self.predict(iter_data)
        loss = self.loss_func(torch.tensor(y_preds), torch.tensor(ys))
        # self.print_cka(reset=False)
        if hasattr(self, 'reset_log_info'):
            self.reset_log_info()
        # self.tensorboard.add_scalar('Loss/valid', loss, epoch)
        # self.tensorboard.add_scalar('CKA/valid', cka, epoch)
        results = self.compute_results(y_preds, ys, None, mode='valid')
        # for m_name, m_val in results['global'].items():
        #     if 'stats' not in m_name:
        #         self.tensorboard.add_scalar(f'{m_name.upper()}/valid', m_val, epoch)
        return results

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def add_auxiliary_loss(self, loss):
        self.auxiliary_loss.append(loss)

    def get_auxiliary_loss(self):
        return torch.sum(self.auxiliary_loss)

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def autoclip_gradient(self):
        obs_grad_norm = self._get_grad_norm()
        self.grad_history.append(obs_grad_norm)
        clip_value = np.percentile(self.grad_history[-1000:], 10)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # clip_value)

    def compute_results(self, y_pred, y, domain_ixs=None, mode=None):
        def compute_metrics(y, y_pred):
            metrics = {}
            if 'auc' in self.metrics:
                metrics[f'{prefix}auc'] = roc_auc_score(y, y_pred)
            if 'logloss' in self.metrics:
                metrics[f'{prefix}logloss'] = log_loss(y, y_pred.astype(np.double))
            if 'accuracy' in self.metrics:
                metrics[f'{prefix}accuracy'] = accuracy_score(y, np.where(y_pred > 0.5, 1, 0))
            return metrics

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
                    results[domain].update(compute_metrics(y[dmask], y_pred[dmask]))
        else:
            results['global'] = {f'{prefix}stats_count': len(y)}
            results['global'].update(compute_metrics(y, y_pred))

        return results

    def _get_optim(self, optimizer, lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim