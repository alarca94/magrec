import numpy as np
import tensorflow as tf
import random
import torch

from utils.constants import RNG_SEED


def init_seed(seed=RNG_SEED, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)  # tf.set_random_seed(seed) for tf1.12 (MAMDR)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def domain_map(domain):
    return ''.join([d[0] for d in domain.split()])


def check_model_config(graph_type, model_cfg, model_name):
    if model_name.lower() == 'magrec':
        if graph_type == 'separate-shared':
            model_cfg.use_domain_info = False
            model_cfg.normalize_domains = False
