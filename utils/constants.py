import os

BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, 'data')
CONFIG_PATH = os.path.join(BASE_PATH, 'config')
HYPEROPT_INTERMEDIATE_PATH = 'model/hyperopt'
GRAPH_DATA_FOLDER = 'graph'
TABULAR_DATA_FOLDER = 'tabular'
RNG_SEED = 42
EPS = 1e-12

MODEL_SPACE = 'model_space'
DEFAULT_MODEL_CONF = 'default_model_conf'
DATA_CONF = 'dataset'
OPT_CONF = 'optimization'
TRAIN_CONF = 'training'
EVAL_CONF = 'evaluation'

RESULTS_KEY = 'results'

PYTORCH = 'torch'
TENSORFLOW = 'tensorflow'
GRAPH_MODELS = ['FGNN', 'MemGNN', 'MAGRec']
MEMPOOL_VARIANT = 'MemGNN'
ASAP_VARIANT = 'ASAP'
SURGE_VARIANT = 'SURGE'

ITEM_COL = 'item_id'
ITEM_LIST_COL = 'hist_item_id'
ITEM_ATTR_LIST_COL = 'hist_item_attr'
USER_COL = 'user_id'
# TARGET_USER_COL = 'target_user_id'
TARGET_COL = 'click'
TIME_COL = 'timestamp'
SEQ_LEN_COL = 'item_list_len'
EDGE_WEIGHT_COL = 'edge_weight'
EDGE_ATTR_COL = 'edge_attr'
EDGE_COL = 'edge_index'
NODE_SEQ_COL = 'node_seq_ixs'
DOMAIN_COL = 'domain_id'

# 6 Domains = "Musical Instruments,Office Products,Patio Lawn and Garden,Prime Pantry,Toys and Games,Video Games"
# AMAZON_CATEGORIES = "Clothing Shoes and Jewelry".split(',')  #,Video Games,Toys and Games
AMAZON_CATEGORIES = [
    "Arts Crafts and Sewing",
    "Digital Music",
    "Gift Cards",
    "Industrial and Scientific",
    "Luxury Beauty",
    "Magazine Subscriptions",
    "Musical Instruments",
    "Office Products",
    "Patio Lawn and Garden",
    "Prime Pantry",
    "Software",
    "Toys and Games",
    "Video Games"
]

MODES = {'train': 0, 'valid': 1, 'test': 2}

DEFAULT_METRICS = ('auc', 'logloss', 'accuracy', 'pcoc')


class Colors:
    BLACK = '\033[1;30m'
    RED = '\033[1;31m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'
    PURPLE = '\033[1;35m'
    CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    ENDC = '\033[0m'
