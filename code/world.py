"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""

import os
import torch
from parse import parse_args
import time
import logging
import sys
from warnings import simplefilter


simplefilter(action="ignore", category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def get_logger(dataset_name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s")

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = f"{dataset_name}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


def set_config():
    config = {}
    config["bpr_batch_size"] = args.bpr_batch
    config["latent_dim_rec"] = args.recdim
    config["n_layers"] = args.layer
    config["dropout"] = args.dropout
    config["keep_prob"] = args.keepprob
    config["test_u_batch_size"] = args.testbatch
    config["multicore"] = args.multicore
    config["lr"] = args.lr
    config["decay"] = args.decay
    config["comment"] = args.comment
    config["early_stop"] = args.early_stop
    config["eval_step"] = args.eval_step
    config["groups"] = args.groups
    config["subg"] = args.subg
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["dataset"] = args.dataset
    config["model"] = args.model
    config["epochs"] = args.epochs
    config["load"] = args.load
    config["path"] = args.path
    config["topks"] = args.topks
    config["tensorboard"] = args.tensorboard
    config["comment"] = args.comment
    config["neg_k"] = args.neg_k
    config["train_h"] = args.train_h
    config["test_h"] = args.test_h
    config["analysis"] = args.analysis

    all_dataset = ["ml-1m"]
    if args.dataset not in all_dataset:
        raise NotImplementedError(f"Haven't supported {args.dataset} yet!, try {all_dataset}")
    all_models = ["lgn", "localgcn", "anchorrec"]
    if args.model not in all_models:
        raise NotImplementedError(f"Haven't supported {args.model} yet!, try {all_models}")
    return config

args = parse_args()
config = set_config()

LOGO = r"""
   _       U  ___ u   ____     _       _       ____     ____  _   _     
  |"|       \/"_ \/U /"___|U  /"\  u  |"|   U /"___|uU /"___|| \ |"|    
U | | u     | | | |\| | u   \/ _ \/ U | | u \| |  _ /\| | u <|  \| |>   
 \| |/__.-,_| |_| | | |/__  / ___ \  \| |/__ | |_| |  | |/__U| |\  |u   
  |_____|\_)-\___/   \____|/_/   \_\  |_____| \____|   \____||_| \_|    
  //  \\      \\    _// \\  \\    >>  //  \\  _)(|_   _// \\ ||   \\,-. 
 (_")("_)    (__)  (__)(__)(__)  (__)(_")("_)(__)__) (__)(__)(_")  (_/  
"""
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = os.path.join(ROOT_PATH, "code")
DATA_PATH = os.path.join(ROOT_PATH, "data")
BOARD_PATH = os.path.join(CODE_PATH, "runs")
EMBED_PATH = os.path.join(ROOT_PATH, "embed") # embed
GRAPH_PATH = os.path.join(ROOT_PATH, "graph")
if args.load:
    FOLDER_PATH = os.path.join(BOARD_PATH, os.path.join(args.dataset, os.path.join(args.model, args.load)))
else:
    FOLDER_PATH = (
        os.path.join(BOARD_PATH, os.path.join(args.dataset, os.path.join(args.model, time.strftime("%m-%d-%Hh%Mm%Ss"))))
        + f"-l{args.layer}-d{args.recdim}-g{args.groups}-{args.comment}"
    )
PRETRAINED_EMB = os.path.join(EMBED_PATH, f"lgn-{args.dataset}.pt")
WMAT = os.path.join(GRAPH_PATH, f"wmat-{args.dataset}-{args.train_h}-{args.test_h}-{args.subg}.npz")
GRAPH = os.path.join(GRAPH_PATH, f"graph-{args.dataset}-g{args.groups}-{args.train_h}-{args.test_h}-{args.subg}.pt")
LOGGER = get_logger(args.dataset, FOLDER_PATH)
ANALYSIS_PATH = os.path.join(
    os.path.join(ROOT_PATH, os.path.join("analysis",
    f"{args.dataset}-{args.model}-l{args.layer}-d{args.recdim}-g{args.groups}-{args.comment}.parquet")
    ) 
)

sys.path.append(os.path.join(CODE_PATH, "sources"))
mkdir_if_not_exist(FOLDER_PATH)
mkdir_if_not_exist(GRAPH_PATH)
