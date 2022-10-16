import logging
import os
import torch
import sys
import numpy as np
import random

from exp.exp_GIF import ExpGraphInfluenceFunction
from exp.exp_attack import ExpAttack
from exp.exp_retrain import ExpRetraining
from parameter_parser import parameter_parser

def _set_random_seed(seed=2022):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")

def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
   

if __name__ == "__main__":
    args = parameter_parser()

    _set_random_seed(20221012)

    # config the logger
    logger_name = "_".join((args['dataset_name'], str(args['test_ratio']), args['target_model'], args['unlearn_task'], str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args["exp"].lower() == "unlearn":
        if args["method"].lower() == "retrain":
            ExpRetraining(args)
        elif args["method"].lower() in ["gif", "if"]:
            ExpGraphInfluenceFunction(args)
        else:
            raise NotImplementedError
    elif args["exp"].lower() == "attack":
        if args["method"] == "Retrain":
            ExpAttack(args)
        if args["method"].lower() in ["gif", "if"]:
            ExpAttack(args)
