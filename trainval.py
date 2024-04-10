import argparse
import ast
import os

import torch
import yaml
import numpy as np
import random

from src.processor import processor

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='STAR')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    # &&&&&&&&&&&&&& LEVEL 1 Arguments: frequent Changes &&&&&&&&&&&&&&
    # path -----------------------------------------------------
    parser.add_argument('--save_base_dir', default='./output', help='Directory for saving caches and models.')
    parser.add_argument('--modelname', default='star', help='Your model name')
    parser.add_argument('--load_model_id', default=None, type=str, help="load pretrained model for test or training")
    parser.add_argument('--test_set', default='hotel', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ') 

    # Training arguments  --------------------------------------
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--scheduler_method', default="None", type=str, help="OneCycleLR/ReduceLROnPlateau/None")
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--early_stop', default=False, type=bool)
    parser.add_argument('--num_epochs', default=50, type=int)

    # &&&&&&&&&&&&&& LEVEL 2 Arguments: Rarely Changes &&&&&&&&&&&&&& 
    # Data Processing  --------------------------------------
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    # parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--dataset', default='eth5')
    # config -------------------------------------------------
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    # parser.add_argument('--model', default='star.STAR')
    # train --------------------------------------------------
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=0, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    
    parser.add_argument('--ifshow_detail', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--clip', default=1, type=int)

    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = p.save_base_dir + '/' + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + '/' + str(p.test_set) + '/' + p.modelname + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)

    args = load_arg(p)

    torch.cuda.set_device(0)

    # model_Hparameters contains 3 Hyper-parameters:
    #     - n_layers: # of layers in the temporal encoder
    #     - ratio: The number of mambas used by the spatial layer is multiple times more than that used by the temporal layer
    #     - embedding
    model_Hparameters = {
        "n_layers": 1,
        "ratio": 1,
        "embedding": 128,
        "dropout": 0
    }
    trainer = processor(args, model_parameters=model_Hparameters)

    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
