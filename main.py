import argparse
import yaml

from model import Dict
from runner import train, eval


# parse argument
def parse_args_and_config():
    parser = argparse.ArgumentParser('3D-Generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or eval the model')
    parser.add_argument('--config', type=str, default='config/train.yml', help='path to config file')
    parser.add_argument('--device', type=str, default='cuda:3', help='use cuda or cpu')
    args = parser.parse_args()

    with open(args.config) as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))
    config.update(vars(args))

    config = recursive_update(config)
    return config


# update configuration
def recursive_update(config):

    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = Dict(recursive_update(v))
    
    return config


if __name__ == '__main__':
    args = parse_args_and_config()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        eval(args)
