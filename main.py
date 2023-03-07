import argparse
import json

from model import Dict
from runner import train, eval


# parse argument
def parse_args_and_config():
    parser = argparse.ArgumentParser('3D-Generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or eval the model')
    parser.add_argument('--config', type=str, default='config/train.json', help='path to config file')
    parser.add_argument('--device', type=str, default='cuda:3', help='use cuda or cpu')
    args = parser.parse_args()

    with open(args.config) as file:
        config = Dict(json.load(file))
    config.update(vars(args))

    return config


if __name__ == '__main__':
    args = parse_args_and_config()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        eval(args)
