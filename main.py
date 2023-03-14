import argparse
import yaml

from runner import train
# from parallel import train_ddp


# parse argument
def parse_args_and_config():
    parser = argparse.ArgumentParser('3D-Generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'ddp'], help='train or eval the model')
    parser.add_argument('--config', type=str, default='config/train.yml', help='path to config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='use cuda or cpu')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config.update(vars(args))

    return config


if __name__ == '__main__':
    args = parse_args_and_config()
    if args['mode'] == 'train':
        train(args)
    # elif args.mode == 'test':
    #     eval(args)
    # elif args.mode == 'ddp':
    #     train_ddp(args)
