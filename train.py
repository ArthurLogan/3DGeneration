import argparse

# parse argument
def parse():
    parser = argparse.ArgumentParser('3D-Generation')

    # model
    parser.add_argument('-num_features', type=int, default=512, help='number of latent features')
    parser.add_argument('-num_channels', type=int, default=512, help='number of channels')
    parser.add_argument('-num_layers', type=int, default=16, help='number of layers in decoder')
    parser.add_argument('-reg', type=bool, default=True, help='if use kl-regularization block')

    # training
    parser.add_argument('-learning_rate', type=float, default=5e-5, help='learning rate')


    args = parser.parse_args()
    return args





