import argparse
import torch


def parse_opts():
    parser = argparse.ArgumentParser()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('--device', default=device, type=str, help='using gpu if cuda is available.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--n_threads', default=0, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--drop_last', default=True, type=bool, help='drop the remaining of the batch if the size '
                                                                     'doesnt match minimum batch size')
    parser.add_argument('--data_path', default='./data/train/', type=str)

    parser.add_argument('--glr', default=1e-3, help='lr of generator_optimizer.')
    parser.add_argument('--dlr', default=1e-4, help='lr of discriminator_optimizer in phase 1.')
    parser.add_argument('--dlr2', default=5e-5, help='lr of discriminator_optimizer in phase 2.')
    parser.add_argument('--phase1_epochs', default=25, help='training epoch number')
    parser.add_argument('--phase2_epochs', default=75, help='training epoch number')
    parser.add_argument('--lamb', default=0.2, help='weight for L_G_D and L_R in phase 1')
    parser.add_argument('--alpha', default=0.1, help='weight for loss in phase 2')
    parser.add_argument('--beta', default=0.001, help='weight for loss in phase 2')
    parser.add_argument('--g_old_epoch', default=0, type=int, help='epoch number to freeze generator as g_old.')

    args = parser.parse_args()
    return args
