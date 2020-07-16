import torch
import configargparse as argparse
import numpy as np

from collections import OrderedDict
import libs.readers as reader
from libs.trainer import Trainer, add_train_params, seed_iterator
from libs.readers.transforms import add_transforms_params
from libs.arg_types import arg_boolean, arg_tuple, arg_kwargs
from models.net_matrixlstm_resnet import MatrixLSTMResNet


def get_params():
    parser = argparse.ArgumentParser()
    parser = add_train_params(parser)
    parser = add_transforms_params(parser)

    parser.add_argument('--input_height', type=int)
    parser.add_argument('--input_width', type=int)
    parser.add_argument('--lstm_type', type=str, default='LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=-1)
    parser.add_argument('--hidden_size', type=int, default=3)
    parser.add_argument('--region_shape', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--region_stride', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--add_coords_feature', type=arg_boolean, default=False)
    parser.add_argument('--add_time_feature_mode', type=str, default='delay_norm')
    parser.add_argument('--normalize_relative', type=arg_boolean, default=True)
    parser.add_argument('--keep_most_recent', type=arg_boolean, default=False)
    parser.add_argument('--frame_intervals', type=int, default=1)
    parser.add_argument('--frame_intervals_mode', type=str, default=None)

    parser.add_argument('--eventdrop', type=float, default=-1)
    parser.add_argument('--framedrop', type=float, default=-1)
    parser.add_argument('--fcdrop', type=float, default=-1)
    parser.add_argument('--frame_actfn', type=str, default=None)

    parser.add_argument('--resnet_type', type=str, default='resnet18')
    parser.add_argument('--resnet_pretrain', type=arg_boolean, default=False)
    parser.add_argument('--resnet_freeze', type=str, nargs='+', default=[])
    parser.add_argument('--resnet_meanstd_norm', type=arg_boolean, default=False)
    parser.add_argument('--resnet_add_last_fc', type=arg_boolean, default=False)
    parser.add_argument('--resnet_replace_first', type=arg_boolean, default=False)
    parser.add_argument('--resnet_replace_first_bn', type=arg_boolean, default=True)
    parser.add_argument('--add_se_layer', type=arg_boolean, default=False)

    parser.add_argument('--decay_scheduler', type=str, default=None)
    parser.add_argument('--decay_kwargs', type=arg_kwargs(), default=None)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":

    params = get_params()
    if params.decay_scheduler:
        scheduler_cls = getattr(torch.optim.lr_scheduler, params.decay_scheduler)
        lr_scheduler = lambda optim: scheduler_cls(optim, **params.decay_kwargs)
    else:
        lr_scheduler = None

    transforms = {'train': reader.get_transforms(params),
                  'test': reader.get_transforms(params, 'test_transforms'),
                  'validation': reader.get_transforms(params, 'test_transforms')}

    acc_results = []
    for params in seed_iterator(params):

        train_loader, val_loader, test_loader = reader.get_splits(data_dir=params.data_dir,
                                                                  val_split=params.val_perc,
                                                                  batch_size=params.batch_size,
                                                                  num_workers=params.num_workers,
                                                                  transform=transforms,  pad=True,
                                                                  usechunks=params.use_chunks,
                                                                  chunks_delta_t=params.chunks_delta_t,
                                                                  min_chunk_delta_t=params.chunks_min_delta_t,
                                                                  min_chunk_n_events=params.chunks_min_n_events,
                                                                  seed=params.seed)

        net = MatrixLSTMResNet((params.input_height, params.input_width),
                               train_loader.dataset.num_classes,
                               params.embedding_size, params.hidden_size,
                               params.region_shape, params.region_stride,
                               params.add_coords_feature, params.add_time_feature_mode,
                               params.normalize_relative,
                               params.lstm_type, params.keep_most_recent,
                               params.frame_intervals, params.frame_intervals_mode,
                               params.resnet_type,
                               params.resnet_pretrain, params.resnet_freeze,
                               params.eventdrop, params.framedrop, params.fcdrop,
                               params.frame_actfn, params.resnet_meanstd_norm,
                               params.resnet_add_last_fc, params.lstm_num_layers,
                               params.resnet_replace_first, params.resnet_replace_first_bn,
                               params.add_se_layer)

        trainer = Trainer(net, torch.optim.Adam, train_loader, val_loader, test_loader, params,
                          lr_scheduler=lr_scheduler)
        acc = trainer.train_network()
        acc_results.append(acc)

    print("\n-------------------------")
    print("Multiple seed evaluation:")
    print("Results:", acc_results)
    print("Aggregate result: {} +/- {}".format(np.mean(acc_results), np.std(acc_results)))
