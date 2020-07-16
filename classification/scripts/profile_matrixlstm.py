import os
import torch
from termcolor import colored
import configargparse as argparse

from tqdm import tqdm
import libs.readers as reader
from libs.trainer import add_train_params
from libs.readers.transforms import add_transforms_params
from libs.arg_types import arg_boolean, arg_tuple
from models.profiler_matrixlstm import MatrixProfiler


def get_params():
    parser = argparse.ArgumentParser()
    parser = add_train_params(parser)
    parser = add_transforms_params(parser)

    parser.add_argument('--input_height', type=int)
    parser.add_argument('--input_width', type=int)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--lstm_type', type=str, default='LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=3)
    parser.add_argument('--region_shape', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--region_stride', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--add_coords_feature', type=arg_boolean, default=False)
    parser.add_argument('--add_time_feature_mode', type=str, default='delay_norm')
    parser.add_argument('--normalize_relative', type=arg_boolean, default=True)
    parser.add_argument('--keep_most_recent', type=arg_boolean, default=False)
    parser.add_argument('--frame_intervals', type=int, default=1)
    parser.add_argument('--frame_intervals_mode', type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = get_params()
    transforms = {'train': reader.get_transforms(params),
                  'test': reader.get_transforms(params, 'test_transforms'),
                  'validation': reader.get_transforms(params, 'test_transforms')}

    train_loader, _, _ = reader.get_splits(data_dir=params.data_dir,
                                           val_split=params.val_perc,
                                           batch_size=params.batch_size,
                                           num_workers=params.num_workers,
                                           transform=transforms,  pad=True,
                                           usechunks=params.use_chunks,
                                           chunks_delta_t=params.chunks_delta_t,
                                           min_chunk_delta_t=params.chunks_min_delta_t,
                                           min_chunk_n_events=params.chunks_min_n_events,
                                           seed=params.seed)

    net = MatrixProfiler((params.input_height, params.input_width),
                         params.hidden_size, params.region_shape, params.region_stride,
                         params.add_coords_feature, params.add_time_feature_mode,
                         params.normalize_relative, params.lstm_type, params.keep_most_recent,
                         params.frame_intervals, params.frame_intervals_mode,
                         params.lstm_num_layers)
    net.to(device)
    net.eval()

    # Performs one epoch of burnin
    # the second one is used for evaluation
    burnin_samples = 2000
    n_samples = 0
    print("Burnin: ")
    for batch in tqdm(train_loader):
        # Get the inputs
        batch_lengths, batch_events, _ = batch
        # Moves batch to the proper device based on GPU availability
        batch_lengths = batch_lengths.to(device)
        batch_events = batch_events.to(device).type(torch.float32)
        _ = net.forward(batch_events, batch_lengths, burned_in=False)
        n_samples += batch_lengths.shape[0]
        if n_samples >= burnin_samples:
            break

    print("Evaluation: ")
    for batch in tqdm(train_loader):
        # Get the inputs
        batch_lengths, batch_events, _ = batch
        # Moves batch to the proper device based on GPU availability
        batch_lengths = batch_lengths.to(device)
        batch_events = batch_events.to(device).type(torch.float32)
        _ = net.forward(batch_events, batch_lengths, burned_in=True)

        mean_time, kevents_s, n_samples = net.get_statistics()

    print("\n-------------------------")
    print("MatrixLSTM reconstruction timings:")
    print("Avg. Sample Time: {} (ms),  Kev/s: {}, num samples: {}"
          "".format(mean_time, kevents_s, n_samples))

    if params.output_file:
        os.makedirs(os.path.dirname(params.output_file), exist_ok=True)
        with open(params.output_file, 'a') as f:
            f.write("batch_size {}, n_intervals {}, hidden_size: {}, "
                    "mean time {}, kevents/s {}\n"
                    "".format(params.batch_size,
                              params.frame_intervals,
                              params.hidden_size,
                              mean_time, kevents_s))
