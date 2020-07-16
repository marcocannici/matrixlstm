#!/usr/bin/env python
import re
import configargparse as argparse
from collections import OrderedDict


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_dict(cast_type, key_type=str):
    regex_pairs = re.compile(r'[^\ ]+:[^\ ]+')
    regex_keyvals = re.compile(r'([^\ ]+):([^\ ]+)')

    def parse_dict(v):
        d = OrderedDict()
        for keyval in regex_pairs.findall(v):
            key, val = regex_keyvals.match(keyval).groups()
            d.update({key_type(key): cast_type(val)})
        return d

    return parse_dict


def arg_tuple(cast_type):
    regex = re.compile(r'\d+\.\d+|\d+')

    def parse_tuple(v):
        vals = regex.findall(v)
        return [cast_type(val) for val in vals]

    return parse_tuple


def configs():
    parser = argparse.ArgumentParser()
    parser.add('-c', is_config_file=True, help='optional config file path')

    parser.add_argument('--use_paper_args', type=arg_boolean, default=False)

    parser.add_argument('--sacred', type=arg_boolean, default=False)
    parser.add_argument('--mongodb_disable', type=arg_boolean, default=False)
    parser.add_argument('--mongodb_url', type=str, default='127.0.0.1')
    parser.add_argument('--mongodb_port', type=str, default='27017')
    parser.add_argument('--mongodb_name', type=str)
    parser.add_argument('--exp_name', type=str, default="")

    parser.add_argument('--gzip',
                        type=arg_boolean, default=False,
                        help='If true, the expected TFRecord has to be compressed.')
    parser.add_argument('--restore',
                        type=arg_boolean, default=False,
                        help='If true, the model is restored from a previous checkpoint.')
    parser.add_argument('--data_path',
                        type=str,
                        help="Path to data directory.",
                        default='data/extracted/')
    parser.add_argument('--sequences',
                        type=str,
                        nargs='+',
                        help="If provided, overrides the train_bags.txt file",
                        default=None)
    parser.add_argument('--load_path',
                        type=str,
                        help="Path to saved model.",
                        default='data/log/saver/')
    parser.add_argument('--training_instance',
                        type=str,
                        help="Specific saved model to load. A new one will be generated if empty.",
                        default='')
    parser.add_argument('--summary_path',
                        type=str,
                        help="Path to log summaries.",
                        default='data/log/summary/')
    parser.add_argument('--base_channels',
                        type=int,
                        help="The number of base channels of the network.",
                        default=64)
    parser.add_argument('--deconv3_channels',
                        type=int,
                        help="The number of channels of the 'deconv3' layer. "
                             "If not provided it is computed using 'base_channels'",
                        default=None)
    parser.add_argument('--batch_size',
                        type=int,
                        help="Training batch size.",
                        default=8)
    parser.add_argument('--optimize_every',
                        type=int,
                        help="Specify the frequency of optimization steps. If a value > 1 is provided, "
                             "gradients are accumulated (and averaged across successive runs) and the "
                             "optimization step is performed only after optimize_every batches.",
                        default=1)
    parser.add_argument('--initial_learning_rate',
                        type=float,
                        help="Initial learning rate.",
                        default=3e-4)
    parser.add_argument('--learning_rate_decay',
                        type=float,
                        help='Rate at which the learning rate is decayed.',
                        default=0.9)
    parser.add_argument('--learning_step_decay',
                        type=float,
                        help='Frequency of lr decay in number of epochs.',
                        default=4.0)
    parser.add_argument('--number_of_steps',
                        type=int,
                        help='Number of training steps (in number of seen samples).',
                        default=600000)
    parser.add_argument('--smoothness_weight',
                        type=float,
                        help='Weight for the smoothness term in the loss function.',
                        default=0.5)
    parser.add_argument('--clip_gradient_norm',
                        type=float,
                        help='The value to be used to clip gradients.',
                        default=0)
    parser.add_argument('--clip_bias_gradient_norm',
                        type=float,
                        help='The value to be used to clip bias gradients.',
                        default=0)
    parser.add_argument('--clip_bn_gradient_norm',
                        type=float,
                        help='The value to be used to clip batch norm gradients.',
                        default=0)
    parser.add_argument('--image_height',
                        type=int,
                        help="Image height.",
                        default=256)
    parser.add_argument('--image_width',
                        type=int,
                        help="Image width.",
                        default=256)
    parser.add_argument('--loader_n_skips',
                        type=int,
                        help="The skip step to be used while reading data.",
                        default=1)
    parser.add_argument('--loader_binarize_polarity',
                        type=arg_boolean, default=False,
                        help="If polaritfies must be mapped to 0, 1.")
    parser.add_argument('--no_batch_norm',
                        type=arg_boolean, default=False,
                        help='If true, batch norm will not be performed at each layer')
    parser.add_argument('--no_aug_rot',
                        type=arg_boolean, default=False,
                        help='If true, random rotations are not performed as data augmentation.')
    parser.add_argument('--do_aug_rewind',
                        type=arg_boolean, default=False,
                        help='If true, random sequence rewind is performed during training.')
    parser.add_argument('--do_aug_flip_updown',
                        type=arg_boolean, default=False,
                        help='If true, random up down flips are performed during training.')
    parser.add_argument('--no_conv_bias',
                        type=arg_boolean, default=False,
                        help='If true, no bias is added to convolutional layers')
    parser.add_argument('--no_weight_decay',
                        type=arg_boolean, default=False,
                        help='If true, no weight decay is added to the loss')
    parser.add_argument('--count_only',
                        type=arg_boolean, default=False,
                        help='If true, inputs will consist of the event counts only.')
    parser.add_argument('--time_only',
                        type=arg_boolean, default=False,
                        help='If true, inputs will consist of the latest timestamp only.')

    # Args for testing only.
    parser.add_argument('--test_sequence',
                        type=str,
                        help="Name of the test sequence.",
                        default='outdoor_day1')
    parser.add_argument('--gt_path',
                        type=str,
                        help='Path to optical flow ground truth npz file.',
                        default='')
    parser.add_argument('--test_plot',
                        type=arg_boolean, default=False,
                        help='If true, the flow predictions will be visualized during testing.')
    parser.add_argument('--test_skip_frames',
                        type=arg_boolean, default=False,
                        help='If true, input images will be 4 frames apart.')
    parser.add_argument('--save_test_output',
                        type=arg_boolean, default=False,
                        help='If true, output flow will be saved to a npz file.')
    parser.add_argument('--test_reference_flow',
                        type=str, default=None,
                        help='The path of a saved prediction (.npz), obtained providing --save_test_output '
                             'and performed on the same test set, to be used for comparison. If provided '
                             'the optical flow error is evaluated only where both networks predicted a value.')

    # Extract Video args
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--nframes", type=int)

    # Args for MatrixLSTM
    parser.add_argument('--matrixlstm_only',
                        type=arg_boolean, default=False,
                        help='DEPRECATED: use \'matrixlstm_mode simple\'.')
    parser.add_argument('--matrixlstm_k_init',
                        type=str, default='uniform',
                        help='The LSTM kernel initializer to use: "uniform" or "orthogonal"')
    parser.add_argument('--matrixlstm_mode',
                        type=str, default='simple',
                        help='\'simple\': Only the MatrixLSTM output is used for prediction, '
                             '\'concat\': EvFlowNet frame and MatrixLSTM output are concatenated, '
                             '\'residual\': EvFlowNet frame and MatrixLSTM output are summed.')
    parser.add_argument('--matrixlstm_in_shape',
                        type=int, nargs=2, default=[256, 256],
                        help='The input shape (height, width) of the MatrixLSTM layer.')
    parser.add_argument('--matrixlstm_channels',
                        type=int, nargs='+', default=[4],
                        help='The number of channels of the events images produced '
                             'by each single MatrixLSTM layer.')
    parser.add_argument('--matrixlstm_force_channels',
                        type=int, default=4,
                        help='If >0 force the resulting MatrixLSTM frame (produced by'
                             'eventually combining different heads) to have a certain number of channels. '
                             'If this is not the case, a 1x1 convolution is added to obtain the desired channels.')
    parser.add_argument('--matrixlstm_region_shape',
                        type=arg_tuple(int), nargs='+', default=[(1, 1)],
                        help='The size of each MatrixLSTM receptive fields')
    parser.add_argument('--matrixlstm_region_stride',
                        type=arg_tuple(int), nargs='+', default=[(1, 1)],
                        help='The stride of MatrixSTM receptive fields')
    parser.add_argument('--matrixlstm_actfn',
                        type=str, default=None,
                        help='The activation function to use on the output produced by the MatrixLSTM.')
    parser.add_argument('--matrixlstm_batchnorm',
                        type=arg_boolean, default=False,
                        help='Whether a batch norm layer must be applied on the output of MatrixLSTM.')
    parser.add_argument('--matrixlstm_grad_mult',
                        type=float, default=1.0,
                        help='A multiplier applied to MatrixLSTM layer gradients.')
    parser.add_argument('--matrixlstm_maxrf',
                        type=int, nargs='+', default=[256],
                        help='The maximum number of events in each receptive field.')
    parser.add_argument('--matrixlstm_keep_last',
                        type=arg_boolean, nargs='+', default=[True],
                        help='Having set a maximum number of events in each receptive field, this '
                             'argument controls if only the last matrixlstm_maxrf events must be kept '
                             'or if matrixlstm_maxrf events must be picked at regular steps from the '
                             'whole time window.')
    parser.add_argument('--matrixlstm_time_mode',
                        type=str, default='delay',
                        help='The strategy to be used to add time information as input to the MatrixLSTM. '
                             '\'ts\': adds the normalized timestamp to each event, '
                             '\'delay\': adds the normalized delay to each event.')
    parser.add_argument('--matrixlstm_add_coords',
                        type=arg_boolean, default=True,
                        help='Iff coordiantes relative to each receptive field must be added as additioanl features.')
    parser.add_argument('--matrixlstm_add_step',
                        type=arg_boolean, default=False,
                        help='If a --matrixlstm_maxrf is provided, it may happen that some of the events '
                             'are not processed by the network. This option allow to attach to each event an '
                             'additional feature containing the number of events that have been excluded '
                             'from the last sampled event.')
    parser.add_argument('--matrixlstm_norm_relative',
                        type=arg_boolean, default=True)
    parser.add_argument('--matrixlstm_residual',
                        type=arg_boolean, default=False,
                        help='If true, the output of MatrixLSTM is added to the usual EVFlowNet input features. '
                             'Note: the number of MatrixLSTM features must match EVFlownet\'s.')
    parser.add_argument('--matrixlstm_frame_intervals',
                        type=int, default=1,
                        help='If the output frame must be composed by the concatenation of multiple '
                             'MatrixLSTM frames, each one computed on successive not overlapping '
                             'time intervals. This argument controls the number of intervals.')
    parser.add_argument('--matrixlstm_frame_intervals_mode',
                        type=str, default=None,
                        help='If "abs_ts", the original range normalized timestamp is kept as event'
                             ' feature. Timestamp normalization within each interval is controlled by the '
                             '--matrixlstm_time_mode argument. If not provided, no additional time feature '
                             'is added.')
    parser.add_argument('--matrixlstm_add_terminator_mode',
                        type=str, default=None,
                        help='If "simple" an event having all -1 features is added to each receptive field'
                             ' sequence. If "background", it uses the same procedure as "simple" but a '
                             'terminator event is also added to empty receptive fields. If not provided, '
                             'no terminator event is added.')
    parser.add_argument('--matrixlstm_add_selayer',
                        type=arg_boolean, default=False,
                        help='If matrixlstm_frame_intervals > 1, optionally add a SELayer to correlate the '
                             'output features.')

    args = parser.parse_args()
    if args.matrixlstm_only:
        print("WARNING: matrixlstm_only is deprecated, use --matrixlstm_mode simple instead!")
        args.matrixlstm_mode = 'simple'

    # Force paper arguments regardless of the provided ones
    if args.use_paper_args:
        print("WARNING: overriding arguments with paper values!!!")
        args.initial_learning_rate = 1e-5
        args.learning_step_decay = 4
        args.learning_rate_decay = 0.8
        args.no_aug_rot = True
        args.no_batch_norm = True
        args.deconv3_channels = 2
        args.loader_n_skips = 2
        args.gzip = True

    return args
