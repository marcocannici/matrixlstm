#!/usr/bin/env python
import copy
import math
import tensorflow as tf
import numpy as np
from basic_layers import *
from MatrixLSTM import dense_matrixlstm

_BASE_CHANNELS = 64


def encoder(inputs, is_training, data_format, do_batch_norm=False, use_conv_bias=True,
            base_channels=_BASE_CHANNELS):
    skip_connections = {}
    with tf.variable_scope('encoder'):
        for i in range(4):
            inputs = general_conv2d(inputs,
                                    name='conv{}'.format(i),
                                    channelsout=(2**i)*base_channels,
                                    do_batch_norm=do_batch_norm,
                                    is_training=is_training,
                                    data_format=data_format,
                                    use_bias=use_conv_bias)
            skip_connections['skip{}'.format(i)] = inputs

    return inputs, skip_connections


def transition(inputs, is_training, data_format, do_batch_norm=False,
               use_conv_bias=True, base_channels=_BASE_CHANNELS):
    with tf.variable_scope('transition'):
        for i in range(2):
            inputs = build_resnet_block(inputs,
                                        channelsout=8*base_channels,
                                        is_training=is_training,
                                        do_batch_norm=do_batch_norm,
                                        data_format=data_format,
                                        use_conv_bias=use_conv_bias,
                                        name='res{}'.format(i))
    return inputs


def decoder(inputs, skip_connection, is_training, data_format, do_batch_norm=False,
            use_conv_bias=True, base_channels=_BASE_CHANNELS, deconv3_channels=None):
    with tf.variable_scope('decoder'):
        flow_dict = {}
        for i in range(4):
            # Skip connection.
            inputs = tf.concat([inputs, skip_connection['skip{}'.format(3-i)]],
                               axis=1 if data_format=='channels_first' else -1)

            channelsout = deconv3_channels \
                if deconv3_channels and i == 3 \
                else (2**(2-i))*base_channels
            inputs = upsample_conv2d(inputs,
                                     name='deconv{}'.format(i),
                                     channelsout=channelsout,
                                     do_batch_norm=do_batch_norm,
                                     is_training=is_training,
                                     data_format=data_format,
                                     use_conv_bias=use_conv_bias)

            flow = predict_flow(inputs,
                                name='flow{}'.format(i),
                                is_training=is_training,
                                data_format=data_format,
                                use_conv_bias=use_conv_bias) * 256.

            inputs = tf.concat([inputs, flow], axis=1 if data_format=='channels_first' else -1)

            if data_format == 'channels_first':
                flow = tf.transpose(flow, [0,2,3,1])

            flow_dict['flow{}'.format(i)] = flow
    return flow_dict


def model(event_image, is_training=True, data_format=None, do_batch_norm=False, use_conv_bias=True,
          base_channels=_BASE_CHANNELS, deconv3_channels=None, count_only=False, time_only=False):
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    if count_only and time_only:
        raise ValueError("'count_only' and 'time_only? cannot be both True")
    if count_only:
        print("Using only event counts")
        event_image = event_image[..., :2]
    elif time_only:
        print("Using only event timestamps")
        event_image = event_image[..., 2:]

    with tf.variable_scope('vs'):
        if data_format == 'channels_first':
            inputs = tf.transpose(event_image, [0,3,1,2])
        else:
            inputs = event_image

        inputs, skip_connections = encoder(inputs, is_training, data_format,
                                           do_batch_norm=do_batch_norm,
                                           use_conv_bias=use_conv_bias,
                                           base_channels=base_channels)
        inputs = transition(inputs, is_training, data_format,
                            do_batch_norm=do_batch_norm,
                            use_conv_bias=use_conv_bias,
                            base_channels=base_channels)
        flow_dict = decoder(inputs, skip_connections, is_training, data_format,
                            do_batch_norm=do_batch_norm,
                            use_conv_bias=use_conv_bias,
                            base_channels=base_channels,
                            deconv3_channels=deconv3_channels)

    return flow_dict


def extend_to_list(val_or_list, num):
    if isinstance(val_or_list, list):
        if len(val_or_list) == 1:
            val_or_list = [copy.deepcopy(val_or_list[0]) for _ in range(num)]
        elif len(val_or_list) != num:
            raise RuntimeError("Cannot extend the provided value to the desired length")
    else:
        val_or_list = [copy.deepcopy(val_or_list) for _ in range(num)]
    return val_or_list


def num_heads(args):
    args_len = [len(a) if isinstance(a, list) else 1 for a in args]
    return max(args_len)


def matrixlstm_model(events, lengths, event_image, batch_size, matrixlstm_mode, matrixlstm_chan,
                     matrixlstm_force_channels=4, matrixlstm_in_shape=(256, 256), matrixlstm_region_shape=(1, 1),
                     matrixlstm_region_stride=(1, 1), matrixlstm_actfn=None, matrixlstm_batchnorm=False,
                     matrixlstm_maxrf=512, matrixlstm_keep_last=True, matrixlstm_time_mode='delay',
                     matrixlstm_norm_relative=False, matrixlstm_add_step=False, matrixlstm_add_coords=True,
                     matrixlstm_frame_intervals=1, matrixlstm_frame_intervals_mode=None,
                     matrixlstm_add_terminator_mode=None, matrixlstm_init='uniform', matrixlstm_add_selayer=False,
                     is_training=True, data_format=None, do_batch_norm=False, use_conv_bias=True, aug_rot_angle=None,
                     aug_crop_bbox=None, base_channels=_BASE_CHANNELS, deconv3_channels=None, count_only=False,
                     time_only=False):

    num_lstm_heads = num_heads([matrixlstm_chan, matrixlstm_region_shape,
                                matrixlstm_region_stride, matrixlstm_maxrf,
                                matrixlstm_keep_last])
    matrixlstm_chan = extend_to_list(matrixlstm_chan, num_lstm_heads)
    matrixlstm_region_shape = extend_to_list(matrixlstm_region_shape, num_lstm_heads)
    matrixlstm_region_stride = extend_to_list(matrixlstm_region_stride, num_lstm_heads)
    matrixlstm_maxrf = extend_to_list(matrixlstm_maxrf, num_lstm_heads)
    matrixlstm_keep_last = extend_to_list(matrixlstm_keep_last, num_lstm_heads)

    print("matrixlstm_keep_last:", matrixlstm_keep_last)
    print("matrixlstm_time_mode:", matrixlstm_time_mode)
    print("matrixlstm_norm_relative:", matrixlstm_norm_relative)
    print("matrixlstm_add_step:", matrixlstm_add_step)

    if count_only and time_only:
        raise ValueError("'count_only' and 'time_only? cannot be both True")
    if count_only:
        print("Using only event counts")
        event_image = event_image[..., :2]
    elif time_only:
        print("Using only event timestamps")
        event_image = event_image[..., 2:]

    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    heads_results = []
    for head_id in range(num_lstm_heads):

        head_name = "head_%d" % head_id if num_lstm_heads > 1 else None
        kh, kw = matrixlstm_region_shape[head_id]
        if (kh > 1 or kw > 1) and matrixlstm_maxrf[head_id] > 128:
            matrixlstm_maxrf[head_id] = 128
            print("WARNING: Reducing max_events_per_rf for head {} with {}x{} receptive field to {}"
                  "".format(head_id, kh, kw, matrixlstm_maxrf[head_id]))

        init_std = 1.0 / math.sqrt(matrixlstm_chan[head_id])
        initializer = tf.random_uniform_initializer(-init_std, init_std)
        if matrixlstm_init == "uniform":
            k_initializer = initializer
        elif matrixlstm_init == "orthogonal":
            k_initializer = tf.orthogonal_initializer()
        else:
            raise ValueError("'{}' is not supported!".format(matrixlstm_init))
        matrixlstm_image = dense_matrixlstm(events, lengths, input_shape=matrixlstm_in_shape,
                                            region_shape=matrixlstm_region_shape[head_id],
                                            region_stride=matrixlstm_region_stride[head_id],
                                            hidden_size=matrixlstm_chan[head_id], num_layers=1,
                                            add_coords_feature=matrixlstm_add_coords,
                                            add_time_mode=matrixlstm_time_mode,
                                            add_step=matrixlstm_add_step,
                                            normalize_relative=matrixlstm_norm_relative,
                                            max_events_per_rf=matrixlstm_maxrf[head_id],
                                            kernel_initializer=k_initializer,
                                            bias_initializer=initializer,
                                            maintain_in_shape=True,
                                            keep_most_recent=matrixlstm_keep_last[head_id],
                                            frame_intervals=matrixlstm_frame_intervals,
                                            frame_intervals_mode=matrixlstm_frame_intervals_mode,
                                            add_terminator_mode=matrixlstm_add_terminator_mode,
                                            is_training=is_training, name=head_name)

        if aug_rot_angle is not None or aug_crop_bbox is not None:
            print("MatrixLSTM augmentation rot_angle {}".format(aug_rot_angle is not None))
            print("MatrixLSTM augmentation crop_bbox {}".format(aug_crop_bbox is not None))

            with tf.name_scope('augmentation'):
                aug_images = []
                for b in range(batch_size):
                    img = matrixlstm_image[b]
                    if aug_rot_angle is not None:
                        img = tf.contrib.image.rotate(img, aug_rot_angle[b],
                                                      interpolation='NEAREST')
                    if aug_crop_bbox is not None:
                        img = tf.image.crop_to_bounding_box(img,
                                                            offset_height=aug_crop_bbox[0][b],
                                                            offset_width=aug_crop_bbox[1][b],
                                                            target_height=aug_crop_bbox[2][b],
                                                            target_width=aug_crop_bbox[3][b])
                    aug_images.append(img)
                matrixlstm_image = tf.stack(aug_images, axis=0)

        if matrixlstm_frame_intervals > 1 and matrixlstm_add_selayer is True:
            print("Using SELayer")
            matrixlstm_image = squeeze_excitation_layer(
                matrixlstm_image,
                out_dim=matrixlstm_chan[head_id] * matrixlstm_frame_intervals,
                ratio=1, layer_name="SELayer")

        if matrixlstm_actfn is not None:
            print("Using activation function tf.%s" % matrixlstm_actfn)
            actfn = getattr(tf, matrixlstm_actfn)
            matrixlstm_image = actfn(matrixlstm_image)

        if matrixlstm_batchnorm:
            print("Using MatrixLSTM BatchNorm")
            matrixlstm_image = tf.layers.batch_normalization(matrixlstm_image,
                                                             axis=-1,
                                                             epsilon=1e-5,
                                                             gamma_initializer=tf.constant_initializer([0.01]),
                                                             name='matrixlstm_bn',
                                                             training=is_training)
        heads_results.append(matrixlstm_image)

    # Merge heads
    matrixlstm_image = tf.concat(heads_results, axis=-1)
    tot_chans = sum(matrixlstm_chan) * matrixlstm_frame_intervals
    if matrixlstm_force_channels > 0 and tot_chans != matrixlstm_force_channels:
        print("WARNING: Adding layer to project matrixlstm_image features to {} values"
              "".format(matrixlstm_force_channels))
        # Map the MatrixLSTM output to a feature map having 4 channels
        matrixlstm_image = tf.layers.conv2d(matrixlstm_image,
                                            filters=matrixlstm_force_channels, kernel_size=1, strides=1,
                                            use_bias=True, activation=tf.nn.relu,
                                            kernel_initializer= \
                                                tf.contrib.layers.variance_scaling_initializer(factor=0.1),
                                            bias_initializer=tf.constant_initializer(0.001),
                                            data_format='channels_last',
                                            name="matrixlstm_out_conv")

    if matrixlstm_mode == 'simple':
        print("Using only MatrixLSTM input layer")
        event_image = matrixlstm_image
    elif matrixlstm_mode == 'concat':
        print("Using both MatrixLSTM and event image layers")
        event_image = tf.concat([matrixlstm_image, event_image], axis=-1)
    elif matrixlstm_mode == 'residual':
        print("Using MatrixLSTM residual mode")
        event_image = matrixlstm_image + event_image
    elif matrixlstm_mode == 'smoothing':
        print("Using MatrixLSTM smoothing mode")
        print("Compute a 3x3x{} convolution on the frame and concatenate it to the frame itself"
              "".format(matrixlstm_image.shape[-1]))
        smooth_frame = tf.layers.conv2d(matrixlstm_image,
                                        filters=matrixlstm_image.shape[-1], kernel_size=3, strides=1,
                                        use_bias=True, padding='SAME',
                                        kernel_initializer= \
                                            tf.contrib.layers.variance_scaling_initializer(factor=0.1),
                                        bias_initializer=tf.constant_initializer(0.001),
                                        data_format='channels_last',
                                        name="matrixlstm_smooth_conv")
        event_image = tf.concat([matrixlstm_image, smooth_frame], axis=-1)

    return model(event_image, is_training, data_format,
                 do_batch_norm, use_conv_bias,
                 base_channels, deconv3_channels), event_image
