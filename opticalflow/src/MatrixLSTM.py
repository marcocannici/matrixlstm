import itertools
import tensorflow as tf
from extensions import group_rf_bounded, intervals_to_batch


def compute_pad(input_shape, region_shape, region_stride):
    """
    Determines the padding that has to be applied in any dimension

    :rtype: tuple
    :returns: tuple (padded_input_shape, paddings)
        * padded_input_shape (tuple): the new (height, width) input shape
        * paddings (tuple): how many pixels must be added to the (left, right, top, bottom)
    """

    in_h, in_w = input_shape
    stride_y, stride_x = region_stride
    reg_h, reg_w = region_shape

    new_w = (in_w - 1) * stride_x + reg_w
    new_h = (in_h - 1) * stride_y + reg_h

    pad_r = (new_w - in_w) // 2
    pad_l = new_w - (in_w + pad_r)
    pad_b = (new_h - in_h) // 2
    pad_t = new_h - (in_h + pad_b)
    return (new_h, new_w), (pad_l, pad_r, pad_t, pad_b)


def coord2idx(pad_in_shape, coords):
    """
    Given a sequence of coordinates it associates them with a unique index using a left-to-right
    top-to-bottom ordering (the pixel having index 0 is the top-left one, the one having
    index region_h * region_w is the bottom right one)
    :param torch.Tensor coords: a tensor providing 2 values in the last dimension
    (shape [..., 2]) and specifying the x and y coordinates of each point to convert
    :return:
    """
    return coords[..., 1] * pad_in_shape[1] + coords[..., 0]


def smallest_not_overlap_stride(kernel_size, stride):
    """
    Finds the smallest number greater or equal to kernel_size
    that is divisible by stride
    """
    rem = (kernel_size + stride) % stride
    return kernel_size if rem == 0 else kernel_size + stride - rem


def compute_topleft_matrix(padded_input_shape, region_shape, region_stride):
    """
    Computes a [num_rf, 2] matrix providing the coordinates of the top-left pixel in each of
    the num_rf receptive fields
    :rtype torch.tensor
    """

    with tf.name_scope('compute_topleft_matrix'):

        i_h, i_w = padded_input_shape
        x_coords = tf.cast(tf.reshape(tf.tile(tf.range(i_w), [i_h]), (i_h, i_w)), tf.float32)
        y_coords = tf.cast(tf.tile(tf.reshape(tf.range(i_h), (-1,1)), (1, i_w)), tf.float32)

        # Takes the first element of each receptive field (x coordinate of the top-left pixel)
        x_left = tf.extract_image_patches(tf.reshape(x_coords, (1, i_h, i_w, 1)),
                                          ksizes=[1, region_shape[0], region_shape[1], 1],
                                          strides=[1, region_stride[0], region_stride[1], 1],
                                          rates=[1, 1, 1, 1],
                                          padding="VALID")
        x_left = tf.reshape(x_left, (-1, region_shape[0]*region_shape[1]))[:, 0]
        # Takes the first element of each receptive field (y coordinate of the top-left pixel)
        y_top = tf.extract_image_patches(tf.reshape(y_coords, (1, i_h, i_w, 1)),
                                         ksizes=[1, region_shape[0], region_shape[1], 1],
                                         strides=[1, region_stride[0], region_stride[1], 1],
                                         rates=[1, 1, 1, 1],
                                         padding="VALID")
        y_top = tf.reshape(y_top, (-1, region_shape[0] * region_shape[1]))[:, 0]
        rf_topleft = tf.cast(tf.stack([x_left, y_top], -1), tf.int32)

        return rf_topleft


def compute_pixels2rf_matrix(region_shape, region_stride, out_shape):
    """
    Computes a [pad_in_h, pad_in_w] matrix containing for each pixel, the receptive field number in which
    the pixel is contained

    :rtype torch.Tensor
    """

    with tf.name_scope('compute_pixels2rf_matrix'):
        nrf_h, nrf_w = out_shape
        kh, kw = region_shape
        num_rf = out_shape[0] * out_shape[1]
        right_pad = region_stride[1] - region_shape[1]
        bottom_pad = region_stride[0] - region_shape[0]
        # grouped_rf.shape = [nrf, kh, kw]
        grouped_rf = tf.tile(tf.reshape(tf.range(num_rf), (num_rf, 1, 1)), (1, kh, kw))
        grouped_rf = tf.pad(grouped_rf, ((0, 0), (0, bottom_pad), (0, right_pad)),
                            mode='CONSTANT', constant_values=-1)

        pixels2rf_matrix = tf.reshape(grouped_rf, (nrf_h, -1, kw + right_pad))
        pixels2rf_matrix = tf.transpose(pixels2rf_matrix, (0, 2, 1))
        pixels2rf_matrix = tf.reshape(pixels2rf_matrix, (nrf_h * (kh + bottom_pad),
                                                         nrf_w * (kw + right_pad)))

        pixels2rf_matrix = pixels2rf_matrix[:(-bottom_pad if bottom_pad > 0 else None),
                           :(-right_pad if right_pad < 0 else None)]
        return pixels2rf_matrix


def compute_not_overlap_matrices(kernel_size, strides, frame_shape_in, frame_shape_out):
    """
    Group convolutional receptive fields together in such a way that
    all receptive fields in the same group do not overlap, given a
    certain kernel_size and stride
    """

    no_strides, initial_offsets = [], []
    for ksize, stride in zip(kernel_size, strides):
        # Compute the smallest multiple of stride that is greater than kernel_size
        no_stride = smallest_not_overlap_stride(ksize, stride)
        # Compute the number of groups, i.e., the strides that are between the first
        # kernel position and the second not overlapping position
        initial_offset = list(range(0, no_stride, stride))

        no_strides += [no_stride]
        initial_offsets += [initial_offset]

    initial_offsets = itertools.product(*initial_offsets)
    coord2rfid = tf.reshape(tf.range(frame_shape_out[0] * frame_shape_out[1]), frame_shape_out)

    abs_topleft_matrices, rel_pixels2rf_matrices = [], []
    rfid_rel2abs, output_shapes = [], []
    for offset in initial_offsets:
        # Restrict the frame removing the initial offset
        noffset_frame_shape = (frame_shape_in[0] - offset[1],
                               frame_shape_in[1] - offset[0])
        # Removes additional space on the right not covered by filters
        restricted_frame_shape = ((noffset_frame_shape[0] - kernel_size[0]) //
                                  no_strides[0] * no_strides[0] + kernel_size[0],
                                  (noffset_frame_shape[1] - kernel_size[1]) //
                                  no_strides[1] * no_strides[1] + kernel_size[1])

        if restricted_frame_shape[0] < kernel_size[0] or \
                restricted_frame_shape[1] < kernel_size[1]:
            continue

        topleft = compute_topleft_matrix(restricted_frame_shape,
                                         kernel_size, no_strides)
        # Adds the offset back
        topleft += tf.constant([offset], dtype=topleft.dtype)
        abs_topleft_matrices.append(topleft)

        # Compute the output shape of each group
        out_shape = ((restricted_frame_shape[0] - kernel_size[0]) // no_strides[0] + 1,
                     (restricted_frame_shape[1] - kernel_size[1]) // no_strides[1] + 1)
        output_shapes.append(out_shape)

        assert topleft.shape[0] == out_shape[0] * out_shape[1]
        pixels2rf = compute_pixels2rf_matrix(kernel_size, no_strides, out_shape)
        # Add back removed pixels not covered by this rf (assign rf_id=-1)
        pixels2rf = tf.pad(pixels2rf,
                           [[offset[1], noffset_frame_shape[0] - restricted_frame_shape[0]],
                            [offset[0], noffset_frame_shape[1] - restricted_frame_shape[1]]],
                           mode='CONSTANT', constant_values=-1)
        rel_pixels2rf_matrices.append(pixels2rf)

        # Compute a mapping from relative to absolute rfids
        rel2abs = tf.gather_nd(coord2rfid, tf.stack([topleft[..., 1], topleft[..., 0]], axis=-1))
        rfid_rel2abs.append(tf.cast(rel2abs, tf.int64))

    return abs_topleft_matrices, rel_pixels2rf_matrices, rfid_rel2abs, output_shapes


def group_samples(input, coords, time, lengths, add_coords, add_time_mode, add_step,
                  normalize_relative, max_events_per_rf, output_shape, region_shape,
                  groups_output_shapes, groups_rfid_rel2abs,
                  groups_rel_pixels2rf, groups_abs_topleft, keep_most_recent,
                  add_terminator_mode=None):

    # input.shape = [batch_size, padded_time_size, feature_size]
    # time.shape = [batch_size, padded_time_size, feature_size]
    n_features = input.shape[-1]
    abs_output_shape = output_shape

    with tf.name_scope('group_samples'):
        if add_coords:
            input = tf.concat([input, coords], -1)
            coords_idx = n_features
            n_features += 2

        if add_time_mode in ['ts', 'ts_max', 'delay', 'none']:

            if normalize_relative:
                # Zero offset events
                time_feature = time - tf.reshape(time[:, 0, :], (-1, 1, 1))
            else:
                if add_time_mode == 'delay':
                    # Compute the delays as the element-wise difference between the next
                    # element and the current one. Add a 0 delay for the first event
                    first_ev_delay = tf.zeros([tf.shape(time)[0], 1, 1], dtype=time.dtype)
                    time_feature = tf.concat([first_ev_delay, time[:, 1:, :] - time[:, :-1, :]], 1)
                elif add_time_mode == 'ts':
                    # Normalize each timestamp with range normalization
                    ts_max = tf.expand_dims(tf.reduce_max(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    ts_min = tf.expand_dims(tf.reduce_min(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    time_feature = (time - ts_min) / (ts_max - ts_min + 1e-8)
                elif add_time_mode == 'ts_max':
                    ts_max = tf.expand_dims(tf.reduce_max(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    ts_max += tf.cast(tf.equal(ts_max, 0), ts_max.dtype)  # prevent div by 0
                    time_feature = time / ts_max
                elif add_time_mode == 'none':
                    time_feature = time

                time_feature = tf.debugging.check_numerics(time_feature, "NaN or Inf in time_feature")

            input = tf.concat([input, tf.cast(time_feature, input.dtype)], -1)
            time_idx = n_features
            n_features += 1
        else:
            raise ValueError("add_time_mode = '{}' unknown!"
                             "".format(add_time_mode))

        # Receptive fields have been grouped into not overlapping groups.
        # We process them iteratively. If receptive fields are not overlapped
        # we will iterate over a single group with id 0

        # Lists containing the results of group_rf's
        # calls on the different receptive fields groups
        all_batch_groups, all_gr_batch_id, all_gr_last_id = [], [], []
        all_gr_h, all_gr_w = [], []
        num_not_overlap_groups = len(groups_output_shapes)

        for group_id in range(num_not_overlap_groups):
            pixel2rf_idx = groups_rel_pixels2rf[group_id]
            output_shape = groups_output_shapes[group_id]
            rfid_rel2abs = groups_rfid_rel2abs[group_id]
            rf_topleft = groups_abs_topleft[group_id]

            rf_idx = tf.gather_nd(pixel2rf_idx,
                                  tf.cast(tf.stack([coords[..., 1], coords[..., 0]], axis=-1), tf.int64))

            group_res = group_rf_bounded(input, rf_idx, lengths,
                                         max_events_per_rf,
                                         output_shape[0],
                                         output_shape[1],
                                         keep_most_recent,
                                         add_step)

            gr_batch_id, gr_last_id, rel_gr_h, rel_gr_w, batch_groups = group_res

            # Convert relative output coordinates to absolute
            rel_rfid = tf.cast((rel_gr_h * output_shape[1] + rel_gr_w), tf.int32)
            abs_rfid = tf.cast(tf.gather(rfid_rel2abs, rel_rfid), tf.int32)
            gr_h = abs_rfid // abs_output_shape[1]
            gr_w = abs_rfid % abs_output_shape[1]

            if add_terminator_mode in ['simple', 'background']:
                # Creates a fake event
                _, n_valid_rf, feature_size = tf.unstack(tf.shape(batch_groups))
                terminator_evs = tf.zeros(shape=[1, n_valid_rf, feature_size],
                                          dtype=batch_groups.dtype)
                terminator_evs -= 1.0  # The terminator event has all features to set to '-1'
                # Adds the event at the end of each sequence
                batch_groups = tf.concat([batch_groups, terminator_evs], axis=0)
                gr_last_id += 1

            if add_coords:
                batch_coords = batch_groups[..., coords_idx:coords_idx+2]  # shape [n_ev, n_rf, 2]
                groups_topleft = tf.reshape(tf.gather(rf_topleft, rel_rfid), (1, -1, 2))  # shape [1, n_rf, 2]

                den = tf.constant([region_shape[::-1]], dtype=batch_coords.dtype) - 1
                # prevent div by zero when receptive field is 1x1
                one = tf.constant(1, dtype=den.dtype)
                coords_den = tf.reshape(tf.maximum(one, den), [1, 1, 2])  # shape [1, 1, 2]
                # normalize coordinates wrt the receptive field size
                groups_topleft = tf.cast(groups_topleft, dtype=batch_coords.dtype)
                batch_norm_coords = (batch_coords - groups_topleft) / coords_den
                batch_groups = tf.concat([batch_groups[..., :coords_idx],
                                          batch_norm_coords,
                                          batch_groups[..., coords_idx+2:]], -1)

            if add_time_mode in ['ts', 'ts_max', 'delay', 'none']:
                batch_times = batch_groups[..., time_idx]

                if normalize_relative:
                    # In this case delay computation is performed after grouping
                    # otherwise delays have already been computed
                    # NOTE: this time the batch dimension is the second one
                    # batch_times.shape = [time_size, batch_size]
                    if add_time_mode == 'delay':
                        first_ev_delay = tf.zeros([1, tf.shape(batch_times)[1]], dtype=batch_times.dtype)
                        batch_times = tf.concat([first_ev_delay,
                                                 batch_times[1:, :] - batch_times[:-1, :]], axis=0)
                        max_delay = tf.reshape(tf.reduce_max(batch_times, axis=0), [1, -1])
                        batch_times = batch_times / (max_delay + 1e-6)
                    elif add_time_mode == 'ts':
                        # Normalize each timestamp with range normalization
                        ts_max = tf.expand_dims(tf.reduce_max(batch_times, axis=0), 0)  # shape [1, batch_size]
                        ts_min = tf.expand_dims(tf.reduce_min(batch_times, axis=0), 0)  # shape [1, batch_size]
                        batch_times = (batch_times - ts_min) / (ts_max - ts_min + 1e-8)
                    elif add_time_mode == 'ts_max':
                        # Normalize each timestamp with range normalization
                        ts_max = tf.expand_dims(tf.reduce_max(batch_times, axis=0), 0)  # shape [1, batch_size]
                        ts_max += tf.cast(tf.equal(ts_max, 0), ts_max.dtype)  # prevent div by 0
                        batch_times = batch_times / ts_max

                batch_groups = tf.concat([batch_groups[..., :time_idx],
                                          tf.expand_dims(batch_times, -1)], -1)
            else:
                raise ValueError("add_time_mode = '{}' unknown!"
                                 "".format(add_time_mode))

            # NOTE: We use tf.stop_gradient() to prevent TensorFlow from computing
            #       gradients of previous transformations, which is useless and
            #       consumes GPU memory
            batch_groups.set_shape([None, None, n_features])
            all_batch_groups.append(tf.stop_gradient(batch_groups))
            all_gr_batch_id.append(tf.stop_gradient(gr_batch_id))
            all_gr_last_id.append(tf.stop_gradient(gr_last_id))
            all_gr_h.append(tf.stop_gradient(gr_h))
            all_gr_w.append(tf.stop_gradient(gr_w))

        return all_gr_batch_id, all_gr_last_id, all_gr_h, all_gr_w, all_batch_groups


def assert_numerics_and_dump(check_tensor, dump_tensors, dump_name):
    import numpy as np
    def dump_tensor_np(tensor, filename):
        np.save(filename, tensor)
        return tensor
    dump_ops = []
    for i, dump_tensor in enumerate(dump_tensors):
        dump_ops.append(tf.py_func(dump_tensor_np, [dump_tensor, dump_name % i], dump_tensor.dtype))
    with tf.control_dependencies(dump_ops):
        data_op = tf.identity(check_tensor)
    assert_ops = [tf.Assert(tf.math.logical_not(tf.reduce_any(tf.is_inf(check_tensor))),
                            data=[data_op], name="assert_is_inf"),
                  tf.Assert(tf.math.logical_not(tf.reduce_any(tf.is_nan(check_tensor))),
                            data=[data_op], name="assert_is_nan")]
    return assert_ops


def ungroup_last_dense(rf_input, rf_idx, batch_size, output_shape, background=None):
    # input.shape = padded_event_size, flat_batch_size, feature_size
    feature_size = rf_input[0].shape[-1]
    dense_shape = [batch_size, output_shape[0], output_shape[1], feature_size]
    dense_out_all = None

    # We fill the output matrix iteratively, one
    # iteration for each not overlapping receptive field
    for input, gr_batch_id, gr_last_id, gr_h, gr_w in zip(rf_input, *rf_idx):
        # Filter out output of empty sequences (ie, those having negative last index position)
        # valid_lengths_mask = tf.math.greater_equal(gr_last_id, 0)
        # gather_input = input[gr_last_id, tf.range(tf.shape(input)[1])]
        input_gather_idx = tf.stack([gr_last_id, tf.range(tf.shape(input)[1])], -1)
        # input_gather_idx = tf.boolean_mask(input_gather_idx, valid_lengths_mask)
        gather_input = tf.gather_nd(input, input_gather_idx)
        # dense_out[gr_batch_id, gr_h, gr_w, hidden_size] = gather_input
        dense_scatter_idx = tf.stack([gr_batch_id, gr_h, gr_w], -1)
        # dense_scatter_idx = tf.boolean_mask(dense_scatter_idx, valid_lengths_mask)
        dense_out = tf.scatter_nd(dense_scatter_idx, gather_input, shape=dense_shape)
        if dense_out_all is None:
            dense_out_all = dense_out
        else:
            dense_out_all += dense_out

    if background is not None:
        background = tf.reshape(background, [1, 1, 1, feature_size])
        background_mask = tf.reduce_all(tf.equal(dense_out_all, 0), axis=-1)
        background_mask = tf.tile(tf.expand_dims(background_mask, -1), (1, 1, 1, feature_size))
        background_values = tf.cast(background_mask, tf.float32)
        background_values *= background
        dense_out_all += background_values

    return dense_out_all


def check_values(tensor, vmin, vmax, msg):
    tensor = tf.check_numerics(
        tensor, message=msg+" Encountered NaN or Inf values")
    ops = [
        tf.assert_less_equal(tensor, tf.constant(vmax, dtype=tensor.dtype),
                             message=msg+" Encountered > {} values".format(vmax)),
        tf.assert_greater_equal(tensor, tf.constant(vmin, dtype=tensor.dtype),
                                message=msg + " Encountered < {} values".format(vmin))]
    return ops


def shared_lstm(input, lengths, num_layers, hidden_size, kernel_initializer,
                bias_initializer=None, training=True, use_dynamic_rnn=False):

    if use_dynamic_rnn:
        with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
            if num_layers > 1:
                cells = []
                for i in range(num_layers):
                    cells.append(tf.nn.rnn_cell.LSTMCell(hidden_size,
                                                         initializer=kernel_initializer,
                                                         name="LSTMCell%d" % i))
                cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            else:
                cells = tf.nn.rnn_cell.LSTMCell(hidden_size,
                                                initializer=kernel_initializer,
                                                name="LSTMCell")

        batch_size = tf.shape(input)[1]  # time major
        initial_state = cells.zero_state(batch_size=batch_size,
                                         dtype=tf.float32)
        return tf.nn.dynamic_rnn(cells, input,
                                 sequence_length=lengths,
                                 initial_state=initial_state,
                                 time_major=True,
                                 swap_memory=True,
                                 dtype=tf.float32)
    else:
        with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers,
                                                  hidden_size,
                                                  kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer,
                                                  name="CudnnLSTM")
        input = tf.debugging.check_numerics(input, "NaN or Inf in input")
        res = lstm(input, training=training)
        r0 = tf.debugging.check_numerics(res[0], "NaN or Inf in res[0]")
        r1 = tf.debugging.check_numerics(res[1], "NaN or Inf in res[1]")
        return (r0, r1)


def dense_matrixlstm(events, lengths, input_shape, region_shape, region_stride, hidden_size, num_layers=1,
                     add_coords_feature=True, add_time_mode='delay', add_step=True, normalize_relative=True,
                     max_events_per_rf=512, kernel_initializer=None,
                     bias_initializer=None, maintain_in_shape=True, keep_most_recent=True,
                     frame_intervals=1, frame_intervals_mode=None, add_terminator_mode=None,
                     is_training=True, use_dynamic_rnn=False, name=None):

    name_scope = 'dense_matrixlstm_%s' % name if name else 'dense_matrixlstm'
    with tf.name_scope(name_scope):

        # Split each sample in mini_samples, one for each time interval, and put
        # the intervals in the batch dimension
        # from [batch_size, time_size, feature_size]
        # to [batch_size * frame_intervals, new_time_size, feature_size]
        if frame_intervals > 1:
            if frame_intervals_mode:
                time = tf.expand_dims(events[..., 2], -1)
                if frame_intervals_mode == 'abs_ts':
                    # Normalize each timestamp with range normalization
                    ts_max = tf.expand_dims(tf.reduce_max(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    ts_min = tf.expand_dims(tf.reduce_min(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    abs_time = (time - ts_min) / (ts_max - ts_min + 1e-8)
                elif frame_intervals_mode == 'abs_ts_max':
                    ts_max = tf.expand_dims(tf.reduce_max(time, axis=1), -1)  # shape [batch_size, 1, 1]
                    ts_max += tf.cast(tf.equal(ts_max, 0), ts_max.dtype)  # prevent div by 0
                    abs_time = time / ts_max

                abs_time = tf.debugging.check_numerics(abs_time, "NaN or Inf in abs_time")

                events = tf.concat([events, abs_time], axis=-1)

            events = tf.cast(events, tf.float32)
            events, lengths = intervals_to_batch(events, lengths, frame_intervals)

        coords = tf.cast(events[..., :2], tf.float32)
        features = tf.cast(events[..., 3:], tf.float32)
        time = tf.expand_dims(events[..., 2], -1)

        batch_size = tf.shape(features)[0]

        # Compute input and output shapes
        if maintain_in_shape:
            pad_in_shape, pads = compute_pad(input_shape, region_shape, region_stride)
            pad_l, pad_r, pad_t, pad_b = pads
        else:
            pad_in_shape = input_shape
            pad_l, pad_r, pad_t, pad_b = [0] * 4
        output_shape = ((pad_in_shape[0] - region_shape[0]) // region_stride[0] + 1,
                        (pad_in_shape[1] - region_shape[1]) // region_stride[1] + 1)
        overlapping_rf = region_stride[0] < region_shape[0] or \
                         region_stride[1] < region_shape[1]
        num_rf = output_shape[0] * output_shape[1]

        if overlapping_rf:
            (abs_topleft, rel_pixels2rf, rfid_rel2abs, output_shapes) = \
                compute_not_overlap_matrices(region_shape,
                                             region_stride,
                                             pad_in_shape,
                                             output_shape)
            assert len(abs_topleft) == len(rel_pixels2rf)
            assert len(abs_topleft) == len(rfid_rel2abs)
            assert len(abs_topleft) == len(output_shapes)
        else:
            output_shapes = [output_shape]
            rfid_rel2abs = [tf.range(num_rf, dtype=tf.int64)]
            rel_pixels2rf = [compute_pixels2rf_matrix(region_shape,
                                                      region_stride,
                                                      output_shape)]
            abs_topleft = [compute_topleft_matrix(pad_in_shape,
                                                  region_shape,
                                                  region_stride)]

        # Pad coordinates by adding additional top-left space
        # Note: padding here means to translate the events
        coords_offset = tf.constant([[pad_l, pad_t]], dtype=coords.dtype)
        pad_coords = coords + coords_offset

        # Group events based on their position. With this operation we select all the events in every
        # sample of the batch which are contained in each one of the receptive fields. Each one of these
        # groups is considered an independent sample (mini_sample), placed in the batch dimension and processed
        # by the LSTM independently (performs LSTM weight sharing among the different receptive fields of the
        # same image).
        # gr_*.shape = list of lists of shape [padded_time_size_mini_samples, batch_mini_samples, features_size]
        # One list for each group on not overlapping receptive fields
        group_ret = group_samples(features, pad_coords, time, lengths,
                                  add_coords=add_coords_feature,
                                  add_time_mode=add_time_mode,
                                  add_step=add_step,
                                  normalize_relative=normalize_relative,
                                  max_events_per_rf=max_events_per_rf,
                                  output_shape=output_shape,
                                  region_shape=region_shape,
                                  groups_output_shapes=output_shapes,
                                  groups_rfid_rel2abs=rfid_rel2abs,
                                  groups_rel_pixels2rf=rel_pixels2rf,
                                  groups_abs_topleft=abs_topleft,
                                  keep_most_recent=keep_most_recent,
                                  add_terminator_mode=add_terminator_mode)
        rf_gr_batch_id, rf_gr_last_id, rf_gr_h, rf_gr_w, rf_batch_groups = group_ret

        # Run the LSTM of grouped receptive fields
        shared_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers,
                                                     hidden_size,
                                                     kernel_initializer=kernel_initializer,
                                                     bias_initializer=bias_initializer)

        lstm_background = None
        # We process each not overlapping group separately
        rf_gr_lstm_out = []
        for i, (batch_groups, batch_len) in enumerate(zip(rf_batch_groups, rf_gr_last_id)):
            # Check numeric preconditions
            # with tf.control_dependencies(check_values(batch_groups, 0, 1, "[pre lstm]")):
            #     batch_groups = tf.identity(batch_groups)
            gr_lstm_out, _ = shared_lstm(batch_groups, training=is_training)
            # gr_lstm_out, _ = shared_lstm(batch_groups,
            #                              lengths=batch_len + 1,
            #                              num_layers=num_layers,
            #                              hidden_size=hidden_size,
            #                              kernel_initializer=kernel_initializer,
            #                              bias_initializer=bias_initializer,
            #                              training=is_training,
            #                              use_dynamic_rnn=use_dynamic_rnn)

            # Check numeric postconditions
            # with tf.control_dependencies(check_values(gr_lstm_out, -1, 1, "[post lstm]")):
            #     gr_lstm_out = tf.identity(gr_lstm_out)
            rf_gr_lstm_out.append(gr_lstm_out)

        if add_terminator_mode and add_terminator_mode == 'background':
            feature_size = tf.shape(rf_batch_groups[0])[-1]
            # Creates a fake event
            terminator_ev = tf.zeros(shape=[1, 1, feature_size],
                                     dtype=rf_batch_groups[0].dtype)
            terminator_ev -= 1.0
            # Compute the LSTM output (shape [1, 1, hidden_size)
            lstm_background, _ = shared_lstm(terminator_ev, training=is_training)

        # Ungroup receptive fields back
        ungr_dense = ungroup_last_dense(rf_gr_lstm_out,
                                        (rf_gr_batch_id, rf_gr_last_id, rf_gr_h, rf_gr_w),
                                        batch_size, output_shape, lstm_background)

        if frame_intervals > 1:
            channel_size = ungr_dense.shape[-1]
            # from [batch_size * frame_intervals, H, W, C]
            # to [batch_size, frame_intervals, H, W, C]
            ungr_dense = tf.reshape(ungr_dense, [-1, frame_intervals, output_shape[0],
                                                 output_shape[1], channel_size])
            # to [batch_size, H, W, frame_intervals, C]
            ungr_dense = tf.transpose(ungr_dense, [0, 2, 3, 1, 4])
            ungr_dense = tf.debugging.check_numerics(ungr_dense, "NaN or Inf in ungr_dense")
            # to [batch_size, H, W, frame_intervals * C]
            ungr_dense = tf.reshape(ungr_dense, [-1, output_shape[0], output_shape[1],
                                                 frame_intervals * channel_size])

        return ungr_dense


if __name__ == "__main__":
    sess = tf.InteractiveSession()

    events = tf.constant(tf.random_uniform([2, 10, 4], 0, 5, dtype=tf.int32).eval(), dtype=tf.float32)
    lengths = tf.constant([10, 10])
    matrixlstm_in_shape = [5, 5]
    matrixlstm_region_shape = [1, 1]
    matrixlstm_region_stride = [1, 1]
    max_events_per_rf = 4

    matrixlstm_image = dense_matrixlstm(events, lengths, input_shape=matrixlstm_in_shape,
                                        region_shape=matrixlstm_region_shape,
                                        region_stride=matrixlstm_region_stride,
                                        hidden_size=2, num_layers=1,
                                        add_coords_feature=True, add_time_mode='delay',
                                        add_step=True, normalize_relative=True,
                                        max_events_per_rf=max_events_per_rf,
                                        kernel_initializer=None, bias_initializer=None,
                                        maintain_in_shape=True, keep_most_recent=False,
                                        frame_intervals=1, is_training=True, name=None)
