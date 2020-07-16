import time as timer
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs import utils
import itertools
from matrixlstm_helpers import (group_rf_gpu, group_rf_bounded_gpu, group_rf_bounded_overlap_gpu,
                                n_interval_events_gpu, intervals_to_batch_gpu)


class MatrixLSTM(nn.Module):

    def __init__(self, input_shape, region_shape, region_stride, input_size, hidden_size, num_layers=1,
                 bias=True, lstm_type="LSTM", add_coords_feature=True, add_time_feature_mode='delay',
                 normalize_relative=True, max_events_per_rf=512, maintain_in_shape=True,
                 keep_most_recent=True, frame_intervals=1, frame_intervals_mode=None):
        super(MatrixLSTM, self).__init__()

        self.input_shape = input_shape
        self.region_shape = region_shape
        self.region_stride = region_stride

        add_features = int(add_coords_feature) * 2 + \
                       int(add_time_feature_mode is not None) + \
                       int(frame_intervals > 1 and frame_intervals_mode is not None)
        self.input_size = input_size + add_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.lstm_type = lstm_type

        self.add_coord_features = add_coords_feature
        self.add_time_feature_mode = add_time_feature_mode
        self.normalize_relative = normalize_relative
        self.register_buffer('max_events_per_rf', torch.tensor(max_events_per_rf))
        self.maintain_in_shape = maintain_in_shape
        self.keep_most_recent = keep_most_recent
        self.frame_intervals = frame_intervals
        self.frame_intervals_mode = frame_intervals_mode

        if self.maintain_in_shape:
            self.padded_input_shape, pads = self.compute_pad()
            self.pad_l, self.pad_r, self.pad_t, self.pad_b = pads
        else:
            self.padded_input_shape = self.input_shape
            self.pad_l, self.pad_r, self.pad_t, self.pad_b = [0] * 4

        self.output_shape = ((self.padded_input_shape[0] - self.region_shape[0]) // self.region_stride[0] + 1,
                             (self.padded_input_shape[1] - self.region_shape[1]) // self.region_stride[1] + 1)
        self.out_channels = self.hidden_size * self.frame_intervals

        assert maintain_in_shape is False or (self.output_shape[0] == self.input_shape[0] and
                                              self.output_shape[1] == self.input_shape[1])

        if self.lstm_type == "LSTM":
            self.shared_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, bias=self.bias, batch_first=False)
        else:
            raise ValueError("LSTM type %s not supported!", self.lstm_type)

        self.overlapping_rf = self.region_stride[0] < self.region_shape[0] or \
                              self.region_stride[1] < self.region_shape[1]

        self.num_rf = self.output_shape[0] * self.output_shape[1]

        if self.overlapping_rf:
            (abs_topleft, rel_pixels2rf, rfid_rel2abs, output_shapes) = \
                self.compute_not_overlap_matrices(self.region_shape,
                                                  self.region_stride,
                                                  self.padded_input_shape,
                                                  self.output_shape)
            assert len(abs_topleft) == len(rel_pixels2rf)
            assert len(abs_topleft) == len(rfid_rel2abs)
            assert len(abs_topleft) == len(output_shapes)

            self.num_not_overlap_groups = len(abs_topleft)
            group_values = zip(abs_topleft, rel_pixels2rf, rfid_rel2abs, output_shapes)
            for i, (topleft, pixels2rf, rel2abs, out_shape) in enumerate(group_values):
                self.register_buffer('rf_topleft_group%d' % i, topleft)
                self.register_buffer('pixel2rf_idx_group%d' % i, pixels2rf)
                self.register_buffer('rfid_rel2abs_group%d' % i, rel2abs)
                setattr(self, 'output_shape_group%d' % i, out_shape)
        else:
            self.num_not_overlap_groups = 1
            self.output_shape_group0 = self.output_shape
            self.register_buffer('rfid_rel2abs_group0', torch.arange(self.num_rf))
            self.register_buffer('pixel2rf_idx_group0', self.compute_pixels2rf_matrix(self.padded_input_shape,
                                                                                      self.region_shape,
                                                                                      self.region_stride,
                                                                                      self.num_rf))
            self.register_buffer('rf_topleft_group0', self.compute_topleft_matrix(self.padded_input_shape,
                                                                                  self.region_shape,
                                                                                  self.region_stride))

    def reset_parameters(self):
        self.shared_lstm.reset_parameters()

    def coord2idx(self, coords):
        """
        Given a sequence of coordinates it associates them with a unique index using a left-to-right
        top-to-bottom ordering (the pixel having index 0 is the top-left one, the one having
        index region_h * region_w is the bottom right one)
        :param torch.Tensor coords: a tensor providing 2 values in the last dimension
        (shape [..., 2]) and specifying the x and y coordinates of each point to convert
        :return:
        """
        return coords[..., 1] * self.padded_input_shape[1] + coords[..., 0]

    def coord2rf(self, coords):
        """
        Given a sequence of coordinates it associates them with the index of the receptive field they are
        contained on. You can use this function only if the receptive fields do not overlap
        :param torch.Tensor coords: a tensor providing 2 values in the last dimension
        (shape [..., 2]) and specifying the x and y coordinates of each point to convert
        :return:
        """
        return self.pixel2rf_idx[coords[..., 1].type(torch.int64), coords[..., 0].type(torch.int64)]

    @staticmethod
    def smallest_not_overlap_stride(kernel_size, stride):
        """
        Finds the smallest number greater or equal to kernel_size
        that is divisible by stride
        """
        rem = (kernel_size + stride) % stride
        return kernel_size if rem == 0 else kernel_size + stride - rem

    @classmethod
    def compute_not_overlap_matrices(cls, kernel_size, strides, frame_shape_in, frame_shape_out):
        """
        Group convolutional receptive fields together in such a way that
        all receptive fields in the same group do not overlap, given a
        certain kernel_size and stride
        """

        no_strides, initial_offsets = [], []
        for ksize, stride in zip(kernel_size, strides):
            # Compute the smallest multiple of stride that is greater than kernel_size
            no_stride = cls.smallest_not_overlap_stride(ksize, stride)
            # Compute the number of groups, i.e., the strides that are between the first
            # kernel position and the second not overlapping position
            initial_offset = list(range(0, no_stride, stride))

            no_strides += [no_stride]
            initial_offsets += [initial_offset]

        initial_offsets = itertools.product(*initial_offsets)
        coord2rfid = torch.arange(frame_shape_out[0] * frame_shape_out[1]).view(frame_shape_out)

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

            topleft = cls.compute_topleft_matrix(restricted_frame_shape,
                                                 kernel_size, no_strides)
            # Adds the offset back
            topleft += topleft.new_tensor([offset])
            abs_topleft_matrices.append(topleft)

            num_rf = topleft.shape[0]
            pixels2rf = cls.compute_pixels2rf_matrix(restricted_frame_shape,
                                                     kernel_size, no_strides, num_rf)
            # Add back removed pixels not covered by this rf (assign rf_id=-1)
            pixels2rf = F.pad(pixels2rf,
                              [offset[0], noffset_frame_shape[1] - restricted_frame_shape[1],
                               offset[1], noffset_frame_shape[0] - restricted_frame_shape[0]],
                              mode='constant', value=-1)
            rel_pixels2rf_matrices.append(pixels2rf)

            # Compute a mapping from relative to absolute rfids
            rel2abs = coord2rfid[topleft[:, 1].type(torch.int64),
                                 topleft[:, 0].type(torch.int64)]
            rfid_rel2abs.append(rel2abs)

            # Compute the output shape of each group
            output_shapes.append(((restricted_frame_shape[0] - kernel_size[0]) // no_strides[0] + 1,
                                  (restricted_frame_shape[1] - kernel_size[1]) // no_strides[1] + 1))

        return abs_topleft_matrices, rel_pixels2rf_matrices, rfid_rel2abs, output_shapes

    @staticmethod
    def compute_topleft_matrix(input_shape, region_shape, region_stride):
        """
        Computes a [num_rf, 2] matrix providing the coordinates of the top-left pixel in each of
        the num_rf receptive fields
        :rtype torch.tensor
        """

        i_h, i_w = input_shape
        x_coords = torch.arange(i_w).repeat(i_h, 1).float()
        y_coords = torch.arange(i_h).view(i_h, 1).expand(i_h, i_w).float()

        # Takes the first element of each receptive field (x coordinate of the top-left pixel)
        x_left = F.unfold(x_coords.view(1, 1, i_h, i_w),
                          kernel_size=region_shape,
                          stride=region_stride).squeeze(0)[0, :]
        # Takes the first element of each receptive field (y coordinate of the top-left pixel)
        y_top = F.unfold(y_coords.view(1, 1, i_h, i_w),
                         kernel_size=region_shape,
                         stride=region_stride).squeeze(0)[0, :]
        rf_topleft = torch.stack([x_left, y_top], dim=-1)

        return rf_topleft

    @staticmethod
    def compute_pixels2rf_matrix(input_shape, region_shape, region_stride, num_rf):
        """
        Computes a [pad_in_h, pad_in_w] matrix containing for each pixel, the receptive field number in which
        the pixel is contained

        :rtype torch.Tensor
        """

        flat_matrix = torch.arange(num_rf) \
            .repeat(region_shape[0] * region_shape[1]) \
            .view(1, -1, num_rf).float()
        pixels2rf_matrix = F.fold(flat_matrix, input_shape,
                                  kernel_size=region_shape, stride=region_stride)
        return pixels2rf_matrix.type(torch.int64).view(input_shape)

    def compute_rf2pixels_matrix(self):
        """
        Computes a [num_rf, region_h * region_w] matrix containing for each of the
        num_rf receptive fields in which the input field of view is divided, which are
        the pixel contained inside it. Pixels are specified as indices using a left-to-right
        top-to-bottom ordering (the pixel having index 0 is the top-left one, the one having
        index region_h * region_w is the bottom right one)

        :rtype torch.Tensor
        """

        in_h, in_w = self.padded_input_shape

        # A matrix containing a unique indexing for the pixels
        # (using a left-to-right top-to-bottom ordering starting from
        # the pixel in the top-left position)
        pixel_idx = torch.arange(in_h * in_w).view(1, 1, in_h, in_w).float()
        # Group pixel indices info receptive fields
        # rf2pixel_idx.shape = [region_h * region_w, num_rf]
        rf2pixel_idx = F.unfold(pixel_idx, kernel_size=self.region_shape,
                                stride=self.region_stride).squeeze(0)

        return rf2pixel_idx.permute(1, 0)

    def compute_pad(self):
        """
        Determines the padding that has to be applied in any dimension to maintain the same
        input frame size also as output to the MatrixLSTM

        :rtype: tuple
        :returns: tuple (padded_input_shape, paddings)
            * padded_input_shape (tuple): the new (height, width) input shape
            * paddings (tuple): how many pixels must be added to the (left, right, top, bottom)
        """

        in_h, in_w = self.input_shape
        stride_y, stride_x = self.region_stride
        reg_h, reg_w = self.region_shape

        new_w = (in_w - 1) * stride_x + reg_w
        new_h = (in_h - 1) * stride_y + reg_h

        pad_r = (new_w - in_w) // 2
        pad_l = new_w - (in_w + pad_r)
        pad_b = (new_h - in_h) // 2
        pad_t = new_h - (in_h + pad_b)
        return (new_h, new_w), (pad_l, pad_r, pad_t, pad_b)

    def intervals_to_batch(self, events, time, coords, lengths):
        batch_size = events.shape[0]
        time = time.squeeze(-1).float()  # shape [batch_size, event_size]
        last_idx = (lengths - 1).long()
        batch_idx = torch.arange(batch_size, device=events.device, dtype=torch.int64)
        # Computes the ts percentage of each event
        ts_mins = time[:, 0].reshape(-1, 1)
        ts_maxs = time[batch_idx, last_idx].reshape(-1, 1)
        # ts_maxs = torch.max(time, dim=-1)[0].reshape(-1, 1)
        ts_percentage = (time - ts_mins) / (ts_maxs - ts_mins + 1e-5)

        # shape: [batch_size, n_intervals]
        inter_ev_count = n_interval_events_gpu(ts_percentage, lengths, self.frame_intervals)
        new_lengths = inter_ev_count.reshape([-1])
        new_event_size = new_lengths.max()
        new_events = intervals_to_batch_gpu(events.contiguous(), inter_ev_count, new_event_size)
        new_time = intervals_to_batch_gpu(time.unsqueeze(-1).contiguous(), inter_ev_count, new_event_size)
        new_coords = intervals_to_batch_gpu(coords.unsqueeze(-1).contiguous(), inter_ev_count, new_event_size)
        return new_events, new_time, new_coords, new_lengths

    def group_samples(self, input, coords, time, lengths=None):
        """
        :param torch.Tensor input: a [batch_size, time_size, features_size] tensor
        :param torch.Tensor coords: a [batch_size, time_size, 2] tensor containing <x,y> coordinates in
            the last dimension
        :param torch.Tensor time: a [batch_size, time_size, 1] tensor providing the timestamp of each input value
        :return:
        """

        device = input.device

        if self.add_coord_features:
            coords_idx = input.shape[-1]
            input = torch.cat([input, coords.type(input.dtype)], dim=-1)

        if self.add_time_feature_mode in ['ts', 'ts_max', 'delay', 'delay_norm', 'none']:
            time = time.float()
            if self.normalize_relative:
                # Delay computation is performed within each receptive field independently,
                # here we only add the timestamps to the input (setting the first event to 0)
                time_feature = time - time[:, 0, :].view(-1, 1, 1)
            else:
                if self.add_time_feature_mode == 'delay':
                    # Compute the delays as the element-wise difference between the next
                    # element and the current one. Add a 0 delay for the first event
                    time_feature = torch.cat([time.new_zeros(time.shape[0], 1, 1),
                                              time[:, 1:, :] - time[:, :-1, :]], dim=1)
                elif self.add_time_feature_mode == 'delay_norm':
                    delays = torch.cat([time.new_zeros(time.shape[0], 1, 1),
                                        time[:, 1:, :] - time[:, :-1, :]], dim=1)
                    delay_max = torch.max(delays, dim=1)[0].unsqueeze(-1)
                    time_feature = delays / (delay_max + 1e-8)
                elif self.add_time_feature_mode == 'ts':
                    # Normalize each timestamp with range normalization
                    ts_max = torch.max(time, dim=1)[0].unsqueeze(-1)  # shape [batch_size, 1, 1]
                    ts_min = torch.min(time, dim=1)[0].unsqueeze(-1)  # shape [batch_size, 1, 1]
                    time_feature = (time - ts_min) / (ts_max - ts_min + 1e-8)
                elif self.add_time_feature_mode == 'ts_max':
                    ts_max = torch.max(time, dim=1)[0].unsqueeze(-1)  # shape [batch_size, 1, 1]
                    ts_max += (ts_max == 0).type(ts_max.dtype)  # prevent div by 0
                    time_feature = time / ts_max
                elif self.add_time_feature_mode == 'none':
                    time_feature = time

            time_idx = input.shape[-1]
            input = torch.cat([input, time_feature.type(input.dtype)], dim=-1)
        else:
            raise ValueError("add_time_feature_mode = %s is not valid!" % self.add_time_feature_mode)

        # Receptive fields have been grouped into not overlapping groups.
        # We process them iteratively. If receptive fields are not overlapped
        # we will iterate over a single group with id 0

        # Lists containing the results of group_rf's
        # calls on the different receptive fields groups
        all_batch_groups, all_gr_batch_id, all_gr_last_id = [], [], []
        all_gr_h, all_gr_w = [], []

        for group_id in range(self.num_not_overlap_groups):
            # Retrieve the pixel2rf matrix of the current group
            pixel2rf_idx = getattr(self, 'pixel2rf_idx_group%d' % group_id)
            output_shape = getattr(self, 'output_shape_group%d' % group_id)
            rfid_rel2abs = getattr(self, 'rfid_rel2abs_group%d' % group_id)
            rf_topleft = getattr(self, 'rf_topleft_group%d' % group_id)

            ids = pixel2rf_idx[coords[..., 1].type(torch.int64),
                               coords[..., 0].type(torch.int64)]

            # groups.shape = [n_ev, n_rf, n_feat]
            gr_batch_id, gr_last_id, rel_gr_h, rel_gr_w, batch_groups = \
                group_rf_bounded_gpu(input, ids, lengths,
                                     self.max_events_per_rf,
                                     *output_shape,
                                     self.keep_most_recent)

            # Convert relative output coordinates to absolute
            rel_rfid = (rel_gr_h * output_shape[1] + rel_gr_w).type(torch.int64)
            abs_rfid = rfid_rel2abs[rel_rfid]
            gr_h = abs_rfid // self.output_shape[1]
            gr_w = abs_rfid % self.output_shape[1]

            if self.add_coord_features:
                batch_coords = batch_groups[..., coords_idx:coords_idx+2]  # shape [n_ev, n_rf, 2]
                groups_topleft = rf_topleft[rel_rfid].view(1, -1, 2)  # shape [1, n_rf, 2]

                den = torch.tensor([self.region_shape[::-1]], dtype=torch.float, device=device) - 1
                # prevent div by zero when receptive field is 1x1
                coords_den = torch.max(torch.ones_like(den), den).view(1, 1, 2)  # shape [1, 1, 2]
                # normalize coordinates wrt the receptive field size
                batch_norm_coords = (batch_coords - groups_topleft) / coords_den
                batch_groups[..., coords_idx:coords_idx + 2] = batch_norm_coords

            if self.add_time_feature_mode in ['ts', 'ts_max', 'delay', 'delay_norm', 'none']:
                batch_times = batch_groups[..., time_idx]

                if self.normalize_relative:
                    # In this case delay computation is performed after grouping
                    # otherwise delays have already been computed
                    # NOTE: this time the batch dimension is the second one
                    # batch_times.shape = [time_size, batch_size]
                    if self.add_time_feature_mode == 'delay':
                        batch_times = torch.cat([batch_times.new_zeros(1, batch_times.shape[1]),
                                                 batch_times[1:, :] - batch_times[:-1, :]], dim=0)
                    elif self.add_time_feature_mode == 'delay_norm':
                        batch_times = torch.cat([batch_times.new_zeros(1, batch_times.shape[1]),
                                                 batch_times[1:, :] - batch_times[:-1, :]], dim=0)
                        max_delay = torch.max(batch_times, dim=0)[0].view(1, -1)  # shape [1, n_rf]
                        batch_times = batch_times / (max_delay + 1e-6)
                    elif self.add_time_feature_mode == 'ts':
                        # Normalize each timestamp with range normalization
                        ts_max = torch.max(batch_times, dim=0)[0].unsqueeze(0)  # shape [1, batch_size]
                        ts_min = torch.min(batch_times, dim=0)[0].unsqueeze(0)  # shape [1, batch_size]
                        batch_times = (batch_times - ts_min) / (ts_max - ts_min + 1e-8)
                    elif self.add_time_feature_mode == 'ts_max':
                        # Normalize each timestamp with range normalization
                        ts_max = torch.max(batch_times, dim=0)[0].unsqueeze(0)  # shape [1, batch_size]
                        ts_max += (ts_max == 0).type(ts_max.dtype)  # prevent div by 0
                        batch_times = batch_times / ts_max

                batch_groups[..., time_idx] = batch_times

            all_batch_groups.append(batch_groups)
            all_gr_batch_id.append(gr_batch_id)
            all_gr_last_id.append(gr_last_id)
            all_gr_h.append(gr_h)
            all_gr_w.append(gr_w)

        return all_gr_batch_id, all_gr_last_id, all_gr_h, all_gr_w, all_batch_groups

    def ungroup_last_dense(self, rf_input, rf_idx, batch_size):

        # input.shape = padded_event_size, flat_batch_size, feature_size
        _, _, feature_size = rf_input[0].shape
        dense_out = torch.zeros([batch_size, *self.output_shape, feature_size],
                                device=rf_input[0].device)

        # We fill the output matrix iteratively, one
        # iteration for each not overlapping receptive field
        for input, gr_batch_id, gr_last_id, gr_h, gr_w in zip(rf_input, *rf_idx):
            dense_out[gr_batch_id, gr_h, gr_w] = input[gr_last_id, torch.arange(input.shape[1])]

        return dense_out

    def forward(self, input, hc_0=None):
        """

        :param Tuple[Tensor, Tensor, Tensor] input: a tuple
        :param hc_0:
        :return:
        """

        if len(input) == 3:
            x, coords, time = input
            lengths = None
        elif len(input) == 4:
            x, coords, time, lengths = input
        else:
            raise ValueError("input must be a 3-tuple or a 4-tuple")

        with torch.no_grad():
            if self.frame_intervals > 1:
                if self.frame_intervals_mode:
                    if self.frame_intervals_mode == 'abs_ts':
                        # Normalize each timestamp with range normalization
                        ts_max = torch.unsqueeze(time.max(dim=1)[0], -1)  # shape [batch_size, 1, 1]
                        ts_min = torch.unsqueeze(time.min(dim=1)[0], -1)  # shape [batch_size, 1, 1]
                        abs_time = (time - ts_min) / (ts_max - ts_min + 1e-8)
                    elif self.frame_intervals_mode == 'abs_ts_max':
                        ts_max = torch.unsqueeze(time.max(dim=1)[0], -1)  # shape [batch_size, 1, 1]
                        ts_max += (ts_max == 0).to(ts_max.dtype)  # prevent div by 0
                        abs_time = time / ts_max

                    x = torch.cat([x, abs_time], dim=-1)

                x, time, coords, lengths = self.intervals_to_batch(x, time, coords, lengths)

            batch_size, time_size, _ = x.shape

            # Pad coordinates by adding additional top-left space
            # Note: padding here means to translate the events
            pad_coords = coords + torch.tensor([self.pad_l, self.pad_t],
                                               dtype=coords.dtype, device=x.device).view(1, 2)
            # Group events based on their position. With this operation we select all the events in every
            # sample of the batch which are contained in each one of the receptive fields. Each one of these
            # groups is considered an independent sample (mini_sample), placed in the batch dimension and processed
            # by the LSTM independently (performs LSTM weight sharing among the different receptive fields of the
            # same image).
            # gr_*.shape = [padded_time_size_mini_samples, batch_mini_samples, features_size]
            group_ret = self.group_samples(x, pad_coords, time, lengths)
            rf_gr_batch_id, rf_gr_last_id, rf_gr_h, rf_gr_w, rf_batch_groups = group_ret

        # We process each not overlapping group separately
        rf_gr_lstm_out = []
        # FIXME hc_0 is not supported
        # TODO implement not overlapping rf splitting also for hc
        if hc_0 is not None:
            raise NotImplementedError("hc_0 argument not supported yet!")

        for batch_groups in rf_batch_groups:
            gr_lstm_out, _ = self.shared_lstm(batch_groups, hc_0)
            rf_gr_lstm_out.append(gr_lstm_out)

        ungr_dense = self.ungroup_last_dense(rf_gr_lstm_out,
                                             (rf_gr_batch_id, rf_gr_last_id, rf_gr_h, rf_gr_w),
                                             batch_size)

        if self.frame_intervals > 1:
            channel_size = ungr_dense.shape[-1]
            # from [batch_size * frame_intervals, H, W, C]
            # to [batch_size, frame_intervals, H, W, C]
            ungr_dense = ungr_dense.reshape([-1, self.frame_intervals,
                                             *self.output_shape, channel_size])
            # to [batch_size, H, W, frame_intervals, C]
            ungr_dense = ungr_dense.permute(0, 2, 3, 1, 4)
            # to [batch_size, H, W, frame_intervals * C]
            ungr_dense = ungr_dense.reshape([-1, *self.output_shape,
                                             self.frame_intervals * channel_size])

        return ungr_dense
