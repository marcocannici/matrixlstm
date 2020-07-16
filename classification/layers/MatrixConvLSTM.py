import torch
import torch.jit as jit

from layers.MatrixLSTM import MatrixLSTM
from libs.lstms.convlstm import ConvLSTM


class MatrixConvLSTM(MatrixLSTM):

    def __init__(self, input_shape, region_shape, region_stride, input_size, hidden_size, num_layers=1,
                 bias=True, lstm_type="ConvLSTM", add_coords_feature=True, add_time_feature_mode='delay',
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

        if maintain_in_shape is False:
            raise ValueError("maintain_in_shape = False not supported!")

        self.add_coord_features = add_coords_feature
        self.add_time_feature_mode = add_time_feature_mode
        self.normalize_relative = normalize_relative
        self.register_buffer('max_events_per_rf', torch.tensor(max_events_per_rf))
        self.maintain_in_shape = maintain_in_shape
        self.keep_most_recent = keep_most_recent
        self.frame_intervals = frame_intervals
        self.frame_intervals_mode = frame_intervals_mode

        self.output_shape = input_shape
        self.out_channels = self.hidden_size * self.frame_intervals

        assert maintain_in_shape is False or (self.output_shape[0] == self.input_shape[0] and
                                              self.output_shape[1] == self.input_shape[1])

        if self.lstm_type != "ConvLSTM":
            raise ValueError("LSTM type %s not supported!", self.lstm_type)

        self.shared_lstm = ConvLSTM(input_dim=self.input_size, hidden_dim=self.hidden_size,
                                    Wi_kernel=self.region_shape, Wh_kernel=self.region_shape,
                                    stride=self.region_stride, dropout_p=0, forget_bias=1.0,
                                    use_pool=False, use_bias=True, use_layer_norm=False,
                                    num_layers=self.num_layers, batch_first=False)
        # self.shared_lstm = jit.script(self.shared_lstm)

        self.overlapping_rf = self.region_stride[0] < self.region_shape[0] or \
                              self.region_stride[1] < self.region_shape[1]

        self.num_rf = self.output_shape[0] * self.output_shape[1]

        self.num_not_overlap_groups = 1
        self.output_shape_group0 = self.output_shape
        self.rfid_rel2abs_group0 = torch.arange(self.num_rf)
        # Note: here we "trick" the grouping operation to think that the layer has a 1x1 receptive field,
        #       so that it will only reconstruct a single densified group of receptive fields. The
        #       convolution is then implemented within the ConvLSTM
        self.register_buffer('pixel2rf_idx_group0', self.compute_pixels2rf_matrix(self.input_shape,
                                                                                  region_shape=(1, 1),
                                                                                  region_stride=(1, 1),
                                                                                  num_rf=self.num_rf))
        self.register_buffer('rf_topleft_group0', self.compute_topleft_matrix(self.input_shape,
                                                                              region_shape=(1, 1),
                                                                              region_stride=(1, 1)))

    def ungroup_full_dense(self, rf_input, rf_idx, batch_size):

        # input.shape = padded_event_size, flat_batch_size, feature_size
        padded_event_size, flat_batch_size, feature_size = rf_input.shape
        # [batch_size, Tmax, C, H, W]
        dense_out = torch.zeros([batch_size, padded_event_size, feature_size, *self.output_shape],
                                device=rf_input[0].device)

        gr_batch_id, gr_last_id, gr_h, gr_w = rf_idx
        dense_out[gr_batch_id, :, :, gr_h, gr_w] = rf_input.permute(1, 0, 2)
        # dense_out[gr_batch_id, :, :, gr_h, gr_w] = rf_input[:, torch.arange(flat_batch_size)].permute(1, 0, 2)

        return dense_out

    def ungroup_last_dense(self, rf_input, rf_idx, batch_size):

        # rf_input.shape = [batch_size, Tmax, C, H, W]
        _, _, feature_size, _, _ = rf_input.shape
        dense_out = torch.zeros([batch_size, *self.output_shape, feature_size],
                                device=rf_input[0].device)

        gr_batch_id, gr_last_id, gr_h, gr_w = rf_idx
        dense_out[gr_batch_id, gr_h, gr_w] = rf_input[gr_batch_id, gr_last_id, :, gr_h, gr_w]

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

            # Group events based on their position. With this operation we select all the events in every
            # sample of the batch which are contained in each one of the receptive fields. Each one of these
            # groups is considered an independent sample (mini_sample), placed in the batch dimension and processed
            # by the LSTM independently (performs LSTM weight sharing among the different receptive fields of the
            # same image).
            # gr_*.shape = [padded_time_size_mini_samples, batch_mini_samples, features_size]
            group_ret = self.group_samples(x, coords, time, lengths)

            # This variant extract a single "non-overlapping" group
            # We remove list encapsulation (each list has a single element)
            group_ret = [retval[0] for retval in group_ret]
            rf_idx, rf_batch_groups = group_ret[:-1], group_ret[-1]

            # Construct a dense frame [batch_size, Tmax, C, H, W]
            x_dense = self.ungroup_full_dense(rf_batch_groups, rf_idx, batch_size)

        # FIXME hc_0 is not supported
        # TODO implement not overlapping rf splitting also for hc
        if hc_0 is not None:
            raise NotImplementedError("hc_0 argument not supported yet!")

        lstm_dense_out, _ = self.shared_lstm(x_dense, hc_0)
        ungr_dense = self.ungroup_last_dense(lstm_dense_out, rf_idx, batch_size)

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
