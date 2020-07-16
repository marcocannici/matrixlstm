import torch
import torch.nn as nn
import torchvision.utils as vutils

import time
import numpy as np

from layers.MatrixLSTM import MatrixLSTM
from layers.MatrixConvLSTM import MatrixConvLSTM


class Statistics:
    def __init__(self):
        self.sum_time = 0.0
        self.num_samples = 0
        self.num_events = 0

    def update(self, time, lengths):
        self.num_events += lengths.cpu().data.numpy().sum()
        self.num_samples += lengths.shape[0]
        self.sum_time += time

    def get(self):
        mean_time_ms = (self.sum_time * 1000.0) / self.num_samples
        kevents_s = self.num_events / (self.sum_time * 1000.0)
        return mean_time_ms, kevents_s, self.num_samples


class MatrixProfiler(nn.Module):

    def __init__(self, input_shape, matrix_hidden_size, matrix_region_shape, matrix_region_stride,
                 matrix_add_coords_feature, matrix_add_time_feature_mode,
                 matrix_normalize_relative, matrix_lstm_type,
                 matrix_keep_most_recent=True, matrix_frame_intervals=1,
                 matrix_frame_intervals_mode=None, lstm_num_layers=1):
        super().__init__()

        self.time_statistics = Statistics()

        self.height, self.width = input_shape
        self.input_shape = input_shape
        self.lstm_num_layers = lstm_num_layers

        self.matrix_lstm_type = matrix_lstm_type
        self.matrix_hidden_size = matrix_hidden_size
        self.matrix_region_shape = matrix_region_shape
        self.matrix_region_stride = matrix_region_stride
        self.matrix_add_coords_feature = matrix_add_coords_feature
        self.matrix_add_time_feature_mode = matrix_add_time_feature_mode
        self.matrix_normalize_relative = matrix_normalize_relative
        self.matrix_keep_most_recent = matrix_keep_most_recent
        self.matrix_frame_intervals = matrix_frame_intervals
        self.matrix_frame_intervals_mode = matrix_frame_intervals_mode

        matrix_input_size = 1
        MatrixLSTMClass = MatrixConvLSTM if self.matrix_lstm_type == "ConvLSTM" else MatrixLSTM
        self.matrixlstm = MatrixLSTMClass(self.input_shape, self.matrix_region_shape,
                                          self.matrix_region_stride, matrix_input_size,
                                          self.matrix_hidden_size, self.lstm_num_layers,
                                          bias=True, lstm_type=self.matrix_lstm_type,
                                          add_coords_feature=self.matrix_add_coords_feature,
                                          add_time_feature_mode=self.matrix_add_time_feature_mode,
                                          normalize_relative=self.matrix_normalize_relative,
                                          keep_most_recent=self.matrix_keep_most_recent,
                                          frame_intervals=self.matrix_frame_intervals,
                                          frame_intervals_mode=self.matrix_frame_intervals_mode)

    def init_params(self):
        self.matrixlstm.reset_parameters()
        if self.use_embedding:
            nn.init.normal_(self.embedding.weight, mean=0, std=1)

    def coord2idx(self, x, y):
        return y * self.width + x

    def forward(self, events, lengths, burned_in):

        # events.shape = [batch_size, time_size, 4]
        batch_size, time_size, features_size = events.shape
        assert(features_size == 4)

        x = events[:, :, 0].type(torch.int64)
        y = events[:, :, 1].type(torch.int64)
        ts = events[:, :, 2].float()
        p = events[:, :, 3].float().unsqueeze(-1)

        # [batch_size, time_size, hidden_size]
        coords = torch.stack([x, y], dim=-1)

        t_before = time.time()
        # out_dense.shape = [batch_size, matrix_out_h, matrix_out_w, matrix_hidden_size]
        out_dense = self.matrixlstm(input=(p, coords, ts.unsqueeze(-1), lengths))
        t_tot = time.time() - t_before

        if burned_in:
            self.time_statistics.update(t_tot, lengths)

        return out_dense

    def get_statistics(self):
        return self.time_statistics.get()
