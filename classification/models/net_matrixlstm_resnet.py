import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from termcolor import colored
import numpy as np

from torchvision.models import resnet
from layers.EventDropout import EventDropout
from layers.SELayer import SELayer
from layers.MatrixLSTM import MatrixLSTM
from layers.MatrixConvLSTM import MatrixConvLSTM
from models.network import Network
from collections import OrderedDict


class MatrixLSTMResNet(Network):

    def __init__(self, input_shape, num_classes, embedding_size,
                 matrix_hidden_size, matrix_region_shape, matrix_region_stride,
                 matrix_add_coords_feature, matrix_add_time_feature_mode,
                 matrix_normalize_relative, matrix_lstm_type,
                 matrix_keep_most_recent=True, matrix_frame_intervals=1, matrix_frame_intervals_mode=None,
                 resnet_type="resnet18", resnet_imagenet_pretrain=False, resnet_freeze=[],
                 event_dropout=0.25, frame_dropout=-1, fc_dropout=-1, frame_actfn=None,
                 resnet_meanstd_norm=False, add_resnet_last_fc=False, lstm_num_layers=1,
                 resnet_replace_first=True, resnet_replace_first_bn=True, add_se_layer=False):
        super().__init__()

        self.frames = None

        self.height, self.width = input_shape
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.use_embedding = self.embedding_size > 0
        self.input_shape = input_shape
        self.lstm_num_layers = lstm_num_layers
        self.add_se_layer = add_se_layer

        self.resnet_freeze = resnet_freeze
        self.resnet_type = resnet_type
        self.resnet_imagenet_pretrain = resnet_imagenet_pretrain
        self.resnet_meanstd_norm = resnet_meanstd_norm
        self.resnet_replace_first = resnet_replace_first
        self.resnet_replace_first_bn = resnet_replace_first_bn
        self.add_resnet_last_fc = add_resnet_last_fc
        self.register_buffer("resnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("resnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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

        self.event_dropout = event_dropout
        self.frame_dropout = frame_dropout
        self.fc_dropout = fc_dropout
        if isinstance(frame_actfn, str):
            self.frame_actfn = getattr(torch, frame_actfn)
        else:
            self.frame_actfn = frame_actfn

        if self.event_dropout > 0:
            self.eventdrop = EventDropout(drop_prob=event_dropout)
        if self.use_embedding:
            self.embedding = nn.Embedding(self.height * self.width, embedding_size)

        matrix_input_size = self.embedding_size + 1 if self.use_embedding else 1
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

        if self.add_se_layer:
            self.se_layer = SELayer(self.matrixlstm.out_channels,
                                    reduction=1)

        if self.frame_dropout > 0:
            self.framedrop = nn.Dropout(p=self.frame_dropout)

        resnet_factory = getattr(resnet, resnet_type)
        self.resnet = resnet_factory(self.resnet_imagenet_pretrain, zero_init_residual=True)

        # First freeze resnet
        if self.resnet_freeze:
            self.freeze_resnet()

        if self.matrixlstm.out_channels != 3:
            if self.resnet_replace_first:
                # Change the first layer according to the input size
                self.resnet.conv1 = nn.Conv2d(self.matrixlstm.out_channels,
                                              64,  # inplanes
                                              kernel_size=7,
                                              stride=2,
                                              padding=3,
                                              bias=False)
                if not self.resnet_replace_first_bn:
                    self.resnet.bn1 = nn.BatchNorm2d(64)  # inplanes
            else:
                # Adds a 1x1 convolution that maps matrix_hidden_size into 3 channels
                self.project_hidden = nn.Conv2d(self.matrixlstm.out_channels,
                                                3,
                                                kernel_size=1,
                                                stride=1,
                                                bias=True)

        if not self.add_resnet_last_fc:
            # Change the last layer to produce the right number of classes
            self.resnet.fc = nn.Linear(512, num_classes)
        else:
            # Add a new layer after resnet mapping original 1000 classes into num_classes
            self.resnet = nn.Sequential(OrderedDict([('backbone', self.resnet),
                                                     ('fc', nn.Linear(1000, num_classes))]))

    def freeze_resnet(self):
        freezed_params = []
        for name, param in self.resnet.named_parameters():
            for freeze_name in self.resnet_freeze:
                if name.startswith(freeze_name):
                    freezed_params.append(name)
                    param.requires_grad = False
                    continue

        params_str = "".join(["%s, " % p for p in freezed_params])[:-len(", ")]
        print(colored("Freezed ResNet parameters:", "red"), params_str)

    def init_params(self):
        self.matrixlstm.reset_parameters()
        if self.use_embedding:
            nn.init.normal_(self.embedding.weight, mean=0, std=1)

    def coord2idx(self, x, y):
        return y * self.width + x

    def forward(self, events, lengths):

        # Events dropout during training
        if self.event_dropout > 0:
            events, lengths = self.eventdrop(events, lengths)

        # events.shape = [batch_size, time_size, 4]
        batch_size, time_size, features_size = events.shape
        assert(features_size == 4)

        x = events[:, :, 0].type(torch.int64)
        y = events[:, :, 1].type(torch.int64)
        ts = events[:, :, 2].float()
        p = events[:, :, 3].float()

        if self.use_embedding:
            # Given the event coordinates, retrieves the pixel number
            # associated to the event position
            # embed_idx.shape = [batch_size, time_size]
            embed_idx = self.coord2idx(x, y)

            # Retrieves the actual embeddings
            # [batch_size, time_size, embedding_size]
            embed = self.embedding(embed_idx)
            # Adds the polarity to each embedding
            # [batch_size, time_size, embedding_size]
            embed = torch.cat([embed, p.unsqueeze(-1)], dim=-1)
        else:
            embed = p.unsqueeze(-1)

        # [batch_size, time_size, hidden_size]
        coords = torch.stack([x, y], dim=-1)

        # out_dense.shape = [batch_size, matrix_out_h, matrix_out_w, matrix_hidden_size]
        out_dense = self.matrixlstm(input=(embed, coords, ts.unsqueeze(-1), lengths))
        out_dense = out_dense.permute(0, 3, 1, 2)

        if self.add_se_layer:
            out_dense = self.se_layer(out_dense)

        if self.matrixlstm.out_channels != 3 and not self.resnet_replace_first:
            out_dense = self.project_hidden(out_dense)

        if self.frame_actfn is not None:
            out_dense = self.frame_actfn(out_dense)

        if self.resnet_meanstd_norm:
            if self.frame_actfn is None:
                # MatrixLSTM uses a tanh as output activation
                # we use range normalization with min=-1, max=1
                img_min, img_max = -1, 1
                out_dense = (out_dense - img_min) / (img_max - img_min)
            elif self.frame_actfn is not torch.sigmoid:
                # range-normalization
                n_pixels = self.matrixlstm.output_shape[0] * self.matrixlstm.output_shape[1]
                flat_dense = out_dense.view(-1, self.matrixlstm.hidden_size, n_pixels)
                img_max = flat_dense.max(dim=-1)[0].view(-1, self.matrixlstm.hidden_size, 1, 1)
                img_min = flat_dense.min(dim=-1)[0].view(-1, self.matrixlstm.hidden_size, 1, 1)
                out_dense = (out_dense - img_min) / (img_max - img_min)
            # z-normalization
            out_dense = (out_dense - self.resnet_mean) / self.resnet_std

        if not self.training:
            self.frames = out_dense

        if self.frame_dropout > 0:
            out_dense = self.framedrop(out_dense)

        out_resnet = self.resnet(out_dense)

        # Computes log probabilities
        # log_probas.shape = [batch_size, num_classes]
        log_probas = F.log_softmax(out_resnet, dim=-1)

        return log_probas

    def loss(self, input, target):
        return F.nll_loss(input, target)

    def log_validation(self, logger, global_step):
        if not self.training:
            logframes = self.frames[:, :3]
            logframes_chans = logframes.shape[1]
            if logframes_chans == 3:
                images = vutils.make_grid(logframes, normalize=True, scale_each=True)
                logger.add_image('validation/frames', images, global_step=global_step)
