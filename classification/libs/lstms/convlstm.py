import math

import torch
import torch.nn as nn
from torch.nn import init

from typing import List, Tuple, Optional


class LayerNorm(nn.Module):

    """
    Implementation of layer normalization, slightly modified from
    https://github.com/pytorch/pytorch/issues/1959.
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.gamma.data, val=1)
        init.constant_(self.beta.data, val=0)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.gamma*(input - mean)/(std + self.eps) + self.beta


class ParallelLayerNorm(nn.Module):

    """
    Faster parallel layer normalization.
    Inspired by the implementation of
    https://github.com/hardmaru/supercell/blob/master/supercell.py.
    """

    def __init__(self, num_inputs, num_features, eps=1e-6):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.gamma.data, val=1)
        init.constant_(self.beta.data, val=0)

    def forward(self, *inputs):
        # type: (List[Tensor]) -> List[Tensor]
        """
        Args:
            input_1, ... (Variable): Variables to which
                layer normalization be applied. The number of inputs
                must be identical to self.num_inputs.
        """

        inputs_stacked = torch.stack(inputs, dim=-2)
        mean = inputs_stacked.mean(dim=-1, keepdim=True)
        std = inputs_stacked.std(dim=-1, keepdim=True)
        outputs_stacked = (self.gamma*(inputs_stacked - mean) /
                           (std + self.eps) + self.beta)
        outputs = torch.unbind(outputs_stacked, dim=-2)
        return outputs


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 Wi_kernel, Wh_kernel=None,
                 stride=1, use_pool=False, use_bias=True,
                 use_layer_norm=False, dropout_p=0,
                 forget_bias=1.0):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        Wi_kernel: (int, int)
            Size of the kernel of input-to-hidden convolution.
        Wh_kernel: (int, int)
            Size of the kernel of hidden-to-hidden convolution.
        stride: int
            Stride of the input convolution.
        use_pool: bool
            Whether or not to use Max pooling.
        use_bias: bool
            Whether or not to add the bias.
        forget_bias: bool
            Force the model to learn at the beginning imposing a
            forgetting

        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_height = -1
        self.out_width = -1

        self.Wi_kernel = Wi_kernel
        self.Wi_padding = (self.Wi_kernel - 1) // 2, (self.Wi_kernel - 1) // 2
        self.Wh_kernel = Wh_kernel if Wh_kernel is not None else Wi_kernel
        self.Wh_padding = (self.Wh_kernel - 1) // 2, (self.Wh_kernel - 1) // 2

        self.stride = stride
        self.use_pool = use_pool
        self.use_bias = use_bias
        self.forget_bias = forget_bias

        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm

        stride_in = self.stride if not self.use_pool else 1

        # NOTE: ih and hh conv are splitted because we might want to
        # use different kernel sizes or strides
        self.ih_conv = nn.Conv2d(in_channels=self.input_dim,
                                 out_channels=4*self.hidden_dim,
                                 kernel_size=self.Wi_kernel,
                                 padding=self.Wi_padding,
                                 stride=stride_in,
                                 bias=self.use_bias)

        self.hh_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                 out_channels=4*self.hidden_dim,
                                 kernel_size=self.Wh_kernel,
                                 padding=self.Wh_padding,
                                 stride=1,
                                 bias=False)

        self.pool: Optional[nn.Module] = None
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.ln_ifoj: Optional[nn.Module] = None
        self.ln_c: Optional[nn.Module] = None
        if self.use_layer_norm:
            self.ln_ifoj = ParallelLayerNorm(num_inputs=4,
                                             num_features=self.hidden_dim)
            self.ln_c = LayerNorm(hidden_dim)

        self.dropout: Optional[nn.Module] = None
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()

    def get_output_size(self, x):
        # type: (Tensor) -> Tuple[int, int]
        in_height, in_width = x.data.size()[2:]
        self.out_height = int(math.ceil(float(in_height) / self.stride))
        self.out_width = int(math.ceil(float(in_width) / self.stride))
        return self.out_height, self.out_width

    def forward(self, x, cur_state):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        if cur_state is None:
            cur_state = self.init_hidden(x)

        h, c = cur_state
        x = self.ih_conv(x)
        h = self.hh_conv(h)
        out = x + h

        # i: input_gate, j: new_input, f: forget_gate, o: output_gate
        i, f, o, j = torch.split(out, self.hidden_dim, dim=1)

        if self.ln_ifoj is not None:
            i, f, o, j = self.ln_ifoj(i, f, o, j)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self.forget_bias)
        o = torch.sigmoid(o)
        j = torch.tanh(j)

        if self.dropout is not None:
            j = self.dropout(j)

        c_next = f * c + i * j
        if self.ln_c is not None:
            c_next = self.ln_c(c_next)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, x):
        # type: (Tensor) -> Tuple[Tensor, Tensor]
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = self.get_output_size(x)
        state_size = [batch_size, self.hidden_dim] + list(spatial_size)
        return x.new_zeros(state_size), x.new_zeros(state_size)

    def reset_parameters(self):
        init.xavier_uniform_(self.ih_conv.weight)
        init.constant_(self.ih_conv.bias, val=0.0)
        hidden_dim = self.hidden_dim
        init.orthogonal_(self.hh_conv.weight[:hidden_dim, ...])
        init.orthogonal_(self.hh_conv.weight[hidden_dim:2*hidden_dim, ...])
        init.orthogonal_(self.hh_conv.weight[2*hidden_dim:3*hidden_dim, ...])
        init.orthogonal_(self.hh_conv.weight[3*hidden_dim:, ...])
        if self.ln_ifoj is not None and self.ln_c is not None:
            self.ln_ifoj.reset_parameters()
            self.ln_c.reset_parameters()


class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 Wi_kernel, Wh_kernel=None,
                 stride=1, use_pool=False, use_bias=True,
                 use_layer_norm=False, dropout_p=0):
        """
        Initialize ConvGRUcell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        Wi_kernel: (int, int)
            Size of the kernel of input-to-hidden convolution.
        Wh_kernel: (int, int)
            Size of the kernel of hidden-to-hidden convolution.
        stride: int
            Stride of the input convolution.
        use_pool: bool
            Whether or not to use Max pooling.
        use_bias: bool
            Whether or not to add the bias.

        """

        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_height = -1
        self.out_width = -1

        self.Wi_kernel = Wi_kernel
        self.Wi_padding = (self.Wi_kernel - 1) // 2, (self.Wi_kernel - 1) // 2
        self.Wh_kernel = Wh_kernel if Wh_kernel is not None else Wi_kernel
        self.Wh_padding = (self.Wh_kernel - 1) // 2, (self.Wh_kernel - 1) // 2

        self.stride = stride
        self.use_pool = use_pool
        self.use_bias = use_bias

        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm

        stride_in = self.stride if not self.use_pool else 1

        # NOTE: ih and hh conv are splitted because we might want to
        # use different kernel sizes or strides
        self.ih_conv_gates = nn.Conv2d(in_channels=self.input_dim,
                                       out_channels=2*self.hidden_dim,
                                       kernel_size=self.Wi_kernel,
                                       padding=self.Wi_padding,
                                       stride=stride_in,
                                       bias=self.use_bias)
        self.hh_conv_gates = nn.Conv2d(in_channels=self.hidden_dim,
                                       out_channels=2*self.hidden_dim,
                                       kernel_size=self.Wh_kernel,
                                       padding=self.Wh_padding,
                                       stride=1,
                                       bias=False)
        self.ih_conv_ct = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=self.Wi_kernel,
                                    padding=self.Wi_padding,
                                    stride=stride_in,
                                    bias=self.use_bias)
        self.hh_conv_ct = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=self.Wh_kernel,
                                    padding=self.Wh_padding,
                                    stride=stride_in,
                                    bias=False)

        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        if use_layer_norm:
            self.ln_ur = ParallelLayerNorm(num_inputs=2,
                                           num_features=self.hidden_dim)
            self.ln_c = LayerNorm(hidden_dim)
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()

    def get_output_size(self, x):
        #type (Tensor) -> Tuple[int, int]
        in_height, in_width = x.data.size()[2:]
        self.out_height = int(math.ceil(float(in_height) / self.stride))
        self.out_width = int(math.ceil(float(in_width) / self.stride))
        return self.out_height, self.out_width

    def forward(self, x, cur_state):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if cur_state is None:
            cur_state = self.init_hidden(x)

        x = self.ih_conv_gates(x)
        h = self.hh_conv_gates(cur_state)
        out = x + h

        reset, update = out.chunk(2, 1)
        if self.use_layer_norm:
            update, reset = self.ln_ur(update, reset)

        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)

        # Compute candidate
        # W_{xc}*x + (r*W_{hc})*h
        c = self.ih_conv_ct(x) + self.hh_conv_c(torch.mul(reset, h))
        if self.use_layer_norm:
            c = self.ln_c(c)
        c = torch.tanh(c)

        h_next = torch.mul(update, h) + (1 - update)*c

        return h_next, h_next

    def init_hidden(self, x):
        # type: (Tensor) -> Tensor
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = self.get_output_size(x)
        state_size = [batch_size, self.hidden_dim] + list(spatial_size)
        return x.new_zeros(state_size)

    def reset_parameters(self):
        init.xavier_uniform_(self.ih_conv_gates.weight)
        init.constant_(self.ih_conv.bias, val=0)
        init.xavier_uniform_(self.ih_conv_ct.weight)
        init.constant_(self.ih_conv_ct.bias, val=0)

        hidden_dim = self.hidden_dim
        init.orthogonal_(self.hh_conv_gates.weight[:hidden_dim, ...])
        init.orthogonal_(self.hh_conv_gates.weight[hidden_dim:2*hidden_dim, ...])
        init.orthogonal_(self.hh_conv_ct.weight)

        init.orthogonal_(self.hh_conv_gates.weight[:hidden_dim, ...])
        init.orthogonal_(self.hh_conv_gates.weight[hidden_dim:2*hidden_dim, ...])
        init.orthogonal_(self.hh_conv_ct.weight)

        if self.use_layer_norm:
            self.ln_ur.reset_parameters()
            self.ln_c.reset_parameters()


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 Wi_kernel, Wh_kernel=None, num_layers=1,
                 batch_first=True, stride=1, use_pool=False,
                 use_bias=True, use_layer_norm=False,
                 dropout_p=0, forget_bias=1.0):

        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.Wi_kernel = self._extend_for_multilayer(Wi_kernel, num_layers)
        self.Wh_kernel = self._extend_for_multilayer(Wh_kernel, num_layers)
        self.stride = self._extend_for_multilayer(stride, num_layers)
        self.use_pool = self._extend_for_multilayer(use_pool, num_layers)
        self.use_bias = self._extend_for_multilayer(use_bias, num_layers)
        self.use_layer_norm = self._extend_for_multilayer(use_layer_norm, num_layers)
        self.dropout_p = self._extend_for_multilayer(dropout_p, num_layers)
        self.forget_bias = self._extend_for_multilayer(forget_bias, num_layers)

        cell_list = []
        for i in range(self.num_layers):
            cell_list.append(ConvLSTMCell(input_dim=self.input_dim if i == 0 else self.hidden_dim[i-1],
                                          hidden_dim=self.hidden_dim[i],
                                          Wi_kernel=self.Wi_kernel[i],
                                          Wh_kernel=self.Wh_kernel[i],
                                          stride=self.stride[i],
                                          use_pool=self.use_pool[i],
                                          use_bias=self.use_bias[i],
                                          use_layer_norm=self.use_layer_norm[i],
                                          dropout_p=self.dropout_p[i],
                                          forget_bias=self.forget_bias[i]))

        self.layers = nn.ModuleList(cell_list)
        self.reset_parameters()

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def reset_parameters(self):
        for cell in self.layers:
            cell.reset_parameters()

    def init_hidden(self, input):
        init_states: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            init_states.append(layer.init_hidden(input[0]))
        return init_states

    def forward(self, x, hx):
        # type: (Tensor, Optional[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        if self.batch_first:
            # (b, t, c, h, w) -> (t, b, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)

        time_size = x.shape[0]

        if hx is None:
            hx = self.init_hidden(x)

        last_h_states, last_c_states = [], []
        cur_layer_input = x

        layer_idx = 0
        for layer in self.layers:

            h, c = hx[layer_idx]
            output_inner = []
            for t in range(time_size):
                h, c = layer(x=cur_layer_input[t],
                             cur_state=(h, c))
                output_inner.append(h)

            dim = 1 if self.batch_first else 0
            cur_layer_input = torch.stack(output_inner, dim=dim)

            last_h_states.append(h)
            last_c_states.append(c)
            layer_idx += 1

        last_h_states = torch.stack(last_h_states, dim=0)
        last_c_states = torch.stack(last_c_states, dim=0)

        return cur_layer_input, (last_h_states, last_c_states)
