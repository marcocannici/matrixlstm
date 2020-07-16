import torch
from torch import nn
from libs import utils


class EventDropout(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, tensor, lengths=None):
        if self.training:
            batch_size, time_size, features_size = tensor.shape
            keep_mask = torch.rand(batch_size, time_size, device=tensor.device) > self.drop_prob
            if lengths is not None:
                valid_mask = utils.padding_mask(lengths, batch_size, time_size)
                keep_mask *= valid_mask
            lengths = keep_mask.sum(dim=-1)
            events = utils.select_padded(tensor, keep_mask)
            return events, lengths
        else:
            return tensor, lengths
