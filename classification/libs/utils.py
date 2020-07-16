import torch
import numpy as np

import re
import itertools
from textwrap import wrap
import matplotlib.pyplot as plt


def padding_mask(lengths, batch_size, time_size=None):
    """
    Computes a [batch_size, time_size] binary mask which selects all and only the
    non padded values in the input tensor

    :param torch.tensor lengths: a [batch_size] tensor containing the actual length
        (before padding) of every sample in the batch
    :param int batch_size: the number of samples in the batch
    :param int time_size: the length of the padded sequences
    :retype: torch.tensors
    """

    max_len = torch.max(lengths) if time_size is None else time_size
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    mask = mask.expand(batch_size, max_len) < lengths.unsqueeze(1)

    return mask.type(torch.uint8)


def cat_arange(counts, dtype=torch.int32):
    """
    Concatenate results of multiple arange calls
    E.g.: cat_arange([2,1,3]) = [0, 1, 0, 0, 1, 2]
    Credits: https://stackoverflow.com/a/20033438

    :param torch.tensor counts: a 1D tensor
    :return: equivalent to torch.cat([torch.arange(c) for c in counts])
    """
    counts1 = counts[:-1].type(dtype)
    reset_index = torch.cumsum(counts1, dim=0).type(torch.int64)

    incr = torch.ones(counts.sum(), dtype=dtype, device=counts.device)
    incr[0] = 0
    incr[reset_index] = 1 - counts1

    # Reuse the incr array for the final result.
    return torch.cumsum(incr, dim=0)


def repeat_arange(counts, dtype=torch.int32):
    """
    Repeat each element of arange multiple times
    E.g.: repeat_arange([2,1,3]) = [0, 0, 1, 2, 2, 2]

    :param counts: a 1D tensor having the same length of 'tensor'
    :return: equivalent to torch.cat([torch.tensor([v]).expand(n) for v, n in enumerate(counts)])
    """

    incr = torch.zeros(counts.sum(), dtype=dtype, device=counts.device)
    set_index = torch.cumsum(counts[:-1], dim=0).type(torch.int64)
    incr[set_index] = 1

    return torch.cumsum(incr, dim=0)


def select_padded(source, mask):

    lengths = mask.sum(-1)
    max_length = lengths.max()
    batch_size, time_size, feature_size = source.shape

    out_tensor = source.new_zeros([batch_size, max_length, feature_size])
    batch_idx = repeat_arange(lengths, torch.int64)
    time_idx = cat_arange(lengths, torch.int64)

    out_tensor[batch_idx, time_idx] = source[mask]
    return out_tensor


def confusion_matrix_fig(cm, labels, normalize=False):

    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = ['\n'.join(wrap(l, 40)) for l in labels]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")

    return fig
