import os
import glob
import torch
import numpy as np
from termcolor import colored

from .eventdataset import EventDataset, EventDetectionDataset
from .eventchunkdataset import EventChunkDataset, EventDetectionChunkDataset
from .filereader import (BinFileReader, AerFileReader, PropheseeReader,
                         NumpyFileReader, MatFileReader)

from collections import namedtuple
chunkparams = namedtuple('chunkparams', ['delta_t', 'min_delta_t', 'min_n_events'])


def get_file_format(files, reader_type='classification'):

    # Takes the first file in the list as a sample to infer file extension
    # TODO: implement a more robust method to infer file format

    sample_file = None
    if isinstance(files, list):
        sample_file = files[0]
    elif isinstance(files, str):
        if os.path.isdir(files):
            if reader_type == 'classification':
                sample_file = glob.glob(os.path.join(files, "*/*.*"))[0]
            else:
                sample_file = glob.glob(os.path.join(files, "*.*"))
                sample_file = [fn for fn in sample_file if not fn.endswith(".txt")][0]
        elif os.path.isfile(files):
            with open(files, 'r') as fp:
                # chunk files are "path start_idx count", we take the path of the first row
                sample_file = fp.read().splitlines()[0].split(" ")[0]
        else:
            raise ValueError("'files' must either be a list or a path to an "
                             "existing file or directory")

    ext = os.path.splitext(sample_file)[1]
    return ext


def get_filereader(file_ext, format=None):
    if file_ext == ".aedat":
        return AerFileReader("DVS128")
    elif file_ext == ".bin":
        return BinFileReader(format)
    elif file_ext == ".dat":
        return PropheseeReader(format)
    elif file_ext == '.mat':
        return MatFileReader(format)
    elif file_ext == ".npy":
        return NumpyFileReader(format)
    else:
        raise ValueError("File extension '%s' is unknown!" % file_ext)


def collate_fn(batch):

    events, labels, lengths = [], [], []
    for ev, lbl in batch:
        events.append(torch.tensor(ev))
        labels.append(lbl)
        lengths.append(ev.shape[0])

    labels = torch.tensor(labels)

    return lengths, events, labels


def collate_pad_fn(batch):

    events, labels, lengths = [], [], []
    for ev, lbl in batch:
        events.append(ev)
        labels.append(lbl)
        lengths.append(ev.shape[0])

    max_length = max(lengths)
    events = [np.pad(ev, ((0, max_length-ln), (0, 0)), mode='constant') for ln, ev in zip(lengths, events)]
    events = torch.tensor(np.stack(events, axis=0))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    return lengths, events, labels


def collate_detection_fn(batch):
    events, anns, lengths = [], [], []

    for ev, ann in batch:
        events.append(ev)
        anns.append(ann)
        lengths.append(ev.shape[0])

    max_length = max(lengths)
    events = [np.pad(ev, ((0, max_length - ln), (0, 0)), mode='constant') for ln, ev in zip(lengths, events)]
    events = torch.tensor(np.stack(events, axis=0))
    anns = [torch.tensor(ann) for ann in anns]
    lengths = torch.tensor(lengths)

    return lengths, events, anns


def get_dataset(files_source, reader_type, transform=None, mixed_transform=None,
                format=None, usechunks=False, chunk_params=None):

    file_ext = get_file_format(files_source, reader_type)
    filereader = get_filereader(file_ext, format)

    if not usechunks:
        if reader_type == 'classification':
            dataset = EventDataset(filereader, files_source=files_source,
                                   transforms=transform, format=format)
        elif reader_type == 'detection':
            dataset = EventDetectionDataset(filereader, files_source=files_source, transforms=transform,
                                            mixed_transforms=mixed_transform, format=format)
        else:
            raise ValueError("Unknown reader type %s" % reader_type)
    else:
        if reader_type == 'classification':
            dataset = EventChunkDataset(filereader, files_source=files_source,
                                        transforms=transform, format=format,
                                        chunk_params=chunk_params)
        elif reader_type == 'detection':
            dataset = EventDetectionChunkDataset(filereader, files_source=files_source, transforms=transform,
                                                 mixed_transforms=mixed_transform, format=format,
                                                 chunk_params=chunk_params)
        else:
            raise ValueError("Unknown reader type %s" % reader_type)

    return dataset


def get_loader(files_source, batch_size, shuffle, num_workers=0, transform=None, mixed_transform=None,
               format=None, reader_type='classification', pad=False, usechunks=False, chunk_params=None, seed=42):

    torch.manual_seed(seed)

    if reader_type == 'classification':
        collate = collate_pad_fn if pad else collate_fn
    elif reader_type == 'detection':
        collate = collate_detection_fn
    else:
        raise ValueError("Unknown reader type %s" % reader_type)

    dataset = get_dataset(files_source, reader_type, transform,
                          mixed_transform, format, usechunks, chunk_params)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
    return loader


def get_dir_or_chunkfile(data_dir, usechunks, split):
    if usechunks[split] is True:
        chunk_file = os.path.join(data_dir, split + "_chunks.txt")
        if os.path.exists(chunk_file):
            return chunk_file
        else:
            return os.path.join(data_dir, split)
    else:
        split_dir = os.path.join(data_dir, split)
        split_file = os.path.join(data_dir, split + ".txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as fp:
                file_list = fp.read().splitlines()
            file_list = [os.path.join(data_dir, p) for p in file_list]
            return file_list
        else:
            return split_dir


def random_split_source(source, file_ext, perc):
    if not isinstance(source, list) and os.path.isdir(source):
        source = glob.glob(os.path.join(source, "*/*" + file_ext))
    elif not isinstance(source, list):
        raise ValueError("'source' must me either a list of files or a path to a directory")
    assert sum(perc) == 1

    tot_files = len(source)
    lengths = [int(p * tot_files) for p in perc[:-1]]
    lengths += [tot_files - sum(lengths)]

    assert sum(lengths) == tot_files
    # create a np.array to perform fancy indexing
    source = np.array(source)
    indices = torch.randperm(tot_files).tolist()
    return [source[indices[offset - length:offset]].tolist() for offset, length in zip(np.cumsum(lengths), lengths)]


def format_usechunks(usechunks):
    if isinstance(usechunks, int) or (isinstance(usechunks, list) and len(usechunks) == 1):
        us = usechunks if isinstance(usechunks, int) else usechunks[0]
        usechunks = {k: us for k in ['train', 'validation', 'test']}
    elif isinstance(usechunks, list):
        usechunks = {k: us for us, k in zip(usechunks, ['train', 'validation', 'test'])}
    else:
        raise ValueError("'usechunks' must be either an int or a list of either one of three elements")

    return usechunks


def format_batch_size(batch_size):
    if isinstance(batch_size, int) or (isinstance(batch_size, list) and len(batch_size) == 1):
        bs = batch_size if isinstance(batch_size, int) else batch_size[0]
        batch_size = {k: bs for k in ['train', 'validation', 'test']}
    elif isinstance(batch_size, list):
        batch_size = {k: bs for bs, k in zip(batch_size, ['train', 'validation', 'test'])}
    else:
        raise ValueError("'batch_size' must be either an int or a list of either one of three elements")

    return batch_size


def format_transform(transform):
    if not isinstance(transform, dict):
        transform = {k: transform for k in ['train', 'validation', 'test']}

    return transform


def get_splits(data_dir, batch_size, num_workers=0, transform=None, mixed_transforms=None, format=None,
               pad=False, usechunks=False, chunks_delta_t=5e4, min_chunk_delta_t=2.5e4, min_chunk_n_events=150,
               val_split=0.2, reader_type='classification', seed=42):

    chunk_params = chunkparams(chunks_delta_t, min_chunk_delta_t, min_chunk_n_events)
    np.random.seed(seed)
    torch.manual_seed(seed)

    usechunks = format_usechunks(usechunks)
    batch_size = format_batch_size(batch_size)
    transform = format_transform(transform)

    if reader_type not in ['classification', 'detection']:
        raise ValueError("Supported types are 'classification' and 'detection'. Provided: %s" % reader_type)

    train_source = get_dir_or_chunkfile(data_dir, usechunks, "train")
    val_source = get_dir_or_chunkfile(data_dir, usechunks, "validation")
    test_source = get_dir_or_chunkfile(data_dir, usechunks, "test")

    if (not isinstance(train_source, list)) and (not os.path.exists(train_source)):
        raise ValueError("Training source %s does not exist" % train_source)
    if (not isinstance(test_source, list)) and (not os.path.exists(test_source)):
        raise ValueError("Test source %s does not exist" % test_source)

    if isinstance(val_source, list) and val_source or os.path.exists(val_source):
        train_loader = get_loader(files_source=train_source, format=format, reader_type=reader_type,
                                  batch_size=batch_size['train'], shuffle=True, transform=transform['train'],
                                  mixed_transform=mixed_transforms, num_workers=num_workers,
                                  pad=pad, usechunks=usechunks['train'], chunk_params=chunk_params, seed=seed)
        val_loader = get_loader(files_source=val_source, format=format, reader_type=reader_type,
                                batch_size=batch_size['validation'], shuffle=False, transform=transform['validation'],
                                mixed_transform=mixed_transforms, num_workers=num_workers,
                                pad=pad, usechunks=usechunks['validation'], chunk_params=chunk_params, seed=seed)
    else:
        train_source = get_dir_or_chunkfile(data_dir, {'train': False}, "train")
        file_ext = get_file_format(train_source, reader_type)
        train_files, val_files = random_split_source(train_source, file_ext, [1-val_split, val_split])
        assert len(set(val_files).intersection(train_files)) == 0

        print(colored("Warning: validation folder does not exist!! "
              "Splitting the training set using {}% of files from the training set. "
              "Training set size: {} (originally: {}), Validation set size: {}"
              "".format(val_split * 100, len(train_files), len(train_files)+len(val_files), len(val_files)), 'red'))

        train_loader = get_loader(files_source=train_files, format=format, reader_type=reader_type,
                                  batch_size=batch_size['train'], shuffle=True, num_workers=num_workers,
                                  transform=transform['train'], mixed_transform=mixed_transforms, pad=pad,
                                  usechunks=usechunks['train'], chunk_params=chunk_params, seed=seed)
        if val_split > 0:
            val_loader = get_loader(files_source=val_files, format=format, reader_type=reader_type,
                                    batch_size=batch_size['validation'], shuffle=False, num_workers=num_workers,
                                    transform=transform['validation'], mixed_transform=mixed_transforms, pad=pad,
                                    usechunks=usechunks['validation'], chunk_params=chunk_params, seed=seed)
        else:
            val_loader = None

    test_loader = get_loader(files_source=test_source, format=format, reader_type=reader_type,
                             batch_size=batch_size['test'], shuffle=False, transform=transform['test'],
                             mixed_transform=mixed_transforms, num_workers=num_workers,
                             pad=pad, usechunks=usechunks['test'], chunk_params=chunk_params, seed=seed)

    if reader_type == "classification":
        assert train_loader.dataset.num_classes == \
               test_loader.dataset.num_classes, \
            "The number of classes is different among the splits"
        assert train_loader.dataset.name_to_id == \
               test_loader.dataset.name_to_id, \
            "Label mapping is different among the splits"

        if val_loader:
            assert train_loader.dataset.num_classes == \
                   val_loader.dataset.num_classes, \
            "The number of classes is different among the splits"
            assert train_loader.dataset.name_to_id == \
                   val_loader.dataset.name_to_id
            "Label mapping is different among the splits"

    return train_loader, val_loader, test_loader
