import os
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class EventChunkDataset(Dataset):

    def __init__(self, filereader, files_source, format=None, transforms=None, chunk_params=None):
        super().__init__()

        self.filereader = filereader
        self.format = format
        self.transforms = transforms

        if isinstance(files_source, str) and os.path.isfile(files_source):
            self.chunks = self.parse_chunk_file(files_source)
        elif isinstance(files_source, str) and os.path.isdir(files_source):
            files = glob.glob(os.path.join(files_source, "*/*" + self.filereader.ext))
            self.chunks = self._chunk_files(files, chunk_params)
        elif isinstance(files_source, list):
            if os.path.isfile(files_source[0]):
                self.chunks = self._chunk_files(files_source, chunk_params)
            elif isinstance(files_source[0], tuple):
                self.chunks = files_source
            else:
                raise ValueError("If a list is provided for the 'files_source' it must contain either a list "
                                 "of files or a list of tuples")
        else:
            raise ValueError("The 'files_source' argument must be either a list or a path to a txt file!")

        self.classes = sorted(np.unique([os.path.basename(os.path.dirname(f)) for f, _, _ in self.chunks]))
        self.name_to_id = {cls: i for i, cls in enumerate(self.classes)}

    @property
    def num_classes(self):
        return len(self.classes)

    @staticmethod
    def gen_chunk_indices(events_ts, delta_t):

        start_ts = events_ts[0]
        last_ts = events_ts[-1]
        chunks = []

        # start_t <= t < start_ts + delta_t
        while start_ts < last_ts:
            events_inside = np.logical_and(start_ts <= events_ts, events_ts < start_ts + delta_t)
            start_idx = np.argmax(events_inside)
            end_idx = events_inside.size - np.argmax(events_inside[::-1]) - 1
            count = end_idx - start_idx + 1

            # argmax returns 0 if the array only contains False values
            # we check if there is at least one element inside the delta_t interval
            if bool(events_inside[start_idx]) is True:
                chunks.append((start_idx, count))

            start_ts = start_ts + delta_t

        return chunks

    @staticmethod
    def filter_chunks(chunks, ts, min_delta_t, min_n_events):

        fchunks = []
        prev_idx, _ = chunks[-1]
        for start_idx, count in chunks:
            ok_min_delta_t = min_delta_t is None or \
                             ts[start_idx + count - 1] - ts[start_idx] >= min_delta_t
            ok_min_n_events = min_n_events is None or count >= min_n_events

            if ok_min_delta_t and ok_min_n_events:
                fchunks.append((start_idx, count))

        return fchunks

    @staticmethod
    def parse_chunk_file(filename):
        with open(filename, "r") as f:
            lines = f.read().splitlines()

        chunks = []
        for line in lines:
            values = line.split(" ")
            chunks.append((values[0], int(values[1]), int(values[2])))
        return chunks

    def _chunk_files(self, files_list, params):
        out_chunks = []
        for file in tqdm(files_list, desc="Computing chunk indices"):
            _, _, _, ts, _ = self.filereader.read_example(file)
            chunks = self.gen_chunk_indices(ts, params.delta_t)
            chunks = self.filter_chunks(chunks, ts, params.min_delta_t, params.min_n_events)
            out_chunks += [(file, idx, count) for idx, count in chunks]
        return out_chunks

    def _path_to_class(self, path):
        cls_name = os.path.basename(os.path.dirname(path))
        return self.name_to_id[cls_name]

    def read_example(self, filename, start=0, count=-1):
        return self.filereader.read_example(filename, start=start, count=count)

    def __getitem__(self, index):
        path, start, count = self.chunks[index]
        l, x, y, ts, p = self.read_example(path, start=start, count=count)
        lbl = self._path_to_class(path)
        events = np.column_stack([x, y, ts, p])

        if self.transforms is not None:
            events = self.transforms(events)

        return events, lbl

    def __len__(self):
        return len(self.chunks)


class EventDetectionChunkDataset(EventChunkDataset):

    def __init__(self, filereader, files_source, format=None,
                 transforms=None, mixed_transforms=None, chunk_params=None):
        super(EventChunkDataset, self).__init__()

        self.filereader = filereader
        self.format = format
        self.transforms = transforms
        self.mixed_transforms = mixed_transforms

        if isinstance(files_source, str) and os.path.isfile(files_source):
            self.chunks = self.parse_chunk_file(files_source)
        elif isinstance(files_source, str) and os.path.isdir(files_source):
            # Detection datasets are not organized in directories based on the class
            # all files and annotations are on the same directory
            files = glob.glob(os.path.join(files_source, "*" + self.filereader.ext))
            self.chunks = self._chunk_files(files, chunk_params)
        elif isinstance(files_source, list):
            if os.path.isfile(files_source[0]):
                self.chunks = self._chunk_files(files_source, chunk_params)
            elif isinstance(files_source[0], tuple):
                self.chunks = files_source
            else:
                raise ValueError("If a list is provided for the 'files_source' it must contain either a list "
                                 "of files or a list of tuples")
        else:
            raise ValueError("The 'files_source' argument must be either a list or a path to a txt file!")

    def read_annotation(self, filename, ts_start=None, ts_end=None):
        return self.filereader.read_annotation(filename, ts_start=ts_start, ts_end=ts_end)

    def _chunk_files(self, files_list, params):
        out_chunks = []
        for file in tqdm(files_list, desc="Computing chunk indices"):
            _, _, _, ts, _ = self.filereader.read_example(file)
            ann = self.filereader.read_annotation(self.filereader.get_ann_path(file))
            ann_ts = ann[:, 4]
            chunks = self.gen_chunk_indices(ts, ann_ts, params.delta_t)
            chunks = self.filter_chunks(chunks, ts, params.min_delta_t, params.min_n_events)
            out_chunks += [(file, idx, count) for idx, count in chunks]
        return out_chunks

    @staticmethod
    def _get_prev_exp(n):
        exp = 0
        while 2 ** (exp + 32) - 1 < n:
            exp += 32
        return exp

    @staticmethod
    def parse_chunk_file(filename):
        with open(filename, "r") as f:
            lines = f.read().splitlines()

        chunks = []
        for line in lines:
            values = line.split(" ")
            chunk = [values[i] if i == 0 else int(values[i])
                     for i, v in enumerate(values)]
            chunks.append(tuple(chunk))
        return chunks

    @staticmethod
    def gen_chunk_indices(events_ts, bboxes_ts, delta_t):

        unique_bboxes_ts = np.unique(bboxes_ts)
        chunks = []

        if len(unique_bboxes_ts) > 1:
            start_ts = -1
            for end_ts in unique_bboxes_ts[1:]:
                # Limits each chunk to be at most delta_t len
                if end_ts - start_ts > delta_t:
                    start_ts = end_ts - delta_t
                events_inside = np.logical_and(events_ts > start_ts, events_ts <= end_ts)
                start_idx = np.argmax(events_inside)
                end_idx = events_inside.size - np.argmax(events_inside[::-1]) - 1
                count = end_idx - start_idx + 1

                # argmax returns 0 if the array only contains False values
                # we check if there is at least one element inside the delta_t interval
                if bool(events_inside[start_idx]) is True:
                    exp_base = EventDetectionChunkDataset._get_prev_exp(events_ts[start_idx])
                    chunks.append((start_idx, count, end_ts, end_ts, exp_base))

                start_ts = end_ts
        else:
            start_ts = events_ts[0] - 1
            end_ts = unique_bboxes_ts[0]
            # Limits each chunk to be at most delta_t len
            if end_ts - start_ts > delta_t:
                start_ts = end_ts - delta_t

            events_inside = np.logical_and(events_ts > start_ts, events_ts <= end_ts)
            start_idx = np.argmax(events_inside)
            end_idx = events_inside.size - np.argmax(events_inside[::-1]) - 1
            count = end_idx - start_idx + 1
            exp_base = EventDetectionChunkDataset._get_prev_exp(events_ts[start_idx])
            chunks = [(start_idx, count, end_ts, end_ts, exp_base)]

        return chunks

    @staticmethod
    def filter_chunks(chunks, ts, min_delta_t, min_n_events):

        fchunks = []
        prev_idx, _ = chunks[-1]
        for start_idx, count, end_ts, end_ts, exp_base in chunks:
            ok_min_delta_t = min_delta_t is None or \
                             ts[start_idx + count - 1] - ts[start_idx] >= min_delta_t
            ok_min_n_events = min_n_events is None or count >= min_n_events

            if ok_min_delta_t and ok_min_n_events:
                fchunks.append((start_idx, count, end_ts, end_ts, exp_base))

        return fchunks

    def __getitem__(self, index):
        path, start, count, bbox_ts_start, bbox_ts_end, base_exp = self.chunks[index]
        ann_path = self.filereader.get_ann_path(path)
        l, x, y, ts, p = self.read_example(path, start=start, count=count)
        if base_exp > 0:
            ts += 2 ** base_exp
        ann = self.read_annotation(ann_path, ts_start=bbox_ts_start, ts_end=bbox_ts_end+1)

        events = np.column_stack([x, y, ts, p])

        if self.transforms is not None:
            events = self.transforms(events)

        if self.mixed_transforms is not None:
            events, ann = self.mixed_transforms(events=events, bboxes=ann)

        return events, ann

    def __len__(self):
        return len(self.chunks)

    @property
    def num_classes(self):
        return NotImplementedError("This functionality is not available for detection datasets")

    def _path_to_class(self, path):
        return NotImplementedError("This functionality is not available for detection datasets")
