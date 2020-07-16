import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset


class EventDataset(Dataset):

    def __init__(self, filereader, files_source, format=None, transforms=None):
        super().__init__()

        self.filereader = filereader
        self.format = format
        self.transforms = transforms

        if isinstance(files_source, str) and os.path.isdir(files_source):
            self.files = glob.glob(os.path.join(files_source, "*/*" + self.filereader.ext))
        elif isinstance(files_source, list):
            self.files = files_source
        else:
            raise ValueError("You should provide either dir_path or paths_list!")

        self.classes = sorted(np.unique([os.path.basename(os.path.dirname(f)) for f in self.files]))
        self.name_to_id = {cls: i for i, cls in enumerate(self.classes)}

    @property
    def num_classes(self):
        return len(self.classes)

    def _path_to_class(self, path):
        cls_name = os.path.basename(os.path.dirname(path))
        return self.name_to_id[cls_name]

    def read_example(self, filename, start=0, count=-1):
        return self.filereader.read_example(filename, start=start, count=count)

    def __getitem__(self, index):
        path = self.files[index]
        l, x, y, ts, p = self.read_example(path)
        lbl = self._path_to_class(path)
        events = np.column_stack([x, y, ts, p])

        if self.transforms is not None:
            events = self.transforms(events)

        return events, lbl

    def __len__(self):
        return len(self.files)


class EventDetectionDataset(EventDataset):

    def __init__(self, filereader, files_source, format=None,
                 transforms=None, mixed_transforms=None):
        super(EventDataset, self).__init__()

        self.filereader = filereader
        self.format = format
        self.transforms = transforms
        self.mixed_transforms = mixed_transforms

        # FIXME: you may need to add an additional argument "classes" providing an ordered
        #        list of strings mapping class ids to class names

        if isinstance(files_source, str) and os.path.isdir(files_source):
            # Detection datasets are not organized in directories based on the class
            # all files and annotations are on the same directory
            self.files = glob.glob(os.path.join(files_source, "*" + self.filereader.ext))
        elif isinstance(files_source, list):
            self.files = files_source
        else:
            raise ValueError("You should provide either dir_path or paths_list!")

    def read_example(self, filename, start=0, count=-1):
        return self.filereader.read_example(filename, start=start, count=count)

    def read_annotation(self, filename, ts_start=None, ts_end=None):
        return self.filereader.read_annotation(filename, ts_start=ts_start, ts_end=ts_end)

    def __getitem__(self, index):
        path = self.files[index]
        ann_path = self.filereader.get_ann_path(path)
        l, x, y, ts, p = self.read_example(path)
        events = np.column_stack([x, y, ts, p])
        ann = self.read_annotation(ann_path)

        if self.transforms is not None:
            events = self.transforms(events)

        if self.mixed_transforms is not None:
            events, ann = self.mixed_transforms(events, ann)

        return events, ann

    def __len__(self):
        return len(self.files)

    @property
    def num_classes(self):
        return NotImplementedError("This functionality is not available for detection datasets")

    def _path_to_class(self, path):
        return NotImplementedError("This functionality is not available for detection datasets")
