import re
import sys
import numba
import numpy as np
from torchvision.transforms import Compose


def add_transforms_params(parser):

    def has_transform(s):
        # Checks if the transform 's' has been provided in the --transforms argument list
        args_str = "".join([" %s" % args for args in sys.argv[1:]])
        return re.findall("--transforms[\w ]* %s[\w ]*(?!--)" % s, args_str)

    parser.add_argument('--transforms', type=str, nargs='+', default=[])
    parser.add_argument('--test_transforms', type=str, nargs='+', default=[])

    parser.add_argument('--transform_timewindow_start', type=float, required=has_transform('timewindow'))
    parser.add_argument('--transform_timewindow_end', type=float, required=has_transform('timewindow'))

    parser.add_argument('--transform_randtimewindow_min', type=float, required=has_transform('randtimewindow'))
    parser.add_argument('--transform_randtimewindow_max', type=float, required=has_transform('randtimewindow'))

    parser.add_argument('--transform_rot_min', type=float, required=has_transform('rot'))
    parser.add_argument('--transform_rot_max', type=float, required=has_transform('rot'))

    parser.add_argument('--transform_hflip_prob', type=float, required=has_transform('hflip'))

    parser.add_argument('--transform_vflip_prob', type=float, required=has_transform('vflip'))

    parser.add_argument('--transform_centercrop_h', type=int, required=has_transform('centercrop'))
    parser.add_argument('--transform_centercrop_w', type=int, required=has_transform('centercrop'))

    parser.add_argument('--transform_randcrop_h', type=int, required=has_transform('randcrop'))
    parser.add_argument('--transform_randcrop_w', type=int, required=has_transform('randcrop'))

    parser.add_argument('--transform_extendrandcrop_h', type=int, required=has_transform('extendrandcrop'))
    parser.add_argument('--transform_extendrandcrop_w', type=int, required=has_transform('extendrandcrop'))

    parser.add_argument('--transform_simplescale_min', type=float, required=has_transform('simplescale'))
    parser.add_argument('--transform_simplescale_max', type=float, required=has_transform('simplescale'))

    parser.add_argument('--transform_scale_min', type=float, required=has_transform('scale'))
    parser.add_argument('--transform_scale_max', type=float, required=has_transform('scale'))
    parser.add_argument('--transform_scale_delta_t', type=int, required=has_transform('scale'))
    parser.add_argument('--transform_scale_radius', type=int, required=has_transform('scale'))

    return parser


def get_transforms(args, transform_name='transforms'):

    transforms = []
    if hasattr(args, transform_name) and getattr(args, transform_name):
        for transf in getattr(args, transform_name):
            if transf == 'rot':
                transforms.append(RandomClockwiseRotateEvents(args.transform_rot_min,
                                                              args.transform_rot_max))
            elif transf == 'hflip':
                transforms.append(RandomHorizontalFlipEvents(args.transform_hflip_prob))
            elif transf == 'vflip':
                transforms.append(RandomVerticalFlipEvents(args.transform_vflip_prob))
            elif transf == 'centercrop':
                transforms.append(CenterCropEvents(args.transform_centercrop_h,
                                                   args.transform_centercrop_w))
            elif transf == 'randcrop':
                transforms.append(CenterCropEvents(args.transform_randcrop_h,
                                                   args.transform_randcrop_w))
            elif transf == 'extendrandcrop':
                transforms.append(ExtendRandomCropEvents(args.transform_extendrandcrop_h,
                                                         args.transform_extendrandcrop_w))
            elif transf == 'timewindow':
                transforms.append(TimeWindow(args.transform_timewindow_start,
                                             args.transform_timewindow_end))
            elif transf == 'randtimewindow':
                transforms.append(RandomTimeWindow(args.transform_randtimewindow_min,
                                                   args.transform_randtimewindow_max))
            else:
                raise ValueError("'%s' transformation not recognized!" % transf)

    if transforms:
        return Compose(transforms)
    return None


class TimeWindow(object):

    def __init__(self, start=0, end=-1):
        self.start_ts = start
        self.end_ts = end

    def __repr__(self):
        return "{0.__class__.__name__}(start_ts={0.start_ts}, " \
               "end_ts={0.end_ts})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """
        _, _, ts, _ = np.split(events, 4, axis=-1)
        end_ts = ts.max() if self.end_ts < 0 else self.end_ts
        inside_mask = np.logical_and(ts >= self.start_ts, ts <= end_ts)
        inside_mask = inside_mask.reshape(-1)

        return events[inside_mask]


class RandomTimeWindow(object):

    def __init__(self, min_end=0, max_end=-1):
        self.min_end = min_end
        self.max_end = max_end

    def __repr__(self):
        return "{0.__class__.__name__}(min_end={0.min_end}, " \
               "max_end={0.max_end})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """
        _, _, ts, _ = np.split(events, 4, axis=-1)
        max_end = ts.max() if self.max_end < 0 else self.max_end
        end_ts = np.random.randint(self.min_end, max_end + 1)
        inside_mask = ts <= end_ts
        inside_mask = inside_mask.reshape(-1)

        return events[inside_mask]


class RandomClockwiseRotateEvents(object):

    def __init__(self, min_rot=-60, max_rot=60):
        self.min_rot = min_rot
        self.max_rot = max_rot

    def __repr__(self):
        return "{0.__class__.__name__}(min_rot={0.min_rot}, " \
               "max_rot={0.max_rot})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        # Compute the center of the events cloud
        xc = (x.max() - x.min()) / 2
        yc = (y.max() - y.min()) / 2
        # Select a random angle
        angle = np.radians(np.random.uniform(self.min_rot, self.max_rot))

        # Apply rotation
        x_rot = ((x - xc) * np.cos(angle)) - ((y - yc) * np.sin(angle)) + xc
        y_rot = ((x - xc) * np.sin(angle)) + ((y - yc) * np.cos(angle)) + yc

        # Translate events so that the top-left most event is in (0,0)
        x_left = np.min(x_rot)
        y_top = np.min(y_rot)
        x_rot -= x_left
        y_rot -= y_top

        x_rot = np.around(x_rot).astype(np.int32)
        y_rot = np.around(y_rot).astype(np.int32)

        return np.column_stack([x_rot, y_rot, ts, p])


class RandomHorizontalFlipEvents(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __repr__(self):
        return "{0.__class__.__name__}(flip_prob={0.flip_prob})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)

        if np.random.uniform() < self.flip_prob:
            x = x.max() + x.min() - x
        return np.column_stack([x, y, ts, p])


class RandomVerticalFlipEvents(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __repr__(self):
        return "{0.__class__.__name__}(flip_prob={0.flip_prob})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)

        if np.random.uniform() < self.flip_prob:
            y = y.max() + y.min() - y
        return np.column_stack([x, y, ts, p])


class CenterCropEvents(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __repr__(self):
        return "{0.__class__.__name__}(height={0.height}, " \
               "width={0.width})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        # Compute the center of the events cloud
        xc = (x.max() - x.min()) // 2
        yc = (y.max() - y.min()) // 2

        # Select events inside the crop space
        crop_w_half = self.width // 2 - 1 if self.width % 2 == 0 else self.width // 2
        crop_h_half = self.height // 2 - 1 if self.height % 2 == 0 else self.height // 2
        inside = np.logical_and(np.abs(x - xc) <= crop_w_half, np.abs(y - yc) <= crop_h_half)

        # Select events inside the crop and place the top-left most
        # event in (0, 0)
        x = x[inside]
        x -= x.min()
        y = y[inside]
        y -= y.min()
        ts = ts[inside]
        p = p[inside]

        w = x.max() - x.min() + 1
        h = y.max() - y.min() + 1
        if w < self.width:
            x += min(self.width // 2 - w // 2, self.width - x.max() - 1)
        if h < self.height:
            y += min(self.height // 2 - h // 2, self.height - y.max() - 1)

        return np.column_stack([x, y, ts, p])


class RandomCropEvents(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __repr__(self):
        return "{0.__class__.__name__}(height={0.height}, " \
               "width={0.width})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        # Compute the frame size
        w = x.max() - x.min()
        h = y.max() - y.min()

        # Select the top-left corner of the crop area at random
        left = np.random.randint(x.min(), max(x.min() + 1, w - self.width))
        top = np.random.randint(y.min(), max(y.min() + 1, h - self.height))

        # Select events inside the crop area
        inside = np.logical_and.reduce([x >= left, x < left + self.width,
                                        y >= top, y < top + self.height])

        # Select events inside the crop and place the top-left most
        # event in (0, 0)
        x = x[inside]
        x -= x.min()
        y = y[inside]
        y -= y.min()
        ts = ts[inside]
        p = p[inside]

        return np.column_stack([x, y, ts, p])


class ExtendRandomCropEvents(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __repr__(self):
        return "{0.__class__.__name__}(height={0.height}, " \
               "width={0.width})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        # Compute the events frame size
        w = x_max - x_min
        h = y_max - y_min

        if w >= self.width:
            left = np.random.randint(x_min, max(x_min + 1, w - self.width))
        else:
            left = np.random.randint(x_max - self.width + 1, x_min + 1)
        if h >= self.height:
            top = np.random.randint(y_min, max(y_min + 1, h - self.height))
        else:
            top = np.random.randint(y_max - self.height + 1, y_min + 1)

        # Select events inside the crop area
        inside = np.logical_and.reduce([x >= left, x < left + self.width,
                                        y >= top, y < top + self.height])

        # Select events inside the crop and place the top-left most
        # event in (0, 0) if the canvas is smaller than the event
        # frame size, and to add a margin otherwise
        x = x[inside]
        x -= left
        y = y[inside]
        y -= top
        ts = ts[inside]
        p = p[inside]

        return np.column_stack([x, y, ts, p])


class RemoveEventOffset(object):

    def __repr__(self):
        return "{0.__class__.__name__}"

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        events[:, 2] -= events[:, 2].min()
        return events
