import os
import numpy as np
import scipy.io as sio
from numba import njit


class FileReader:

    def __init__(self, format=None):
        self.format = format
        self.ext = ""

    def read_example(self, filename, start=0, count=-1):
        raise NotImplementedError("Abstract FileReader method called. You should implement this method "
                                  "with a concrete child class.")

    def read_annotation(self, filename, start=0, count=-1):
        raise NotImplementedError("Abstract FileReader method called. You should implement this method "
                                  "with a concrete child class.")

    def get_ann_path(self, event_path):
        raise NotImplementedError("Abstract FileReader method called. You should implement this method "
                                  "with a concrete child class.")


class BinFileReader(FileReader):

    def __init__(self, format=None):
        super().__init__(format=format)
        self.ext = ".bin"

    def read_example(self, filename, start=0, count=-1):
        f = open(filename, 'rb')
        # Moves the file pointer at the 'start' position based on the event size
        f.seek(start * 5)  # ev_size = 40bit (5 * 8bit)
        # Reads count events (-1, default values, means all), i.e. (count * 5) bytes
        raw_data = np.fromfile(f, dtype=np.uint8, count=count * 5)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        x = np.array(all_x[td_indices], dtype=np.int32)
        y = np.array(all_y[td_indices], dtype=np.int32)
        ts = np.array(all_ts[td_indices], dtype=np.int32)
        p = np.array(all_p[td_indices], dtype=np.int32)
        length = len(x)

        return length, x, y, ts, p


class AerFileReader(FileReader):

    def __init__(self, format=None):
        super().__init__(format=format)
        self.ext = ".aedat"

    def _get_camera_format(self):

        if self.format == "DVS128":
            x_mask = 0xFE
            x_shift = 1
            y_mask = 0x7F00
            y_shift = 8
            p_mask = 0x1
            p_shift = 0
        else:
            raise ValueError("Unsupported camera: {}".format(self.format))

        return x_mask, x_shift, y_mask, y_shift, p_mask, p_shift

    def _read_aedat20_events(self, f, count=-1):

        raw_data = np.fromfile(f, dtype=np.int32, count=count*2).newbyteorder('>')
        f.close()

        all_data = raw_data[0::2]
        all_ts = raw_data[1::2]

        # Events' representation depends of the camera format
        x_mask, x_shift, y_mask, y_shift, p_mask, p_shift = self._get_camera_format()

        all_x = ((all_data & x_mask) >> x_shift).astype(np.int32)
        all_y = ((all_data & y_mask) >> y_shift).astype(np.int32)
        all_p = ((all_data & p_mask) >> p_shift).astype(np.int32)
        all_ts = all_ts.astype(np.int32)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def _read_aedat31_events(self, f, count=-1):

        # WARNING: This function assumes that all the events are of type POLARITY_EVENT and so
        # each packet has a fixed size and structure. If your dataset may contain other type of events you
        # must write a function to properly handle different packets' sizes and formats.
        # See: https://inilabs.com/support/software/fileformat/#h.w7vjqzw55d5b

        raw_data = np.fromfile(f, dtype=np.int32, count=count*2)
        f.close()

        all_x, all_y, all_ts, all_p = [], [], [], []

        while raw_data.size > 0:

            # Reads the header
            block_header, raw_data = raw_data[:7], raw_data[7:]
            eventType = block_header[0] >> 16
            eventSize, eventTSOffset, eventTSOverflow, eventCapacity, eventNumber, eventValid = block_header[1:]
            size_events = eventNumber * eventSize // 4
            events, raw_data = raw_data[:size_events], raw_data[size_events:]

            if eventValid and eventType == 1:
                data = events[0::2]
                ts = events[1::2]

                x = ((data >> 17) & 0x1FFF).astype(np.int32)
                y = ((data >> 2) & 0x1FFF).astype(np.int32)
                p = ((data >> 1) & 0x1).astype(np.int32)
                valid = (data & 0x1).astype(np.bool)
                ts = ((eventTSOverflow.astype(np.int64) << 31) | ts).astype(np.int64)

                # The validity bit can be used to invalidate events. We filter out the invalid ones
                if not np.all(valid):
                    x = x[valid]
                    y = y[valid]
                    ts = ts[valid]
                    p = p[valid]

                all_x.append(x)
                all_y.append(y)
                all_ts.append(ts)
                all_p.append(p)

        all_x = np.concatenate(all_x, axis=-1)
        all_y = np.concatenate(all_y, axis=-1)
        all_ts = np.concatenate(all_ts, axis=-1)
        all_p = np.concatenate(all_p, axis=-1)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def read_example(self, filename, start=0, count=-1):
        f = open(filename, 'rb')

        # If comment section is not present, version 1.0 is assumed by the standard
        version = "1.0"
        prev = 0
        line = f.readline().decode("utf-8", "ignore")
        # Reads the comments and extracts the aer format version
        while line.startswith('#'):
            if line[0:9] == '#!AER-DAT':
                version = line[9:12]
            prev = f.tell()
            line = f.readline().decode("utf-8", "ignore")
        # Repositions the pointer at the beginning of data section
        f.seek(prev)

        # Moves the file pointer at the 'start' position based on the event size
        f.seek(start * 4 * 2, os.SEEK_CUR)  # ev_size = 2 * 32bit (2*4*8bit)
        # Reads count events (-1, default values, means all), i.e. (count * 5) bytes

        if version == "2.0":
            length, all_x, all_y, all_ts, all_p = self._read_aedat20_events(f, count=count)
        elif version == "3.1":
            length, all_x, all_y, all_ts, all_p = self._read_aedat31_events(f, count=count)
        else:
            raise NotImplementedError("Reader for version {} has not yet been implemented.".format(version))

        return length, all_x, all_y, all_ts, all_p


class PropheseeReader(FileReader):

    def __init__(self, format=None):
        super().__init__(format=format)
        self.ext = ".dat"

    def read_example(self, filename, start=0, count=-1):

        #                ts                nothing  p     y         x
        # |______________________________||________|_|_________|__________|
        #              32bit                      1bit  14bit     14bit

        with open(filename, 'rb') as f:
            # If comment section is not present, version 1.0 is assumed by the standard
            headers = {}
            has_comments = False
            prev = 0
            line = f.readline().decode("utf-8", "ignore")
            # Reads the comments and extracts the aer format version
            while line.startswith('%'):
                line = line[:-1] if line[-1] == '\n' else line
                words = line.split(' ')
                if len(words) > 2:
                    has_comments = True
                    if words[1] == 'Date':
                        if len(words) > 3:
                            headers.update({words[1]: words[2] + ' ' + words[3]})
                    else:
                        headers.update({words[1]: words[2:]})
                prev = f.tell()
                line = f.readline().decode("utf-8", "ignore")
            # Repositions the pointer at the beginning of data section
            f.seek(prev)

            if has_comments:
                ev_type = int.from_bytes(f.read(1), byteorder='little')
                ev_size = int.from_bytes(f.read(1), byteorder='little')
            else:
                ev_type = 0
                ev_size = 8

            # Moves the file pointer at the 'start' position based on the event size
            f.seek(start * ev_size, os.SEEK_CUR)
            # Reads count events (-1, default value, means all)
            raw_data = np.fromfile(f, dtype=np.uint32, count=max(-1, count * ev_size//4)).newbyteorder('<')

        all_ts = raw_data[0::1+(ev_size-4)//4].astype(np.float)
        all_addr = raw_data[1::1+(ev_size-4)//4]

        version = int(headers.get('Version', ["0"])[0])
        xmask = 0x00003FFF
        ymask = 0x0FFFC000
        polmask = 0x10000000
        xshift = 0
        yshift = 14
        polshift = 28

        all_addr = np.abs(all_addr)
        all_x = ((all_addr & xmask) >> xshift).astype(np.float)
        all_y = ((all_addr & ymask) >> yshift).astype(np.float)
        all_p = ((all_addr & polmask) >> polshift).astype(np.float)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p


class NumpyFileReader(FileReader):

    def __init__(self, format=None):
        super().__init__(format=format)
        self.ext = ".npy"

    def read_example(self, filename, start=0, count=-1):
        data = np.load(filename)
        if count >= 0:
            data = data[start:start+count]
        else:
            data = data[start:]

        x = data[..., 0]
        y = data[..., 1]
        ts = data[..., 2]
        p = (data[..., 3] + 1) // 2  # map -1 -> 0, 0 -> 1
        length = len(x)

        return length, x, y, ts, p


class MatFileReader(FileReader):

    def __init__(self, format=None):
        super().__init__(format=format)
        self.ext = ".mat"

    def read_example(self, filename, start=0, count=-1):
        data = sio.loadmat(filename)
        if count < 0:
            count = -start-1  # all events

        x = data['x'][start:start+count].astype(np.float32)
        y = data['y'][start:start+count].astype(np.float32)
        ts = data['ts'][start:start+count].astype(np.float32)
        p = data['pol'][start:start+count].astype(np.float32)
        length = len(x)

        return length, x, y, ts, p
