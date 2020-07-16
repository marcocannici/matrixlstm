#!/usr/bin/env python
import os
import time
import h5py
import numpy as np
import tensorflow as tf
# NOTE: same as slim version, but exposes reader_kwargs
from dataset_data_provider import DatasetDataProvider
from event_augmentation import *

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1


class EventDataDecoder(tf.contrib.slim.data_decoder.DataDecoder):
    """
    Decoder to read events and grayscale images from TFRecords and PNG files.
    """

    def __init__(self,
                 items_to_features,
                 image_width,
                 image_height,
                 root_path,
                 split,
                 skip_frames=False,
                 nskips=_N_SKIP,
                 time_only=False,
                 count_only=False):
        self._items_to_features = items_to_features
        self._image_width = image_width
        self._image_height = image_height
        self._root_path = tf.convert_to_tensor(root_path, dtype=tf.string)
        self._split = split
        self._skip_frames = skip_frames
        self._time_only = time_only
        self._count_only = count_only
        self._nskips = nskips

    def list_items(self):
        return list(self._items_to_features.keys())

    @staticmethod
    def _read_hdf5(hdf5_path, cam, event_start_idx, event_end_idx, n_frames, start_ts):
        if not os.path.isfile(hdf5_path):
            raise RuntimeError("File %s doest not exist!" % hdf5_path)
        h5_data = h5py.File(hdf5_path, 'r')
        event_data = h5_data['davis'][cam]['events']

        # Take all the events between the first event of the first image
        # and the last event of the n_frames image
        start_idx = event_start_idx[0]
        end_idx = event_end_idx[n_frames-1]

        events = event_data[start_idx: end_idx + 1].copy()
        # Zero-offset events wrt to the timestamp of the first frame
        events[:, 2] -= start_ts

        try:
            h5_data.close()
        except:
            pass  # Was already closed
        del h5_data
        del event_data

        return events.astype(np.float64)


    """
    Reads a grayscale image from a png.
    """

    def _read_image(self, img_path):
        img_path = tf.read_file(self._root_path + img_path)
        image = tf.image.decode_png(img_path, channels=1)
        image = tf.cast(image, tf.float32)
        return image

    """
    Reads raw events corresponding to n_frame consecutive frames
    """

    def _read_events(self, data, n_frames, start_ts):
        # Each folder (ie, outdoor_day1, oudoor_day2, ..) contains one hdf5 file
        # with all the events, we read it using hd5py
        hdf5_path = self._root_path + tf.string_join([data['prefix'], "/",
                                                      data['prefix'] + "_data.hdf5"])

        # Reads the id of the first (start) and last (end) event associated
        # to each image
        event_start_idx = tf.decode_raw(data['event_start_idx'], tf.int32)
        event_end_idx = tf.decode_raw(data['event_end_idx'], tf.int32)

        events = tf.py_func(self._read_hdf5,
                            [hdf5_path,
                             data['cam'],
                             event_start_idx,
                             event_end_idx,
                             n_frames,
                             start_ts],
                            tf.float64)
        return events

    """
        Reads n_frames of events from data (from a TFRecord) and generates an event image consisting
        of [pos counts, neg counts, pos last time, neg last time].
        """

    def _read_event_image(self, data, n_frames):
        shape = tf.decode_raw(data['shape'], tf.uint16)
        shape = tf.cast(shape, tf.int32)

        event_count_images = tf.decode_raw(data['event_count_images'], tf.uint16)
        event_count_images = tf.reshape(event_count_images, shape)
        event_count_images = tf.cast(event_count_images, tf.float32)
        event_count_image = event_count_images[:n_frames, :, :, :]
        event_count_image = tf.reduce_sum(event_count_image, axis=0)

        event_time_images = tf.decode_raw(data['event_time_images'], tf.float32)
        event_time_images = tf.reshape(event_time_images, shape)
        event_time_images = tf.cast(event_time_images, tf.float32)
        event_time_image = event_time_images[:n_frames, :, :, :]
        event_time_image = tf.reduce_max(event_time_image, axis=0)

        # Normalize timestamp image to be between 0 and 1.
        event_time_image /= tf.reduce_max(event_time_image)

        if self._count_only:
            event_image = event_count_image
        elif self._time_only:
            event_image = event_time_image
        else:
            event_image = tf.concat([event_count_image, event_time_image], 2)

        event_image = tf.cast(event_image, tf.float32)
        return event_image


    """
    Decode a TFRecord into Tensorflow friendly data.
    """

    def decode(self, serialized_example, items=None):
        global _MAX_SKIP_FRAMES, _TEST_SKIP_FRAMES
        features = {
            'image_iter': tf.FixedLenFeature([], tf.int64),
            'shape': tf.FixedLenFeature([], tf.string),
            'event_count_images': tf.FixedLenFeature([], tf.string),
            'event_time_images': tf.FixedLenFeature([], tf.string),
            'event_start_idx': tf.FixedLenFeature([], tf.string),
            'event_end_idx': tf.FixedLenFeature([], tf.string),
            'image_times': tf.FixedLenFeature([], tf.string),
            'prefix': tf.FixedLenFeature([], tf.string),
            'cam': tf.FixedLenFeature([], tf.string)
        }

        # Extract metadata and event frames
        data = tf.parse_single_example(serialized_example,
                                       features)
        image_iter = data['image_iter']
        prefix = data['prefix']
        cam = data['cam']
        image_times = tf.decode_raw(data['image_times'], tf.float64)

        if self._split is 'test':
            if self._skip_frames:
                n_frames = _TEST_SKIP_FRAMES
            else:
                n_frames = 1
        else:
            n_frames = tf.random_uniform([], 1, _MAX_SKIP_FRAMES, dtype=tf.int64) * self._nskips

        timestamps = [image_times[0], image_times[n_frames]]
        events = self._read_events(data, n_frames, image_times[0])
        event_image = self._read_event_image(data, n_frames)

        prev_img_path = tf.string_join([prefix,
                                        "/",
                                        cam,
                                        "_image",
                                        tf.as_string(image_iter, width=5, fill='0'),
                                        ".png"])
        next_img_path = tf.string_join([prefix,
                                        "/",
                                        cam,
                                        "_image",
                                        tf.as_string(image_iter + n_frames, width=5, fill='0'),
                                        ".png"])
        prev_image = self._read_image(prev_img_path)
        next_image = self._read_image(next_img_path)

        outputs = []
        # Returns only the requested features as output
        for item in self._items_to_features.keys():
            if item == 'events':
                outputs.append(events)
            elif item == 'event_image':
                outputs.append(event_image)
            elif item == 'prev_image':
                outputs.append(prev_image)
            elif item == 'next_image':
                outputs.append(next_image)
            elif item == 'timestamps':
                outputs.append(timestamps)
            else:
                raise NameError("Item {} is not valid.".format(item))

        return outputs


"""
Generates paths of training data, read from data_folder_path/${split}_bags.txt.
"""

def read_file_paths(data_folder_path,
                    split,
                    sequence=None):
    tfrecord_paths = []
    n_ima = 0
    if sequence is None:
        bag_list_file = open(os.path.join(data_folder_path, "{}_bags.txt".format(split)), 'r')
        lines = bag_list_file.read().splitlines()
        bag_list_file.close()
    else:
        if isinstance(sequence, (list,)):
            lines = sequence
        else:
            lines = [sequence]

    print("%s bags:" % split.capitalize(), sequence)

    for line in lines:
        bag_name = line

        num_ima_file = open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r')
        num_imas = num_ima_file.read()
        num_ima_file.close()
        num_imas_split = num_imas.split(' ')
        n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
        n_ima += n_left_ima
        tfrecord_paths.append(os.path.join(data_folder_path,
                                           bag_name,
                                           "left_event_images.tfrecord"))

        n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
        if n_right_ima > 0 and not split is 'test':
            n_ima += n_right_ima
            tfrecord_paths.append(os.path.join(data_folder_path,
                                               bag_name,
                                               "right_event_images.tfrecord"))

    return tfrecord_paths, n_ima


"""
Generates batched data.
"""

def get_loader(root,
               batch_size,
               image_width,
               image_height,
               split=None,
               shuffle=True,
               sequence=None,
               skip_frames=False,
               time_only=False,
               count_only=False,
               rotation=True,
               rewind=False,
               flip_updown=False,
               nskips=_N_SKIP,
               num_epochs=None,
               binarize_polarity=False,
               original_shape=(260, 346)):
    print("Loading data!")
    if split is None:
        split = 'train'

    tfrecord_paths_np, n_ima = read_file_paths(
        root,
        split,
        sequence)

    items_to_features = {
        'events': tf.FixedLenFeature([], tf.string),
        'event_image': tf.FixedLenFeature([], tf.string),
        'prev_image': tf.FixedLenFeature([], tf.string),
        'next_image': tf.FixedLenFeature([], tf.string),
        'timestamps': tf.FixedLenFeature([], tf.string)
    }

    items_to_descriptions = {
        'events': 'Sequence of camera events',
        'event_image': 'Timestamps of the start and end of the time window',
        'prev_image': 'Previous grayscale image',
        'next_image': 'Next grayscale image',
        'timestamps': 'Timestamps of the start and end of the time window'
    }

    event_data_decoder = EventDataDecoder(items_to_features,
                                          image_width,
                                          image_height,
                                          root,
                                          split,
                                          skip_frames,
                                          nskips,
                                          time_only,
                                          count_only)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=tfrecord_paths_np,
        reader=tf.TFRecordReader,
        decoder=event_data_decoder,
        num_samples=n_ima,
        items_to_descriptions=items_to_descriptions)

    #num_epochs = None
    if split is 'test':
        num_epochs = 1

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    provider = DatasetDataProvider(
        dataset,
        num_readers=4,
        shuffle=shuffle,
        num_epochs=num_epochs,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size,
        reader_kwargs={'options': options})

    keys = items_to_features.keys()
    values = provider.get(keys)

    dict_batch = dict(zip(keys, values))
    events = dict_batch['events']
    prev_image = dict_batch['prev_image']
    next_image = dict_batch['next_image']
    timestamps = dict_batch['timestamps']
    event_image = dict_batch['event_image']

    events.set_shape([None, 4])
    event_size = 2 if time_only or count_only else 4
    rot_angle, crop_bbox = None, None

    # Do data augmentation during training. Random flipping, rotations, and cropping.
    if split == 'train':
        # ## random rewind ###
        if rewind:
            if count_only:
                times_image, counts_image = None, event_image
            elif time_only:
                times_image, counts_image = event_image, None
            else:
                times_image, counts_image = event_image[..., :2], event_image[..., 2:]
            rewind_values = random_rewind_forward(times_image, counts_image,
                                                 prev_image, next_image,
                                                 events, timestamps,
                                                 original_shape)
            times_image, counts_image, prev_image, next_image, events = rewind_values
            events_images_list = [i for i in [times_image, counts_image] if i is not None]
            event_image = tf.concat(events_images_list, axis=-1)

        # ## random flip up down ###
        images_concat = tf.concat([event_image, prev_image, next_image], axis=-1)
        if flip_updown:
            images_concat, events = random_flip_up_down(images_concat, events)

        # ## random flip right left ###
        images_concat, events = random_flip_left_right(images_concat, events)

        if rotation:
            # ## random rotation +/- 30 ###
            # Crop grayscale images and compute events mask
            images_concat, mask, rot_angle = rotate_partial(images_concat,
                                                            partial_mask=None,
                                                            minval=-30, maxval=30,
                                                            interpolation='NEAREST')
            # ## random crop ###
            images_concat, mask, crop_bbox = random_crop_partial(images_concat,
                                                                 partial_mask=mask,
                                                                 shape=[image_height, image_width])
            # Revers transformations to obtain the mask to apply on events
            mask = tf.contrib.image.rotate(mask, -rot_angle, interpolation='NEAREST')
            events = apply_event_mask(events, mask)
            # Filter events using the computed mask
        else:
            # ## random crop ###
            images_concat, events = random_crop(images_concat, events,
                                                [image_height, image_width])
            # This disable dynamic rotation and crop performed
            # within the model dataflow
            rot_angle = None
            crop_bbox = None

        event_image, prev_image, next_image = tf.split(images_concat, [event_size, 1, 1], axis=-1)
    # Otherwise just centrally crop the images.
    else:
        images_concat = tf.concat([event_image, prev_image, next_image], axis=-1)
        images_concat, events = center_crop(images_concat, events,
                                            [image_height, image_width])
        event_image, prev_image, next_image = tf.split(images_concat,
                                                       [event_size, 1, 1],
                                                       axis=-1)
        event_image.set_shape([image_height, image_width, event_size])

    prev_image.set_shape([image_height, image_width, 1])
    next_image.set_shape([image_height, image_width, 1])
    n_events = tf.shape(events)[0]

    if binarize_polarity:
        # Maps polarities in [0, 1], originally were [-1, 1]
        events = tf.concat([events[..., :3], (events[:, 3:] + 1) / 2], axis=-1)

    if split == 'train':
        # shuffle_batch does not have a dynamic_pad flag, so we use a RandomShuffleQueue and
        # then the batch operator with the proper dynamic_pad value
        inputs = [events, n_events, event_image,
                  prev_image, next_image,
                  timestamps[0], timestamps[1]]
        if rotation:
            inputs += [rot_angle, crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]]
        dtypes = [x.dtype for x in inputs]
        shapes = [x.get_shape() for x in inputs]
        queue = tf.RandomShuffleQueue(capacity=3000, min_after_dequeue=2999, dtypes=dtypes)
        enqueue_op = queue.enqueue(inputs)
        qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        inputs = queue.dequeue()
        for tensor, shape in zip(inputs, shapes):
            tensor.set_shape(shape)

        values_batch = tf.train.batch(inputs,
                                      num_threads=4,
                                      batch_size=batch_size,
                                      capacity=100,
                                      dynamic_pad=True)

        ret_vals = [values_batch[0], values_batch[1], values_batch[2], values_batch[3],
                    values_batch[4], [values_batch[5], values_batch[6]]]
        if rotation:
            ret_vals += [values_batch[7], [values_batch[8], values_batch[9],
                                           values_batch[10], values_batch[11]], n_ima]
        else:
            ret_vals += [None, None, n_ima]

        return ret_vals

    else:
        values_batch = tf.train.batch([events, n_events, event_image, prev_image, next_image,
                                       timestamps],
                                      num_threads=4,
                                      batch_size=batch_size,
                                      capacity=10 * batch_size,
                                      dynamic_pad=True)

        return (values_batch[0], values_batch[1], values_batch[2], values_batch[3],
                values_batch[4], values_batch[5], n_ima)


if __name__ == "__main__":
    from config import configs

    args = configs()
    with tf.Session() as sess:

        results = get_loader(
            args.data_path, args.batch_size, args.image_width, args.image_height,
            split='test',
            shuffle=False,
            rotation=False,
            sequence=args.sequences,
            num_epochs=1)
        events, lengths, event_image = results[:3]
        timestamps = results[5]

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        iters = 0
        while not coord.should_stop():
            try:
                #_np_n_events, _np_events, _np_next_image = sess.run([n_events, events, next_image])
                events_np, lengths_np, event_image_np = \
                    sess.run([events, lengths, event_image])

                for sample_events, sample_length, sample_event_image in \
                        zip(events_np, lengths_np, event_image_np):
                    h, w = sample_event_image.shape[:2]
                    count_image = np.zeros(shape=[h, w, 2])
                    time_image = np.zeros(shape=[h, w, 2], dtype=np.float64)
                    sample_events = sample_events[:int(sample_length)]
                    for x, y, ts, p in sample_events:
                        if p > 0:
                            count_image[int(y), int(x), 0] += 1
                            time_image[int(y), int(x), 0] = max(ts, time_image[int(y), int(x), 0])
                        else:
                            count_image[int(y), int(x), 1] += 1
                            time_image[int(y), int(x), 1] = max(ts, time_image[int(y), int(x), 1])
                    time_image /= time_image.max()

                    import pdb; pdb.set_trace()
                    print(sample_events[:, 2].max(), sample_events[:, 2].min())
                    assert bool(np.all(count_image == sample_event_image[:, :, :2]))
                    #assert bool(np.all(time_image == sample_event_image[:, :, 2:]))

                if iters % 100 == 0:
                    print("Processed %d batches" % iters)
                iters += 1
            except tf.errors.OutOfRangeError:
                break

            # for i in range(_np_n_events.shape[0]):
            #     np_n_events = int(_np_n_events[i])
            #     np_events = _np_events[i][:np_n_events]
            #     np_next_image = np.tile(_np_next_image[i].astype(np.uint8), (1,1,3))
            #
            #     event_img = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
            #     for x, y, ts, p in np_events:
            #         x, y = int(x), int(y)
            #         event_img[y, x, 0] = 255
            #
            #     debug_img = cv2.addWeighted(np_next_image, 0.8, event_img, 0.2, 0.0)
            #     cv2.imwrite("debug_img.png", debug_img)
            #     cv2.imwrite("next_image.png", np_next_image)
            #     cv2.imwrite("event_img.png", event_img)
