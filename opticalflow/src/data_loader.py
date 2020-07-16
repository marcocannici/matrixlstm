#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
# NOTE: same as slim version, but exposes reader_kwargs
from dataset_data_provider import DatasetDataProvider

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

    """
    Reads n_frames of events from data (from a TFRecord) and generates an event image consisting
    of [pos counts, neg counts, pos last time, neg last time].
    """
    def _read_events(self, data, n_frames):
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
    Reads a grayscale image from a png.
    """
    def _read_image(self, img_path):
        img_path = tf.read_file(self._root_path + img_path)
        image = tf.image.decode_png(img_path, channels=1)
        image = tf.cast(image, tf.float32)
        return image

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
            'image_times': tf.FixedLenFeature([], tf.string),
            'prefix': tf.FixedLenFeature([], tf.string),
            'cam': tf.FixedLenFeature([], tf.string)
        }
        
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

        event_image = self._read_events(data, n_frames)
        
        # Get paths to grayscale png files.
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
                                        tf.as_string(image_iter+n_frames, width=5, fill='0'), 
                                        ".png"])
        prev_image = self._read_image(prev_img_path)
        next_image = self._read_image(next_img_path)

        outputs = []
        for item in self._items_to_features.keys():
            if item == 'event_image':
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
        if isinstance(sequence, (list, )):
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
               flip_updown=False,
               nskips=_N_SKIP,
               gzip=False):
    print("Loading data!")
    if split is None:
        split = 'train'

    tfrecord_paths_np, n_ima = read_file_paths(
        root,
        split,
        sequence)
  
    items_to_features = {
        'event_image': tf.FixedLenFeature([], tf.string),
        'prev_image': tf.FixedLenFeature([], tf.string),
        'next_image': tf.FixedLenFeature([], tf.string),
        'timestamps': tf.FixedLenFeature([], tf.string)
    }
    
    items_to_descriptions = {
        'event_image': 'Event image',
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

    num_epochs = None
    if split is 'test':
        num_epochs = 1

    if gzip:
        options = {'options': tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)}
        print("Decoding TFRecord using GZip")
    else:
        options = None
    provider = DatasetDataProvider(
        dataset,
        num_readers=4,
        shuffle=shuffle,
        num_epochs=num_epochs,
        common_queue_capacity=20*batch_size,
        common_queue_min=10*batch_size,
        reader_kwargs=options)

    keys = items_to_features.keys()
    values = provider.get(keys)

    dict_batch = dict(zip(keys,values))
    event_image = dict_batch['event_image']
    prev_image = dict_batch['prev_image']
    next_image = dict_batch['next_image']
    timestamps = dict_batch['timestamps']
    
    n_split = 6
    event_size = 4
    if time_only or count_only:
        n_split = 4
        event_size = 2

    # Do data augmentation during training. Random flipping, rotations, and cropping.
    if split == 'train':
        images_concat = tf.concat([event_image, prev_image, next_image], axis=2)
        # ## random flip up down ###
        if flip_updown:
            images_concat = tf.image.random_flip_up_down(images_concat)

        # ## random flip right left ###
        images_concat = tf.image.random_flip_left_right(images_concat)

        # ## random rotation +/- 30 ###
        if rotation:
            random_angles = tf.random_uniform([1],
                                              minval=-30,
                                              maxval=30,
                                              dtype=tf.float32)
            images_rotated = tf.contrib.image.rotate(images_concat,
                                                     random_angles * np.pi / 180.,
                                                     interpolation='NEAREST')
        else:
            images_rotated = images_concat

        # ## random crop ###
        image_cropped = tf.random_crop(images_rotated,
                                       [image_height, image_width, n_split])
        event_image, prev_image, next_image = tf.split(image_cropped,
                                                       [event_size, 1, 1],
                                                       axis=2)
    # Otherwise just centrally crop the images.
    else:
        event_image = tf.image.resize_image_with_crop_or_pad(event_image, 
                                                             image_height, 
                                                             image_width)
        prev_image = tf.image.resize_image_with_crop_or_pad(prev_image, 
                                                            image_height, 
                                                            image_width)
        next_image = tf.image.resize_image_with_crop_or_pad(next_image, 
                                                            image_height, 
                                                            image_width)

    event_image.set_shape([image_height, image_width, event_size])
    prev_image.set_shape([image_height, image_width, 1])
    next_image.set_shape([image_height, image_width, 1])

    if split == 'train':
        values_batch = tf.train.shuffle_batch([event_image, prev_image, next_image, timestamps],
                                              num_threads=4,
                                              batch_size=batch_size, 
                                              capacity=20000,
                                              min_after_dequeue=8000)
    else:
        values_batch = tf.train.batch([event_image, prev_image, next_image, timestamps],
                                      num_threads=4,
                                      batch_size=batch_size, 
                                      capacity=10*batch_size)

    return values_batch[0], values_batch[1], values_batch[2], values_batch[3], n_ima
