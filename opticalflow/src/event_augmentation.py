import numpy as np
import tensorflow as tf


def get_mask(h, w, partial_mask):
    if partial_mask is not None:
        return partial_mask
    else:
        return tf.ones([h, w], dtype=tf.uint8)


def apply_event_mask(events, mask):
    with tf.name_scope('apply_event_mask'):
        # For each event coordinate, determines its value
        # inside the mask
        events_yx = tf.stack([events[:, 1], events[:, 0]], axis=-1)
        events_mask = tf.gather_nd(mask, tf.cast(events_yx, tf.int32))
        # Make sure the mask is boolean
        events_mask = tf.cast(events_mask, tf.bool)
        # Apply the mask on events
        inside_events = tf.boolean_mask(events, events_mask)

        return inside_events


def random_rewind_forward(times_image, counts_image,
                          prev_image, next_image,
                          events, timestamps, shape):
    with tf.name_scope('random_rewind_forward'):
        def rewind_events(events, timestamps):
            # Reverse events ordering
            new_events = events[::-1]
            new_events_xy = new_events[:, :2]
            # Swap events polarities
            new_events_p = new_events[:, 3:4] * -1
            # Revert timestamps
            new_events_ts = (timestamps[-1] - timestamps[0]) - new_events[:, 2:3]

            return tf.concat([new_events_xy, new_events_ts, new_events_p], axis=-1)

        def compute_time_image(events, shape, dtype):
            events_x = events[..., 0]
            events_y = events[..., 1]
            events_ts = events[..., 2]
            events_idx = tf.cast(events_y * shape[1] + events_x, tf.int32)

            pos_events_mask = events[..., 3] > 0
            neg_events_mask = tf.math.logical_not(pos_events_mask)

            with tf.variable_scope("random_rewind_forward", reuse=True):
                pos_times_image_var = tf.get_variable("rewind_time_image_pos", dtype=tf.float64)
                neg_times_image_var = tf.get_variable("rewind_time_image_neg", dtype=tf.float64)
            reset_pos_times_image_op = tf.assign(pos_times_image_var,
                                                 tf.zeros(shape=shape[0] * shape[1],
                                                          dtype=tf.float64))
            reset_neg_times_image_op = tf.assign(neg_times_image_var,
                                                 tf.zeros(shape=shape[0] * shape[1],
                                                          dtype=tf.float64))

            with tf.control_dependencies([reset_pos_times_image_op]):
                pos_times_image = tf.scatter_max(pos_times_image_var,
                                                 tf.boolean_mask(events_idx, pos_events_mask),
                                                 tf.boolean_mask(events_ts, pos_events_mask))
                pos_times_image = tf.reshape(pos_times_image, shape)
            with tf.control_dependencies([reset_neg_times_image_op]):
                neg_times_image = tf.scatter_max(neg_times_image_var,
                                                 tf.boolean_mask(events_idx, neg_events_mask),
                                                 tf.boolean_mask(events_ts, neg_events_mask))
                neg_times_image = tf.reshape(neg_times_image, shape)

            # new_times_image /= tf.reduce_max(new_times_image)
            new_times_image = tf.cast(tf.stack([pos_times_image, neg_times_image], axis=-1), dtype)

            return new_times_image

        # Defines a variable that will be used inside one of the branches
        with tf.variable_scope("random_rewind_forward"):
            tf.get_variable("rewind_time_image_pos", shape=shape[0] * shape[1],
                            dtype=tf.float64, trainable=False)
            tf.get_variable("rewind_time_image_neg", shape=shape[0] * shape[1],
                            dtype=tf.float64, trainable=False)

        rand = tf.random_uniform([], dtype=tf.float32)
        new_prev_image = tf.cond(rand > 0.5, lambda: next_image, lambda: prev_image)
        new_next_image = tf.cond(rand > 0.5, lambda: prev_image, lambda: next_image)
        new_events = tf.cond(rand > 0.5,
                             lambda: rewind_events(events, timestamps),
                             lambda: events)
        new_counts_image = counts_image  # this never changes
        if times_image is not None:
            new_times_image = tf.cond(rand > 0.5,
                                      lambda: compute_time_image(events, shape, times_image.dtype),
                                      lambda: times_image)
        else:
            new_times_image = None

    return new_times_image, new_counts_image, new_prev_image, new_next_image, new_events


def random_flip_left_right(images, events):
    with tf.name_scope('random_flip_left_right'):
        in_w = tf.shape(images)[1]

        def flip_events(_events):
            _events_x = tf.expand_dims(_events[..., 0], -1)
            _events_ytsp = _events[..., 1:]
            _events_x = tf.cast(in_w - 1, _events.dtype) - _events_x
            return tf.concat([_events_x, _events_ytsp], -1)

        rand = tf.random_uniform([], dtype=tf.float32)
        flip_images = tf.cond(rand > 0.5,
                              lambda: tf.image.flip_left_right(images),
                              lambda: images)
        flip_events = tf.cond(rand > 0.5,
                              lambda: flip_events(events),
                              lambda: events)

        return flip_images, flip_events


def random_flip_up_down(images, events):
    with tf.name_scope('random_flip_up_down'):
        in_h = tf.shape(images)[0]

        def flip_events(_events):
            _events_x = tf.expand_dims(_events[..., 0], -1)
            _events_y = tf.expand_dims(_events[..., 1], -1)
            _events_tsp = _events[..., 2:]
            _events_y = tf.cast(in_h - 1, _events.dtype) - _events_y
            return tf.concat([_events_x, _events_y, _events_tsp], -1)

        rand = tf.random_uniform([], dtype=tf.float32)
        flip_images = tf.cond(rand > 0.5,
                              lambda: tf.image.flip_up_down(images),
                              lambda: images)
        flip_events = tf.cond(rand > 0.5,
                              lambda: flip_events(events),
                              lambda: events)

        return flip_images, flip_events


def rotate_partial(images, partial_mask, interpolation='NEAREST', minval=-30, maxval=30):
    with tf.name_scope('rotate_partial'):
        in_h, in_w = tf.shape(images)[0], tf.shape(images)[1]
        random_angle = tf.random_uniform([],
                                         minval=minval,
                                         maxval=maxval,
                                         dtype=tf.float32)
        random_angle = random_angle * np.pi / 180.

        images_rotated = tf.contrib.image.rotate(images,
                                                 random_angle,
                                                 interpolation=interpolation)

        # With partial rotation, we don't rotate the events but keep
        # track of a mask identifying the events to keep and rotate it
        # instead
        mask = get_mask(in_h, in_w, partial_mask)
        mask_rotated = tf.contrib.image.rotate(mask, random_angle,
                                               interpolation='NEAREST')

        return images_rotated, mask_rotated, random_angle


def rotate(images, events, interpolation='NEAREST', minval=-30, maxval=30):
    with tf.name_scope('rotate'):
        in_h, in_w = tf.shape(images)[0], tf.shape(images)[1]
        random_angle = tf.random_uniform([],
                                         minval=minval,
                                         maxval=maxval,
                                         dtype=tf.float32)
        random_angle = random_angle * np.pi / 180.

        images_rotated = tf.contrib.image.rotate(images,
                                                 random_angle,
                                                 interpolation=interpolation)

        events_x = tf.cast(tf.expand_dims(events[..., 0], -1), tf.float32)
        events_y = tf.cast(tf.expand_dims(events[..., 1], -1), tf.float32)
        events_tsp = events[..., 2:]

        # Compute the center of the events cloud
        xc = (tf.reduce_max(events_x) - tf.reduce_min(events_x)) / 2
        yc = (tf.reduce_max(events_y) - tf.reduce_min(events_y)) / 2

        cos_angle = tf.cos(random_angle)
        sin_angle = tf.sin(random_angle)

        # Apply counterclockwise rotation
        x_rot = ((events_x - xc) * cos_angle) \
                + ((events_y - yc) * sin_angle) + xc
        y_rot = - ((events_x - xc) * sin_angle) \
                + ((events_y - yc) * cos_angle) + yc

        x_rot = tf.cast(tf.math.rint(x_rot), events.dtype)
        y_rot = tf.cast(tf.math.rint(y_rot), events.dtype)

        # Translate events
        inside_x = tf.math.logical_and(x_rot >= 0.0, x_rot < tf.cast(in_w, x_rot.dtype))
        inside_y = tf.math.logical_and(y_rot >= 0.0, y_rot < tf.cast(in_h, x_rot.dtype))
        inside = tf.math.logical_and(inside_x, inside_y)
        inside = tf.squeeze(inside, -1)

        # Select events inside the crop and place the top-left most
        # event in (0, 0)
        x_rot = tf.boolean_mask(x_rot, inside)
        y_rot = tf.boolean_mask(y_rot, inside)
        tsp_rot = tf.boolean_mask(events_tsp, inside)

        # Merge events back
        events_rotated = tf.concat([x_rot, y_rot, tsp_rot], -1)

        return images_rotated, events_rotated


def random_crop_partial(images, partial_mask, shape):
    with tf.name_scope('random_crop_partial'):
        out_h, out_w = shape[0], shape[1]
        in_h, in_w = tf.shape(images)[0], tf.shape(images)[1]

        # Select the top-left corner of the crop area at random
        left = tf.random_uniform(dtype=tf.int32,
                                 minval=0,
                                 maxval=tf.cast(in_w - out_w, tf.int32),
                                 shape=[])
        top = tf.random_uniform(dtype=tf.int32,
                                minval=0,
                                maxval=tf.cast(in_h - out_h, tf.int32),
                                shape=[])

        # Crop images
        images_cropped = tf.image.crop_to_bounding_box(images,
                                                       offset_height=top,
                                                       offset_width=left,
                                                       target_height=out_h,
                                                       target_width=out_w)

        # Compute the bounding box mask
        inside_square = tf.ones(shape, dtype=tf.uint8)
        inside_mask = tf.pad(inside_square, [[top, in_h - (top + out_h)],
                                             [left, in_w - (left + out_w)]])

        mask = get_mask(in_h, in_w, partial_mask)
        cropped_mask = mask * inside_mask

        return images_cropped, cropped_mask, [top, left, tf.constant(out_h), tf.constant(out_w)]


def random_crop(images, events, shape):
    with tf.name_scope('random_crop'):
        out_h, out_w = shape[0], shape[1]
        in_h, in_w = tf.shape(images)[0], tf.shape(images)[1]

        events_x = tf.expand_dims(events[..., 0], -1)
        events_y = tf.expand_dims(events[..., 1], -1)
        events_tsp = events[..., 2:]

        # Select the top-left corner of the crop area at random
        left = tf.random_uniform(dtype=tf.int32,
                                 minval=0,
                                 maxval=tf.cast(in_w - out_w, tf.int32),
                                 shape=[])
        top = tf.random_uniform(dtype=tf.int32,
                                minval=0,
                                maxval=tf.cast(in_h - out_h, tf.int32),
                                shape=[])

        # Crop images
        images_cropped = tf.image.crop_to_bounding_box(images,
                                                       offset_height=top,
                                                       offset_width=left,
                                                       target_height=out_h,
                                                       target_width=out_w)

        # Select events inside the crop area
        left = tf.cast(left, events_x.dtype)
        top = tf.cast(top, events_y.dtype)
        inside_x = tf.math.logical_and(events_x >= left, events_x < left + out_w)
        inside_y = tf.math.logical_and(events_y >= top, events_y < top + out_h)
        inside = tf.math.logical_and(inside_x, inside_y)
        inside = tf.squeeze(inside, -1)

        # Select events inside the crop and place the top-left most
        # event in (0, 0)
        events_x = tf.boolean_mask(events_x, inside) - left
        events_y = tf.boolean_mask(events_y, inside) - top
        events_tsp = tf.boolean_mask(events_tsp, inside)

        # Merge events back
        events_cropped = tf.concat([events_x, events_y, events_tsp], -1)

        return images_cropped, events_cropped


def center_crop(images, events, shape):
    with tf.name_scope('center_crop'):
        out_h, out_w = shape[0], shape[1]
        in_h, in_w = tf.shape(images)[0], tf.shape(images)[1]

        top = tf.cast((in_h-out_h) / 2, tf.int32)
        left = tf.cast((in_w-out_w) / 2, tf.int32)

        events_x = tf.expand_dims(events[..., 0], -1)
        events_y = tf.expand_dims(events[..., 1], -1)
        events_tsp = events[..., 2:]

        # Crop images
        images_cropped = tf.image.crop_to_bounding_box(images,
                                                       offset_height=top,
                                                       offset_width=left,
                                                       target_height=out_h,
                                                       target_width=out_w)

        # Select events inside the crop area
        left = tf.cast(left, events_x.dtype)
        top = tf.cast(top, events_y.dtype)
        inside_x = tf.math.logical_and(events_x >= left, events_x < left + out_w)
        inside_y = tf.math.logical_and(events_y >= top, events_y < top + out_h)
        inside = tf.math.logical_and(inside_x, inside_y)
        inside = tf.squeeze(inside, -1)

        # Select events inside the crop and place the top-left most
        # event in (0, 0)
        crop_events_x = tf.boolean_mask(events_x, inside) - left
        crop_events_y = tf.boolean_mask(events_y, inside) - top
        crop_events_tsp = tf.boolean_mask(events_tsp, inside)

        # Merge events back
        events_cropped = tf.concat([crop_events_x, crop_events_y, crop_events_tsp], -1)

        return images_cropped, events_cropped


