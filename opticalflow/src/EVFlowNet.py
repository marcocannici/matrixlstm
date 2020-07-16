#!/usr/bin/env python
import os
import math
import tensorflow as tf
from accum_training import create_accum_train_op
import numpy as np

from losses import *
from model import model
from vis_utils import *

class EVFlowNet():
    def __init__(self,
                 args,
                 event_img_loader,
                 prev_img_loader,
                 next_img_loader,
                 n_ima,
                 is_training=True,
                 weight_decay_weight=1e-4):
        self._args = args
        self._restore_path = self._args.restore_path

        self._event_img_loader = event_img_loader
        self._prev_img_loader = prev_img_loader
        self._next_img_loader = next_img_loader

        self._n_ima = n_ima
        self._base_channels = self._args.base_channels
        self._weight_decay_weight = weight_decay_weight
        self._is_training = is_training

    def _build_graph(self, global_step):
        #Model
        with tf.variable_scope('vs'):
            flow_dict = model(self._event_img_loader,
                              self._is_training,
                              do_batch_norm=not self._args.no_batch_norm,
                              use_conv_bias=not self._args.no_conv_bias,
                              base_channels=self._base_channels,
                              deconv3_channels=self._args.deconv3_channels,
                              count_only=self._args.count_only,
                              time_only=self._args.time_only)

        with tf.name_scope('loss'):
            # Weight decay loss.
            if not self._args.no_weight_decay:
                with tf.name_scope('weight_decay'):
                    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vs/')
                    for v in var:
                        wd_loss = tf.multiply(tf.nn.l2_loss(v), self._weight_decay_weight)
                        tf.add_to_collection('wd_loss', wd_loss)
                    weight_decay_loss = tf.add_n(tf.get_collection('wd_loss'))
                    tf.summary.scalar('weight_decay_loss', weight_decay_loss)
            else:
                weight_decay_loss = 0.0

            # Smoothness loss.
            smoothness_loss = 0
            for i in range(len(flow_dict)):
                smoothness_loss_i = compute_smoothness_loss(flow_dict["flow{}".format(i)])
                smoothness_loss += smoothness_loss_i
                tf.summary.scalar('smoothness_loss_flow%d' % i, smoothness_loss_i / 4.)
            smoothness_loss *= self._args.smoothness_weight / 4.
            tf.summary.scalar("smoothness_loss", smoothness_loss)

            # Photometric loss.
            photometric_loss, photometric_losses = compute_photometric_loss(self._prev_img_loader,
                                                                            self._next_img_loader,
                                                                            self._event_img_loader,
                                                                            flow_dict)
            for i, photo_loss_i in enumerate(photometric_losses):
                tf.summary.scalar('photometric_loss_flow%d' % i, photo_loss_i)
            tf.summary.scalar('photometric_loss', photometric_loss)

            # Warped next image for debugging.
            next_image_warped = warp_images_with_flow(self._next_img_loader,
                                                      flow_dict['flow3'])

            loss = weight_decay_loss + photometric_loss + smoothness_loss
            tf.summary.scalar('total_loss', loss)
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            learning_rate = tf.train.exponential_decay(self._args.initial_learning_rate,
                                                       global_step,
                                                       self._args.learning_step_decay * self._n_ima \
                                                       / (self._args.batch_size * self._args.optimize_every),
                                                       self._args.learning_rate_decay,
                                                       staircase=True)
            # learning_rate = tf.train.piecewise_constant(global_step,
            #                                             [20000, 60000],
            #                                             [5e-5, 0.5 * 5e-5, 0.25 * 5e-5])

            tf.summary.scalar('lr', learning_rate)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
        return flow_dict, loss, optimizer, next_image_warped

    def train(self):
        print("Starting training.")
        global_step = tf.train.get_or_create_global_step()

        #load data
        flow_dict, self._loss, self._optimizer, next_image_warped = self._build_graph(global_step)

        final_flow = flow_dict['flow3']

        if self._args.optimize_every <= 1:
            train_op = tf.contrib.training.create_train_op(total_loss=self._loss,
                                                           optimizer=self._optimizer,
                                                           global_step=global_step,
                                                           summarize_gradients=True)
        else:
            train_op = create_accum_train_op(total_loss=self._loss,
                                             optimizer=self._optimizer,
                                             optimize_every=self._args.optimize_every,
                                             global_step=global_step,
                                             summarize_gradients=True)

        # Visualization for Tensorboard.
        with tf.device('/cpu:0'):
            event_img = tf.expand_dims(self._event_img_loader[:, :, :, 0] + \
                                       self._event_img_loader[:, :, :, 1], axis=3)

            event_time_img = tf.reduce_max(self._event_img_loader[:, :, :, 2:4],
                                           axis=3,
                                           keepdims=True)
            flow_rgb, flow_norm, flow_ang_rad = flow_viz_tf(final_flow)

            image_error = tf.abs(next_image_warped - self._prev_img_loader)
            image_error = tf.clip_by_value(image_error, 0., 20.)

            # Color wheel to visualize the flow directions.
            color_wheel_rgb = draw_color_wheel_tf(self._args.image_width, self._args.image_height)

            # Appending letters to each title allows us to control the order of display.
            tf.summary.image("a-Color wheel",
                             color_wheel_rgb,
                             max_outputs=1)
            tf.summary.image("b-Flow",
                             flow_rgb,
                             max_outputs=self._args.batch_size)
            tf.summary.image("c-Event time image", event_time_img,
                             max_outputs=self._args.batch_size)
            tf.summary.image('d-Warped_next_image', next_image_warped,
                             max_outputs=self._args.batch_size)
            tf.summary.image("e-Prev image",
                             self._prev_img_loader,
                             max_outputs=self._args.batch_size)
            tf.summary.image("e-Next image",
                             self._next_img_loader,
                             max_outputs=self._args.batch_size)
            tf.summary.image("f-Image error",
                             image_error,
                             max_outputs=self._args.batch_size)
            tf.summary.image("g-Event image",
                             event_img,
                             max_outputs=self._args.batch_size)

        writer = tf.summary.FileWriter(self._args.summary_path)

        debug_rate = 5000
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Adds an additional saver to make sure that
        # specific checkpoints are not deleted during training
        checkpoint_saver = tf.train.Saver()

        def train_step_fn(_sess, _train_op, _global_step, _train_step_kwargs):
            ret_val = tf.contrib.slim.learning.train_step(_sess,
                                                          _train_op,
                                                          _global_step,
                                                          _train_step_kwargs)
            step = _sess.run(_global_step)
            if step % 300000 == 0:
                checkpoint_saver.save(_sess, os.path.join(self._args.load_path,
                                                          "checkpoint%d.ckpt" % step))
            return ret_val

        init_fn = None
        if self._restore_path is not None:
            if '.ckpt' in self._restore_path:
                model_path = self._restore_path
            else:
                model_path = tf.train.latest_checkpoint(self._restore_path)

            if model_path:
                variables_to_restore = tf.contrib.slim.get_variables_to_restore()
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                    model_path, variables_to_restore)

        tf.contrib.slim.learning.train(
            train_op=train_op,
            init_fn=init_fn,
            logdir=self._args.load_path,
            number_of_steps=self._args.number_of_steps,
            log_every_n_steps=debug_rate,
            save_summaries_secs=240.,
            summary_writer=writer,
            save_interval_secs=240.,
            train_step_fn=train_step_fn)
