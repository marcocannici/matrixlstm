#!/usr/bin/env python
import os
import tensorflow as tf
from accum_training import create_accum_train_op

from losses import *
from model import matrixlstm_model
from vis_utils import *


class EVFlowNetMatrixLSTM:
    def __init__(self,
                 args,
                 events_loader,
                 lengths_loader,
                 event_image_loader,
                 prev_img_loader,
                 next_img_loader,
                 n_ima,
                 rot_angle=None,
                 crop_bbox=None,
                 is_training=True,
                 weight_decay_weight=1e-4):
        self._args = args
        self._restore_path = self._args.restore_path

        self._events_loader = events_loader
        self._lengths_loader = lengths_loader
        self._event_img_loader = event_image_loader
        self._image_shape = [self._args.image_height, self._args.image_width]

        self._prev_img_loader = prev_img_loader
        self._next_img_loader = next_img_loader

        self._n_ima = n_ima
        self._base_channels = self._args.base_channels
        self._weight_decay_weight = weight_decay_weight
        self._is_training = is_training
        self._rot_angle = rot_angle
        self._crop_bbox = crop_bbox

    def _build_graph(self, global_step):
        # Model
        with tf.variable_scope('vs'):
            augment_within_network = self._is_training and \
                                     (self._rot_angle is not None or
                                      self._crop_bbox is not None)
            matrixlstm_in_shape = self._args.matrixlstm_in_shape if augment_within_network else self._image_shape
            flow_dict, event_image = matrixlstm_model(
                self._events_loader,
                self._lengths_loader,
                self._event_img_loader,
                self._args.batch_size,
                self._args.matrixlstm_mode,
                self._args.matrixlstm_channels,
                self._args.matrixlstm_force_channels,
                matrixlstm_in_shape,
                self._args.matrixlstm_region_shape,
                self._args.matrixlstm_region_stride,
                self._args.matrixlstm_actfn,
                self._args.matrixlstm_batchnorm,
                self._args.matrixlstm_maxrf,
                self._args.matrixlstm_keep_last,
                self._args.matrixlstm_time_mode,
                self._args.matrixlstm_norm_relative,
                self._args.matrixlstm_add_step,
                self._args.matrixlstm_add_coords,
                self._args.matrixlstm_frame_intervals,
                self._args.matrixlstm_frame_intervals_mode,
                self._args.matrixlstm_add_terminator_mode,
                self._args.matrixlstm_k_init,
                self._args.matrixlstm_add_selayer,
                self._is_training,
                do_batch_norm=not self._args.no_batch_norm,
                use_conv_bias=not self._args.no_conv_bias,
                aug_rot_angle=self._rot_angle,
                aug_crop_bbox=self._crop_bbox,
                base_channels=self._args.base_channels,
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
            tf.summary.scalar('lr', learning_rate)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
        return flow_dict, loss, optimizer, next_image_warped, event_image

    def get_grads_mult(self):
        if self._args.matrixlstm_grad_mult == 1.0:
            return None

        mult_dict = {}
        for var in tf.trainable_variables():
            if 'lstm' in var.name.lower():
                print("Applying multiplier x{} to {}".format(self._args.matrixlstm_grad_mult,
                                                             var.name))
                mult_dict.update({var.name: self._args.matrixlstm_grad_mult})
        return mult_dict

    @staticmethod
    def clip_grad(grad, max_norm):
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.contrib.slim.learning.clip_ops.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.contrib.slim.learning.clip_ops.clip_by_norm(grad, max_norm)
        return grad

    def transform_grads_fn(self, grads):
        grad_multipliers = self.get_grads_mult()
        if grad_multipliers:
            with tf.name_scope('multiply_grads'):
                grads = tf.contrib.slim.learning.multiply_gradients(grads, grad_multipliers)

        grads_and_vars = []
        with tf.name_scope('clip_grads'):
            for grad, var in grads:
                if 'bias' in var.name and self._args.clip_bias_gradient_norm > 0:
                    print("Clipping {} to {}".format(var.name,
                                                     self._args.clip_bias_gradient_norm))
                    clip_grad = self.clip_grad(grad, self._args.clip_bias_gradient_norm)
                    grads_and_vars.append((clip_grad, var))
                elif '_bn' in var.name and self._args.clip_bn_gradient_norm > 0:
                    print("Clipping {} to {}".format(var.name,
                                                     self._args.clip_bn_gradient_norm))
                    clip_grad = self.clip_grad(grad, self._args.clip_bn_gradient_norm)
                    grads_and_vars.append((clip_grad, var))
                elif self._args.clip_gradient_norm > 0:
                    print("Clipping {} to {}".format(var.name,
                                                     self._args.clip_gradient_norm))
                    clip_grad = self.clip_grad(grad, self._args.clip_gradient_norm)
                    grads_and_vars.append((clip_grad, var))
                else:
                    grads_and_vars.append((grad, var))
        return grads_and_vars

    def train(self):
        print("Starting training.")
        global_step = tf.train.get_or_create_global_step()

        #load data
        flow_dict, self._loss, self._optimizer, \
        next_image_warped, event_image = self._build_graph(global_step)

        final_flow = flow_dict['flow3']

        if self._args.optimize_every <= 1:
            train_op = tf.contrib.training.create_train_op(total_loss=self._loss,
                                                           optimizer=self._optimizer,
                                                           global_step=global_step,
                                                           summarize_gradients=True,
                                                           transform_grads_fn=self.transform_grads_fn)
        else:
            train_op = create_accum_train_op(total_loss=self._loss,
                                             optimizer=self._optimizer,
                                             optimize_every=self._args.optimize_every,
                                             global_step=global_step,
                                             summarize_gradients=True,
                                             transform_grads_fn=self.transform_grads_fn)

        # Visualization for Tensorboard.
        with tf.device('/cpu:0'):
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
                             event_image[..., :3],
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
            if step % 100000 == 0:
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
