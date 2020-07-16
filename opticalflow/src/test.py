#!/usr/bin/env python
import os
import time

import tensorflow as tf
import numpy as np

from config import *
from data_loader import get_loader
from eval_utils import *
from losses import *
from model import *
from vis_utils import *

def drawImageTitle(img, title):
    cv2.putText(img,
                title,
                (60, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=2,
                bottomLeftOrigin=False)
    return img

def to_rb(frame):
    assert frame.shape[-1] == 2

    # creates a red green image by placing an additional empty blue channel
    frame_rg = np.pad(frame, [(0,0), (0,0), (0,1)], mode='constant')
    # move the empty channel to the green position
    frame_rb = frame_rg[..., [0, 2, 1]]
    return frame_rb

def frame_normalize(frame):
    fmax = frame.max()
    fmin = frame.min()
    frame = 255 * (frame - fmin) / (fmax - fmin)
    return frame.astype(np.uint8)

def test(sess,
         args,
         event_image_loader,
         prev_image_loader,
         next_image_loader,
         timestamp_loader):
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('vs'):
        flow_dict = model(event_image_loader,
                          is_training=False,
                          do_batch_norm=not args.no_batch_norm,
                          base_channels=args.base_channels,
                          deconv3_channels=args.deconv3_channels,
                          time_only=args.time_only,
                          count_only=args.count_only)

    photometric_loss, _ = compute_photometric_loss(prev_image_loader,
                                                   next_image_loader,
                                                   event_image_loader,
                                                   flow_dict)
    smoothness_loss = 0
    smoothness_weight = 0.5
    for i in range(len(flow_dict)):
        smoothness_loss += compute_smoothness_loss(flow_dict["flow{}".format(i)])
    smoothness_loss *= smoothness_weight / 4.

    event_image = tf.reduce_sum(event_image_loader[:, :, :, :2], axis=-1, keepdims=True)
    flow_rgb, flow_norm, flow_ang_rad = flow_viz_tf(flow_dict['flow3'])
    color_wheel_rgb = draw_color_wheel_np(args.image_width, args.image_height)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, args.load_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tot_photometric_loss = 0
    tot_smoothness_loss = 0
    max_flow_sum = 0
    min_flow_sum = 0
    iters = 0

    if args.test_plot:
        import cv2
        cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)

    if args.gt_path:
        args.gt_path = args.gt_path.replace("events_", "")
        print("Loading ground truth {}".format(args.gt_path))
        gt = np.load(args.gt_path)
        gt_timestamps = gt['timestamps']
        U_gt_all = gt['x_flow_dist']
        V_gt_all = gt['y_flow_dist']
        print("Ground truth loaded")

        AEE_sum = 0.
        percent_AEE_sum = 0.
        AEE_list = []
        percent_outliers_sum = 0.
        tot_outliers = 0.
        tot_points = 0.

    if args.save_test_output:
        import cv2
        output_flow_list = []
        gt_flow_list = []
        event_image_list = []
        training_paths = args.training_instance.split("/")
        training_path = training_paths[0]
        training_epoch = training_paths[1].replace(".ckpt", "") \
            if len(training_paths) > 1 else "last_checkpoint"
        savepath = os.path.join(args.load_dir, training_path, "pred_flow", training_epoch,
                                "w_skips" if args.test_skip_frames else "wo_skips",
                                args.test_sequence)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

    if args.test_reference_flow:
        print("Loading reference optical flow")
        ref_flow = np.load(args.test_reference_flow)
        ref_flow = ref_flow['output_flows']

    while not coord.should_stop():
        start_time = time.time()
        try:
            flow_dict_np, \
            prev_image, \
            next_image, \
            event_image, \
            image_timestamps, \
            photometric_loss_cpu, \
            smoothness_loss_cpu = sess.run([flow_dict,
                                            prev_image_loader,
                                            next_image_loader,
                                            event_image_loader,
                                            timestamp_loader,
                                            photometric_loss,
                                            smoothness_loss])
        except tf.errors.OutOfRangeError:
            break

        tot_photometric_loss += photometric_loss_cpu
        tot_smoothness_loss += smoothness_loss_cpu

        network_duration = time.time() - start_time
        event_image = np.array(event_image)

        pred_flow = np.squeeze(flow_dict_np['flow3'])

        max_flow_sum += np.max(pred_flow)
        min_flow_sum += np.min(pred_flow)

        event_count_image = np.sum(event_image[..., :2], axis=-1)
        event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
        event_count_image = np.squeeze(event_count_image)

        if args.save_test_output:
            output_flow_list.append(pred_flow)
            event_image_list.append(event_count_image)

        if args.gt_path:
            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all,
                                                        gt_timestamps,
                                                        image_timestamps[0][0],
                                                        image_timestamps[0][1])

            gt_flow = np.stack((U_gt, V_gt), axis=2)

            if args.save_test_output:
                gt_flow_list.append(gt_flow)

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2

            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]

            if args.test_reference_flow:
                ref_flow_mask = np.all(np.isclose(ref_flow[iters], 0, atol=0.01), axis=-1)
                cur_flow_mask = np.all(np.isclose(pred_flow, 0, atol=0.01), axis=-1)
                event_count_image_masked = event_count_image.copy()
                event_count_image_masked[ref_flow_mask] = 0
                event_count_image_masked[cur_flow_mask] = 0
            else:
                event_count_image_masked = event_count_image.copy()

            if np.sum(event_count_image_masked != 0) > 0 or not args.test_reference_flow:
                # Calculate flow error.
                AEE, percent_AEE, n_points, percent_outliers, n_outliers = \
                    flow_error_dense(gt_flow,
                                     pred_flow,
                                     event_count_image_masked,
                                     'outdoor' in args.test_sequence)
                AEE_list.append(AEE)
                AEE_sum += AEE
                percent_AEE_sum += percent_AEE
                percent_outliers_sum += percent_outliers
                tot_outliers += n_outliers
                tot_points += n_points

        iters += 1
        if iters % 100 == 0:
            print('-------------------------------------------------------')
            print('Iter: {}, time: {:f}, run time: {:.3f}s\n'
                  'Mean max flow: {:.2f}, mean min flow: {:.2f}\n'
                  'Smoothness loss: {:.4f}, Photometric loss: {:.4f}'
                  .format(iters, image_timestamps[0][0], network_duration,
                          max_flow_sum / iters, min_flow_sum / iters,

                          tot_smoothness_loss / iters,
                          tot_photometric_loss / iters))
            if args.gt_path:
                print('Mean AEE: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}\n'
                      '%Outliers {}\n'
                      .format(AEE_sum / iters, percent_AEE_sum / iters, n_points,
                              100 * (percent_outliers_sum / iters)))

        # Prep outputs for nice visualization.
        if args.test_plot or args.save_test_output:
            pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            pred_flow_rgb = drawImageTitle(pred_flow_rgb, 'Predicted Flow')

            event_count_image_vis = event_image[..., :2].squeeze(0)
            event_count_image_vis = to_rb(frame_normalize(event_count_image_vis))

            event_time_image_vis = event_image[..., 2:].squeeze(0)
            event_time_image_vis = to_rb(frame_normalize(event_time_image_vis))

            event_time_image_vis = drawImageTitle(event_time_image_vis, 'Timestamp Image')
            event_count_image_vis = drawImageTitle(event_count_image_vis, 'Count Image')

            prev_image = np.squeeze(prev_image)
            prev_image = np.tile(prev_image[..., np.newaxis], [1, 1, 3])

            prev_image = drawImageTitle(prev_image, 'Grayscale Image')

            gt_flow_rgb = np.zeros(pred_flow_rgb.shape)
            errors = np.zeros(pred_flow_rgb.shape)

            gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow - No GT')
            errors = drawImageTitle(errors, 'Flow Error - No GT')

            if args.gt_path:
                errors = np.linalg.norm(gt_flow - pred_flow, axis=-1)
                errors = (errors * 255. / errors.max()).astype(np.uint8)
                errors = np.tile(errors[..., np.newaxis], [1, 1, 3])
                errors[event_count_image == 0] = 0

                if 'outdoor' in args.test_sequence:
                    errors[190:, :] = 0

                gt_flow_rgb = flow_viz_np(gt_flow[...,0], gt_flow[...,1])

                gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow')
                errors = drawImageTitle(errors, 'Flow Error')

            top_cat = np.concatenate([event_count_image_vis, prev_image, pred_flow_rgb], axis=1)
            bottom_cat = np.concatenate([event_time_image_vis, errors, gt_flow_rgb], axis=1)
            cat = np.concatenate([top_cat, bottom_cat], axis=0)
            cat = cat.astype(np.uint8)

            if args.save_test_output:
                impath = os.path.join(savepath, "result_%d.png" % iters)
                cv2.imwrite(impath, cat)
            elif args.test_plot:
                cv2.imshow('EV-FlowNet Results', cat)
                cv2.waitKey(1)

    print('Testing done. ')
    if args.gt_path:
        print('mean AEE {:02f}, mean %AEE {:02f}\n'
              '%Outliers {}\n'
              'Smoothness loss: {:.4f}, Photometric loss: {:.4f}\n'
              .format(AEE_sum / iters, percent_AEE_sum / iters,
                      100 * (percent_outliers_sum / iters),
                      tot_smoothness_loss / iters, tot_photometric_loss / iters))

        if args.test_reference_flow is None:
            suffix = '_skip_frames' if args.test_skip_frames else ''
            if '.ckpt' in args.training_instance:
                suffix += '_' + os.path.basename(args.training_instance)
            else:
                suffix += '_last_ckpt'
            res_path = os.path.join(os.path.dirname(args.load_path),
                                    args.test_sequence + suffix + "_results.txt")
            with open(res_path, 'w') as fp:
                fp.write('mean AEE {:02f}, mean %AEE {:02f}\n'
                         '%Outliers {}\n'
                         'Smoothness loss: {:.4f}, Photometric loss: {:.4f}\n'
                         .format(AEE_sum / iters, percent_AEE_sum / iters,
                                 100 * (percent_outliers_sum / iters),
                                 tot_smoothness_loss / iters, tot_photometric_loss / iters))

    if args.save_test_output:
        images_pattern = os.path.join(savepath, "result_%d.png")
        images_glob = os.path.join(savepath, "result_*.png")
        dest_path = os.path.join(savepath, "flow.mp4")
        os.system("ffmpeg -framerate 24 -start_number 1 -i " + images_pattern + " " + dest_path)
        os.system("rm " + images_glob)

        # filepath = os.path.join(savepath, "output_gt.npz")
        # if args.gt_path:
        #     print('Saving data to {}'.format(filepath))
        #     np.savez(filepath,
        #              output_flows=np.stack(output_flow_list, axis=0),
        #              gt_flows=np.stack(gt_flow_list, axis=0),
        #              event_images=np.stack(event_image_list, axis=0))
        # else:
        #     print('Saving data to {}'.format(filepath))
        #     np.savez(filepath,
        #              output_flows=np.stack(output_flow_list, axis=0),
        #              event_images=np.stack(event_image_list, axis=0))

    coord.request_stop()

def main():
    args = configs()
    args.load_dir = ""+args.load_path
    args.load_path = os.path.join(args.load_path, args.training_instance)
    if not '.ckpt' in args.training_instance:
        args.load_path = tf.train.latest_checkpoint(args.load_path)

    sess = tf.Session()
    event_image_loader, prev_image_loader, next_image_loader, timestamp_loader, n_ima = get_loader(
        args.data_path,
        1,
        args.image_width,
        args.image_height,
        split='test',
        shuffle=False,
        sequence=args.test_sequence,
        skip_frames=args.test_skip_frames,
        gzip=args.gzip)

    if not args.load_path:
        raise Exception("You need to set `load_path` and `training_instance`.")

    print("Read {} images".format(n_ima))
    test(sess,
         args,
         event_image_loader,
         prev_image_loader,
         next_image_loader,
         timestamp_loader)
    sess.close()


if __name__ == "__main__":
    main()
