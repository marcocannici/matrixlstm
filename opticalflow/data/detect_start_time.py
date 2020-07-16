#!/usr/bin/env python

import math
import os
import cv2
import argparse

import rospy
from rosbag import Bag
from cv_bridge import CvBridge

import cv2
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", dest="bag",
                        help="Path to ROS bag.",
                        required=True)
    parser.add_argument("--left_image_0",
                        help="Path to 'left_image_00000.png'")
    parser.add_argument("--right_image_0",
                        help="Path to 'right_image_00000.png'")
    parser.add_argument("--start_time",
                        help="A start time to check",
                        type=float,
                        default=None)
    args = parser.parse_args()

    bridge = CvBridge()
    bag = Bag(args.bag)
    t_start = bag.get_start_time()
    n_msgs = 0

    if args.left_image_0 is None and args.right_image_0 is None:
        raise ValueError("You must provide either left_image_0 or right_image_0")
    isLeft = args.left_image_0 is not None
    ref_img = args.left_image_0 if isLeft else args.right_image_0
    print("Using ref image %s" % ref_img)
    ref_image_0 = cv2.imread(ref_img)[:, :, 0]
    assert ref_image_0 is not None

    topic = '/davis/left/image_raw' if isLeft else '/davis/right/image_raw'
    if args.start_time:
        topic, msg, t = next(bag.read_messages(topics=[topic],
                                               start_time=rospy.Time(args.start_time + t_start)))
        width = msg.width
        height = msg.height
        image = np.asarray(bridge.imgmsg_to_cv2(msg, msg.encoding))
        image = np.reshape(image, (height, width))

        print("Correct? {}".format(bool(np.all(np.isclose(image, ref_image_0)))))

    else:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            n_msgs += 1
            width = msg.width
            height = msg.height
            image = np.asarray(bridge.imgmsg_to_cv2(msg, msg.encoding))
            image = np.reshape(image, (height, width))

            if bool(np.all(np.isclose(image, ref_image_0))):
                print("Ref image found at {} ({} from start)".format(t.to_sec(), t.to_sec()-t_start))
                return

        print("Processed {} images, ref not found!".format(n_msgs))


if __name__ == "__main__":
    main()