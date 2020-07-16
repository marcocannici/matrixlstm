#!/bin/bash

ref_images_dir=mvsec_bags/ref_images/

flying_bag_dir=mvsec_bags/indoor_flying/
num_indoor_flying_bags=3
bagnames[1]=indoor_flying1
bagnames[2]=indoor_flying2
bagnames[3]=indoor_flying3

for bag_iter in `seq 1 $num_indoor_flying_bags`;
do
    echo Checking indoor_flying$bag_iter
    python detect_start_time.py --bag $flying_bag_dir${bagnames[$bag_iter]}.bag --left_image_0 bagnames[$bag_iter]_left_0.png
done

outdoor_day_bag_dir=mvsec_bags/outdoor_day/
num_outdoor_day_bags=2
outdoor_day_bagnames[1]=outdoor_day1
outdoor_day_bagnames[2]=outdoor_day2

for bag_iter in `seq 1 $num_outdoor_day_bags`;
do
    echo Checking indoor_flying$bag_iter
    python detect_start_time.py --bag $outdoor_day_bag_dir${outdoor_day_bagnames[$bag_iter]}.bag --left_image_0 outdoor_day_bagnames[$bag_iter]_left_0.png
done

outdoor_night_bag_dir=mvsec_bags/outdoor_night/
num_outdoor_night_bags=3
outdoor_night_bagnames[1]=outdoor_night1
outdoor_night_bagnames[2]=outdoor_night2
outdoor_night_bagnames[3]=outdoor_night3

for bag_iter in `seq 1 $num_outdoor_night_bags`;
do
    echo Checking outdoor_night_bagnames
    python detect_start_time.py --bag $outdoor_night_bag_dir${outdoor_night_bagnames[$bag_iter]}.bag --left_image_0 outdoor_night_bagnames[$bag_iter]_left_0.png
done