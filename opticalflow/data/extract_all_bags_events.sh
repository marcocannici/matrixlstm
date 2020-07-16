#!/bin/bash

output_folder=extracted/
mkdir ${output_folder}


###################
## Indoor Flying ##
###################

num_indoor_flying_bags=3
flying_bag_dir=/exp/download/indoor_flying/

indoor_flying_bagnames[1]=indoor_flying1
indoor_flying_bagnames[2]=indoor_flying2
indoor_flying_bagnames[3]=indoor_flying3

indoor_flying_starttimes[1]=5.20545
indoor_flying_starttimes[2]=10.20681
indoor_flying_starttimes[3]=8.2017

indoor_flying_endtimes[1]=69.9894
indoor_flying_endtimes[2]=69.9889
indoor_flying_endtimes[3]=89.9848

for bag_iter in `seq 1 $num_indoor_flying_bags`;
do
    echo Extracting indoor_flying bag$bag_iter
    mkdir ${output_folder}/indoor_flying${bag_iter}_events
    python extract_rosbag_to_tf.py \
        --bag $flying_bag_dir${indoor_flying_bagnames[$bag_iter]}_data.bag \
        --prefix indoor_flying${bag_iter}_events \
        --start_time ${indoor_flying_starttimes[$bag_iter]} \
        --end_time ${indoor_flying_endtimes[$bag_iter]} \
        --max_aug 6 \
        --n_skip 1 \
        --output_folder $output_folder
done


#################
## Outdoor Day ##
#################

num_outdoor_day_bags=2
outdoor_day_bag_dir=/exp/download/outdoor_day/

outdoor_day_bagnames[1]=outdoor_day1
outdoor_day_bagnames[2]=outdoor_day2

outdoor_day_starttimes[1]=3.0
outdoor_day_starttimes[2]=45.0

for bag_iter in `seq 1 $num_outdoor_day_bags`;
do
    echo Extracting outdoor_day bag$bag_iter
    mkdir ${output_folder}/outdoor_day${bag_iter}_aug12_skip1
    python extract_rosbag_to_tf.py \
        --bag $outdoor_day_bag_dir${outdoor_day_bagnames[$bag_iter]}_data.bag \
        --prefix outdoor_day${bag_iter}_aug12_skip1 \
        --start_time ${outdoor_day_starttimes[$bag_iter]} \
        --max_aug 12 \
        --n_skip 1 \
        --output_folder $output_folder
done