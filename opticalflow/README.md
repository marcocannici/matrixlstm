## EV-FlowNet + Matrix-LSTM

This folder contains code to reproduce optical flow experiments

## Reference
Code in this folder is a fork of the official EV-FlowNet repository.
A LICENSE.md file is included to comply with the EV-FlowNet code license (condition 1).

EV-FlowNet paper: [Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney, Kostas Daniilidis.
                   "EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras",
                   Proceedings of Robotics: Science and Systems, 2018. DOI: 10.15607/RSS.2018.XIV.062.
                  ](http://www.roboticsproceedings.org/rss14/p62.html)

## Requirements

#### Docker Images

In order to reproduce the experiments you have to first build the following Docker images:

*matrixlstm/opticalflow-experiments*
- Move inside the `docker/experiments` sub-folder:<br> `cd opticalflow/docker/experiments`
- Build the image:<br> `docker build -t matrixlstm/opticalflow-experiments .`

*matrixlstm/opticalflow-data*
- Move inside the `docker/data` sub-folder:<br> `cd opticalflow/docker/data`
- Build the image:<br> `docker build -t matrixlstm/opticalflow-data .`

#### Compile Custom CUDA kernels

- Move into the `src/extensions` sub-folder:<br> `cd opticalflow/src/extensions`
- Compile the kernels within a Docker container (this creates binaries inside the `extensions` folder):<br>
    `sh build_kernels_docker.sh`

<!---
```
docker run -it --rm \
   -v ${PWD}:/exp \
   -t tensorflow/tensorflow:1.12.0-devel-gpu \
   sh build_kernels.sh
```
-->

### Data

As in the EV-FlowNet paper, data need to be first converted into TfRecord files for training and testing.
Since our Matrix-LSTM layer needs to directly process events, we provide a modified conversion script, 
`extract_rosbag_events_to_tf.py`, that keeps track of the events associated to each pair of gray-scale images.

In order to generate small TfRecords in size we only save the position of the first and last event in each sub-sequence
within the record and then load the events from a separate HDF5 file at run-time.

Note: Since the original code does not provide the exact start_time and end_time for indoor_flying sequences, used for testing, 
we compute them using the `detect_start_time.py` script, which identifies the rosbag message associated with 
an input image. We downloaded the test split provided in the original repository and detected the time of the first and 
last image in each indoor_flying sequence.

*Generate TFRecords*
- Move to the `data` directory: `cd opticalflow/data`
- Download the MVSEC rosbag files:<br> `sh download_mvsec_rosbags.sh`
- Convert the dataset:<br>`sh extract_all_bags_events_docker.sh`

<!---
```
docker run -it --rm \
   -v ${PWD}:/exp \
   -t paper6097/opticalflow-data \
   sh extract_all_bags_events.sh
```
-->

- You can now optionally delete the .bag files contained in `opticalflow/data/download`

*Download ground truth files*

- Manually download the HDF5 dataset files and place (and rename) them in the correct subfolder:
    - [indoor_flying1_data.hdf5](https://drive.google.com/open?id=18lISHaEtIHxTETuuNHBRHPTtsVhq29l_) 
    &rightarrow; `data/extracted/indoor_flying1_events/indoor_flying1_events_data.hdf5`
    - [indoor_flying2_data.hdf5](https://drive.google.com/open?id=1BD7cVNUdDgvqTYPdzxy9Tcyl1FrjhrjF) 
    &rightarrow; `data/extracted/indoor_flying2_events/indoor_flying2_events_data.hdf5`
    - [indoor_flying3_data.hdf5](https://drive.google.com/open?id=1Q7Mm_oZrSI_cTRktY8LcUaLSrGUmAERJ) 
    &rightarrow; `data/extracted/indoor_flying3_events/indoor_flying3_events_data.hdf5`
    - [outdoor_day1_data.hdf5](https://drive.google.com/open?id=1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0)
    &rightarrow; `data/extracted/outdoor_day1_aug12_skip1/outdoor_day1_aug12_skip1_data.hdf5`
    - [outdoor_day2_data.hdf5](https://drive.google.com/open?id=1fu9GhjYcET00mMN-YbAp3eBK1YMCd3Ox)
    &rightarrow; `data/extracted/outdoor_day2_aug12_skip1/outdoor_day2_aug12_skip1_data.hdf5`
    
    
- Manually download ground truth optical flow files for indoor_flying sequences:
    - [indoor_flying1_gt_flow_dist.npz](https://drive.google.com/open?id=1AmwVGm5oH2Fk-XPcaIJNzZ9hKV8U7Fs3) 
    &rightarrow; `data/download/indoor_flying1_gt_flow_dist.npz`
    - [indoor_flying2_gt_flow_dist.npz](https://drive.google.com/open?id=1X6Fa0ZuEwRTEshQhKyekKYXcDZYoe_eh) 
    &rightarrow; `data/download/indoor_flying2_gt_flow_dist.npz`
    - [indoor_flying3_gt_flow_dist.npz](https://drive.google.com/open?id=1nkjEIL_uMSQXHKP25LWXbgV2shb4dmxL) 
    &rightarrow; `data/download/indoor_flying3_gt_flow_dist.npz`
    
Note: All the links point to the original dataset folder, here we provide direct links to the required files 
to ease the download process. Files can also be downloaded on a headless system using the 
[drive](https://github.com/odeke-em/drive) utility.

## Training

We provide different configuration files for each experiment within the `opticalflow/configs` folder.

Select the configuration file and then run the experiment using Docker with the following command 
from within the `opticalflow` folder:

```
sh run_matrixlstm_training.sh <gpu_id> -c <config_path>
```

<!---
```
NV_GPU=<gpu_id> nvidia-docker run -it --rm \
   --runtime=nvidia \
   --shm-size=8G \
   -v ${PWD}:/exp \
   -e PYTHONPATH=/exp \
   -t matrixlstm/opticalflow-experiments \
   python src/train_matrixlstm.py -c configs/<config_name>.yaml
```
-->

Make sure to substitute `<gpu_id>` and `<config_name>` in the previous command. 

### Testing

To compute the test performance of a trained model located in `data/log/saver/<model_name>`, run the following command:
```
sh run_matrixlstm_test_all.sh <gpu_id> <model_name> -c <config_path>
```

If you want to test a specific checkpoint (e.g., `data/log/saver/<model_name>/last.ckpt` in pre-trained models), run:
```
sh run_matrixlstm_test_all.sh <gpu_id> <model_name>/last.ckpt -c <config_path>
```

You can optionally provide the `--save_test_output` option to save predicted optical flow videos.
Videos will be saved in the `<model_name>` training path, under the `pred_flow` directory. 

Note: A file for each test sequence containing the test scores is saved within the checkpoint directory.
