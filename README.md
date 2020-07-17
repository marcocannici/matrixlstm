# A Differentiable Recurrent Surface for Asynchronous Event-Based Data

Code for the ECCV2020 paper "A Differentiable Recurrent Surface for Asynchronous Event-Based Data"<br>
Authors: Marco Cannici, Marco Ciccone, Andrea Romanoni, Matteo Matteucci

### Citing:
If you use Matrix-LSTM for research, please cite our accompanying ECCV2020 paper:
```
@InProceedings{Cannici_2020_ECCV,
    author = {Cannici, Marco and Ciccone, Marco and Romanoni, Andrea and Matteucci, Matteo},
    title = {A Differentiable Recurrent Surface for Asynchronous Event-Based Data},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {August},
    year = {2020}
}
```

## Project Structure
The code is organized in two folders:
- `classification/` containing PyTorch code for N-Cars and N-Caltech101 experiments
- `opticalflow/` containing TensorFlow code for MVSEC experiments (code based on EV-FlowNet repository)

**Note:** the naming convention used within the code is not exactly the same as the one used in the paper. In particular, the
`groupByPixel` operation is named `group_rf_bounded` in the code (i.e., _group by receptive field_, since it also 
supports receptive fields larger than `1x1`), while the `groupByTime` operation is named `intervals_to_batch`.

## Requirements
We provide a Dockerfile for both codebases in order to replicate the environments we used to run the paper experiments. 
In order to build and run the containers, the following packages are required:

- Docker CE - version 18.09.0 (build 4d60db4)
- NVIDIA Docker - version 2.0

If you have installed the latest version, you may need to modify the .sh files substituting:
- `nvidia-docker run` with `docker run`
- `--runtime=nvidia` with `--gpus=all`

You can verify which command works for you by running:<br>
- (scripts default) `nvidia-docker run -ti --rm --runtime=nvidia -t nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 nvidia-smi`
- `docker run -ti --rm --gpus=all -t nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 nvidia-smi`

You should be able to see the output of `nvidia-smi`

## Run Experiments
Details on how to run experiments are provided in separate README files contained in the classification 
and optical flow sub-folders:
- [Classification Experiments](classification)
- [Optical Flow Experiments](opticalflow)

**Note:** using Docker is not mandatory, but it will allow you to automate the process of installing dependencies and 
building CUDA kernels, all within a safe environment that won't modify any of your previous installations. 
Please, read the `Dockerfile` and `requirements.yml` files contained inside the `<classification or opticalflow>/docker/` 
subfolders if you want to perform a standard conda/pip installation (you just need to manually run all `RUN` commands).
