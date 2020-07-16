## ResNet + Matrix-LSTM

This folder contains code to reproduce classification experiments

## Requirements

#### Docker Images

In order to reproduce the experiments you have to first build the following Docker image:

*matrixlstm/classification*
- Move inside the `docker/` sub-folder:<br> `cd classification/docker`
- Build the image:<br> `docker build -t matrixlstm/classification .`


### Data

Experiments were performed on the N-Cars and N-Caltech101 datasets.
- Download the N-Caltech101 split which saw used to train
  [EST](https://github.com/uzh-rpg/rpg_event_representation_learning):
    - Move to the `data` folder
    - Run the script: `sh download_ncaltech.sh`
- Download the N-Cars dataset:
    - Go to the N-Cars download page: `https://www.prophesee.ai/dataset-n-cars/`
    - Place the `Prophesee_Dataset_n_cars.zip` zip inside the data folder 
    - Run `sh extract_ncars.sh`

## Training

We provide different configuration files for each experiment within the `classification/configs` folder.

Select the configuration file and then run the experiment using Docker with the following command 
from within the `classification` folder:

```
sh train_resnet.sh <gpu_id> -c configs/<config_name>.yaml
```

Make sure to substitute `<gpu_id>` and `<config_name>` in the previous command. 

## Testing

The training procedure automatically performs early stopping on a validation dataset, creating one if it does not exist,
and computes the test set accuracy using the best epoch on validation.

## Profiling

You can profile the different configurations on a dataset (N-Cars is used in the paper) using the `profile_matrixlstm.sh` script.

Use the command:
```
sh profile_matrixlstm.sh <gpu_id> -c configs/<config_name>.yaml --val_perc 0.0
```


You can also profile configurations on synthetic data (with variable density) using the `profile_synthetic.sh` script.

Use the command:
```
sh profile_synthetic.sh <gpu_id> -c <config_path> --density <density> --output_file <out_csv_path> --training True
```

Note that profiling does not require a pre-trained model since time performance does not depend on the learned event integration mechanism.