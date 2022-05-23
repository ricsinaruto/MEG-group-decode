# MEG-transfer-decoding

First requirements need to be installed.
```
pip install -r requirements.txt
```


## Behaviour
For each run modify ```args.py``` to specify parameters and behaviour, then run ```launch.py``` which calls ```training.py```, which contains the main experimental pipeline:
```
python launch.py
```
Set the function you want to run to True in the ```func``` dictionary in ```args.py```.
The following functions are available:
* **train**: Does training for a specified neural network model and dataset.
* **kernel_network_FIR**: This is the kernel FIR analysis in the paper. Analyse frequency characteristics of kernels in a trained models.
* **save_validation_subs**: For group data - compute validation loss on each subject's validation set.
* **PFIts**: runs temporal PFI for a trained model.
* **PFIch**: runs spatial or spatiotemporal PFI for a trained model.
* **PFIfreq**: runs spectral PFI for a trained model.
* **PFIfreq_ch**: runs spatiospectral PFI for a trained model.

Generally a model and dataset (specified in the corresponding variables in ```args.py```) is required to run anything. The dataset for this paper is always ```CichyData```.

## Multi-run modes
To facilitate running multiple trainings with the same command, the following variables in ```args.py``` can be lists:  
```load_model```, ```result_dir```, ```data_path```, ```dump_data```, ```load_data```, ```subjects_data```

To run on a dataset with multiple training set ratios, the following parameters can also be lists:  
```max_trials```, ```learning_rate```, ```batch_size```

To run cross-validation, set ```split``` to a list of validation split ratios.

## Data
To preprocess continuous .fif data use ```scripts/cichy_preprocess.py```. This filters the data and creates trials.  
To download and preprocess the publicly available epoched data run ```scripts/cichy_download.py``` and ```scripts/cichy_preproc_epoched.py```.

Before using any of the functions given in **Behaviour**, first training and validation splits need to be created for epoched data. This can be achieved by giving no function in ```func``` in ```args.py```, and setting ```load_data``` to False. This mode loads preprocessed data given by ```data_path``` variable, and saves ready-to-use splits to ```dump_data``` path. This ready-to-use data can then be loaded by setting ```load_data``` to the same path.

## Models
The following classification models are available:
* ```WavenetClassifier```: Subject-level wavenet classifier model. Can be used for group-level modeling as well without subject embeddings.
* ```WavenetClassifierSemb```: Group-level wavenet classifier with subject embedding. Needs to be used with a model with *semb* in its name from ```wavenet_simple.py```.  

These models are the sub-blocks of the above models:
* ```ClassifierModule```: Implements the fully-connected block for any classification model.
* ```WavenetSimple```: This is the convolutional block of the ```WavenetClassifier```.
* ```WavenetSimpleSembConcat```: This is the convolutional block of ```WavenetClassificationSemb```.
* ```WavenetSimpleSembNonlinear1```: Similar to ```WavenetSimpleSembConcat```, but with nonlinearity only in the first layer.

## Examples
To replicate some of the results in the paper we provide args files in the *examples* folder. To try these out on the publicly available MEG data, follow these steps:  
1. ```python scripts/cichy_download.py``` to download data.
2. ```python scripts/cichy_preproc_epoched.py``` to preprocess data.
3. Copy the contents of the example args file you want to run into *args.py*
4. ```python launch.py```


Note that results from running the examples will not 100% reproduce our results, because we used the raw continuous MEG data. Also, different random seeds may cause (very) small differences.

## Visualizations
Some of the figures for the paper have been created by using the jupyter notebooks provided in the scripts folder.

## License
MIT license. See LICENSE file.
