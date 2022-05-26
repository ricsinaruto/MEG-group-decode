# MEG-group-decode

Code for reproducing results in [Generalizing Brain Decoding Across Subjects with Deep Learning](arxiv). We propose a neuroscientifically interpretable deep learning model capable of jointly decoding multiple subjects in neuroimaging data aided by subject embeddings.


## Features
  :magnet: &nbsp; Train WaveNet-based decoding models on MEG data.  
  :rocket: &nbsp; Subject and group-level models included with optional subject embeddins.  
  :brain: &nbsp; Neuroscientifically interpretable features can be extracted and visualized from trained models.  
  :twisted_rightwards_arrows: &nbsp; Flexible pipeline with support for [multi-run modes](https://github.com/ricsinaruto/MEG-group-decode/edit/main/README.md#multi-run-modes) (e.g. cross-validation within and across subjects, different training set ratios, etc.)


## Usage
First requirements need to be installed.
```
pip install -r requirements.txt
```

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

If running any function on some data for the first time ```load_data``` needs to be False. This loads preprocessed data given by ```data_path``` variable, and saves ready-to-use splits to ```dump_data``` path. This ready-to-use data can then be used in subsequent runs by setting ```load_data``` to the ```dump_data``` path.

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
To replicate some of the results in the paper we provide args files in the examples folder. To try these out on the publicly available MEG data, follow these steps:  
1. ```python scripts/cichy_download.py``` to download data.
2. ```python scripts/cichy_preproc_epoched.py``` to preprocess data.
3. Copy the contents of the example args file you want to run into ```args.py```
4. ```python launch.py```

The following example args files are available:
* ```args_linear_subject.py```: trains a subject-level linear Wavenet Classifier (WC) on each subject.
* ```args_nonlinear_subject.py```: trains a subject-level nonlinear WC on each subject.
* ```args_nonlinear_group.py```: trains a group-level nonlinear WC on all subjects.
* ```args_linear_group.py```: trains a group-level linear WC on all subjects.
* ```args_linear_group-emb.py```: trains a group-level linear WC with subject embeddings.
* ```args_nonlinear_group-emb.py```: trains a group-level nonlinear WC with subject embeddings. After the training is done, temporal and spatial PFI is computed corresponding to Figure 4b in the paper.
* ```args_nonlinear_group-emb_finetuned.py```: trains a subject-level nonlinear WC which is initialized with the *nonlinear_group-emb* model (which needs to be trained first).
* ```args_generalization_subject.py```: trains a *linear_subject* model with 10 training data ratios for each subject (Figure 4a in the paper).
* ```args_kernel_spatiotemporalPFI.py```: Runs temporal and spatial PFI for selected kernels of the *nonlinear_group-emb* model, which needs to be trained first. Corresponds to Figure 5a and 5c in the paper.
* ```args_kernelFIR_spectralPFI.py```: Runs spectral PFI and kernel FIR analysis for selected kernels of the *nonlinear_group-emb* model, which needs to be trained first. Corresponds to Figure 6a and 6c in the paper.

Steps 1 and 2 can be skipped if running on non-public data. The relevant data paths in the args file have to modified in this case. Note that results from running the examples will not 100% reproduce our results, because we used the raw continuous MEG data. Also, different random seeds may cause (very) small differences.

## Visualizations
Some of the figures for the paper have been created by using the jupyter notebooks provided in the scripts folder.

## Authors
* **[Richard Csaky](https://ricsinaruto.github.io)**

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ricsinaruto/MEG-group-decode/blob/master/LICENSE) file for details.  
Please include a link to this repo if you use any of the dataset or code in your work and consider citing the following paper:
```
@article{Csaky:2022,
    title = "Generalizing Brain Decoding Across Subjects with Deep Learning",
    author = "Csaky, Richard and van Es, Mats and Jones, Oiwi Parker and Woolrich, Mark",
    url = "arxiv",
}
```
