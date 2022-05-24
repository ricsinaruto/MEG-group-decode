import os
import torch
import torch.nn.functional as F
import numpy as np

from classifiers_wavenet import WavenetClassifier
from wavenets_simple import WavenetSimple
from cichy_data import CichyData


class Args:
    gpu = '0'  # cuda gpu index
    func = {'train': True}  # dict of functions to run from training.py

    def __init__(self):
        n = 1  # can be used to do multiple runs, e.g. over subjects

        # experiment arguments
        self.name = 'args.py'  # name of this file, don't change
        self.common_dataset = False  # use a shared dataset for all runs
        self.load_dataset = True  # whether to load self.dataset
        self.learning_rate = 0.0001  # learning rate for Adam
        self.max_trials = 1  # ratio of training data (1=max)
        self.batch_size = 590  # batch size for training and validation data
        self.epochs = 2000  # number of loops over training data
        self.val_freq = 20  # how often to validate (in epochs)
        self.print_freq = 5  # how often to print metrics (in epochs)
        self.save_curves = True  # whether to save loss curves to file
        self.load_model = False  # either False or path to model(s) to load
        self.result_dir = [os.path.join(  # path(s) to save model and others
            'results',
            'nonlinear_group')]
        self.model = WavenetClassifier  # classifier model to use
        self.dataset = CichyData  # dataset class for loading and handling data

        # wavenet arguments
        self.activation = torch.asinh  # activation function for models
        self.subjects = 15  # number of subjects used for training
        self.embedding_dim = 0  # subject embedding size
        self.p_drop = 0.4  # dropout probability
        self.ch_mult = 2  # channel multiplier for hidden channels in wavenet
        self.kernel_size = 2  # convolutional kernel size
        self.timesteps = 1  # how many timesteps in the future to forecast
        self.sample_rate = [0, 256]  # start and end of timesteps within trials
        self.rf = 64  # receptive field of wavenet
        rf = 64
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        self.dilations = [ks**i for i in range(nl)]  # dilation: 2^num_layers

        # classifier arguments
        self.wavenet_class = WavenetSimple  # class of wavenet model
        self.num_classes = 118  # number of classes for classification
        self.units = [1000, 400]  # hidden layer sizes of fully-connected block

        # dataset arguments
        data_path = os.path.join('data', 'preproc')
        self.data_path = [os.path.join(data_path)]  # path(s) to data directory
        self.num_channels = list(range(306))  # channel indices
        # should be 306 when self.load_data loading raw data, and 307
        # when loading data with self.load_data
        # (target labels are placed in extra channel)

        self.numpy = True  # whether data is saved in numpy format
        self.crop = 1  # cropping ratio for trials
        self.whiten = False  # pca components used in whitening
        self.group_whiten = False  # whether to perform whitening at the GL
        self.split = np.array([0, 0.2])  # validation split (start, end)
        self.sr_data = 250  # sampling rate used for downsampling
        self.save_data = True  # whether to save the created data
        self.subjects_data = False  # list of subject inds to use in group data
        self.dump_data = [os.path.join(d, 'data_files', 'c')
                          for d in self.data_path]  # path(s) for dumping data
        self.load_data = ''#self.dump_data  # path(s) for loading data files

        # analysis arguments
        self.closest_chs = 20  # channel neighbourhood size for spatial PFI
        self.PFI_inverse = False  # invert which channels/timesteps to shuffle
        self.kernelPFI = True  # whether to run PFI with kernel output metric
        self.pfich_timesteps = [0, 256]  # time window for spatiotemporal PFI
        self.PFI_perms = 20  # number of PFI permutations
        self.halfwin = 5  # half window size for temporal PFI
        self.halfwin_uneven = False  # whether to use even or uneven window
        self.generate_length = self.sr_data * 1000  # generated timeseries len
        self.kernel_limit = 1  # max number of kernels to analyse per layer
        self.kernel_inds = [  # kernel indices to use for analysis
                              # (in_channel, out_channel) for kernel_limit
                              # number of kernels in each layer:
                              # [layer0K0, layer0K1,... layer1K0, layer1K1,...]
                             (155, 166),
                             (261, 514),
                             (217, 422),
                             (380, 39),
                             (360, 605),
                             (320, 455)]

        '''
        .
        .
        The following parameters are not used currently.
        .
        .
        '''
        self.order = 64
        self.uni = False
        self.save_AR = False
        self.do_anal = False
        self.AR_load_path = False
        self.num_plot = 1
        self.plot_ch = 1
        self.linear = False
        self.dropout2d_bad = False
        self.mu = 255
        self.groups = 1
        self.conv1x1_groups = 1
        self.pos_enc_type = 'cat'
        self.pos_enc_d = 128
        self.l1_loss = False
        self.norm_alpha = 0.0
        self.num_components = 0
        self.resample = 7
        self.save_norm = True
        self.norm_path = os.path.join(data_path, 'norm_coeff')
        self.pca_path = os.path.join(data_path, 'pca128_model')
        self.load_pca = False
        self.compare_model = False
        self.channel_idx = 0
        self.individual = True
        self.anal_lr = 0.001
        self.anal_epochs = 200
        self.norm_coeff = 0.0001
        self.generate_mode = 'IIR'
        self.generate_input = 'gaussian_noise'
        self.generate_noise = 1
        self.load_conv = False
        self.pred = False
        self.init_model = True
        self.reg_semb = True
        self.fixed_wavenet = False
        self.alpha_norm = self.norm_alpha
        self.dim_red = 80
        self.stft_freq = 0
