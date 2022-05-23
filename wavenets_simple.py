import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from torch.nn import Sequential, Module, Conv1d
from torch.nn import MSELoss, Dropout2d, Embedding
import torch.nn.functional as F

from scipy.signal import welch
from scipy.io import savemat


class WavenetSimple(Module):
    '''
    Implements a simplified version of wavenet without padding.
    '''
    def __init__(self, args):
        super(WavenetSimple, self).__init__()
        self.args = args
        self.inp_ch = args.num_channels
        self.out_ch = args.num_channels
        self.kernel_inds = []

        self.timesteps = args.timesteps
        self.build_model(args)

        self.criterion = MSELoss().cuda()
        self.activation = self.args.activation

        # add dropout to each layer
        self.dropout2d = Dropout2d(args.p_drop)

    def loaded(self, args):
        '''
        When model is loaded from file, assign the new args object.
        '''
        self.kernel_inds = []
        self.args = args
        self.shuffle_embeddings = False
        self.dropout2d = Dropout2d(args.p_drop)

    def build_model(self, args):
        '''
        Specify the layers of the model.
        '''
        self.ch = int(args.ch_mult * self.inp_ch)

        conv1x1_groups = args.conv1x1_groups
        modules = []

        # 1x1 convolution to project to hidden channels
        self.first_conv = Conv1d(
            self.inp_ch, self.ch, kernel_size=1, groups=conv1x1_groups)

        # each layer consists of a dilated convolution
        # followed by a nonlinear activation
        for rate in args.dilations:
            modules.append(Conv1d(self.ch,
                                  self.ch,
                                  kernel_size=args.kernel_size,
                                  dilation=rate,
                                  groups=args.groups))

        # 1x1 convolution to go back to original channel dimension
        self.last_conv = Conv1d(
            self.ch, self.out_ch, kernel_size=1, groups=conv1x1_groups)

        self.cnn_layers = Sequential(*modules)

        self.subject_emb = Embedding(args.subjects, args.embedding_dim)

    def get_weight_nograd(self, layer):
        return layer.weight.detach().clone().requires_grad_(False)

    def get_weight(self, layer):
        return layer.weight

    def get_weights(self, grad=False):
        '''
        Return a list of all weights in the model.
        '''
        get_weight = self.get_weight if grad else self.get_weight_nograd

        weights = [get_weight(layer) for layer in self.cnn_layers]
        weights.append(get_weight(self.first_conv))
        weights.append(get_weight(self.last_conv))

        return weights

    def save_embeddings(self):
        '''
        Save subject embeddings.
        '''
        weights = {'X': self.subject_emb.weight.detach().cpu().numpy()}
        savemat(os.path.join(self.args.result_dir, 'sub_emb.mat'), weights)

    def dropout(self, x):
        '''
        Applies 2D dropout to 1D data by unsqueezeing.
        '''
        if self.args.dropout2d_bad:
            x = self.dropout2d(x)
        else:
            x = torch.unsqueeze(x, 3)
            x = self.dropout2d(x)
            x = x[:, :, :, 0]

        return x

    def forward4(self, x, sid=None):
        '''
        Only use the first few layers of Wavenet.
        '''
        x = self.first_conv(x)

        # the layer from which we should get the output is
        # automatically calculated based on the receptive field
        lnum = int(np.log(self.args.rf) / np.log(self.args.kernel_size)) - 1
        for i, layer in enumerate(self.cnn_layers):
            x = self.activation(self.dropout(layer(x)))
            if i == lnum:
                break

        return self.last_conv(x), x

    def forward(self, x, sid=None):
        '''
        Run a forward pass through the network.
        '''
        x = self.first_conv(x)

        for layer in self.cnn_layers:
            x = self.activation(self.dropout(layer(x)))

        return self.last_conv(x), x

    def end(self):
        pass

    def loss(self, x, i=0, sid=None, train=True, criterion=None):
        '''
        If timesteps is bigger than 1 this loss can be used to predict any
        timestep in the future directly, e.g. t+2 or t+5, etc.
        sid: subject index
        '''
        output, _ = self.forward(x[:, :, :-self.timesteps], sid)
        target = x[:, :, -output.shape[2]:]
        if criterion is None:
            loss = self.criterion(output, target)
        else:
            loss = criterion(output, target)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/Validation loss: ': loss,
                  'valloss/saveloss/none': loss}

        return losses, output, target

    def repeat_loss(self, batch):
        '''
        Baseline loss for repeating the same timestep for future.
        '''
        start = int(batch.shape[2] / 2)
        loss = self.criterion(batch[:, :, start:-1], batch[:, :, start + 1:])

        return {'valloss/Repeat loss: ': loss}

    def ar_loss(self, output, target):
        '''
        Applies the MSE loss between output and target.
        '''
        return self.criterion(output, target)

    def channel_output(self, x, num_l, num_c):
        '''
        Compute the output for a specific layer num_l and channel num_c.
        '''
        x = self.layer_output(x, num_l)
        return -torch.mean(x[:, num_c, :])

    def layer_output(self, x, num_l, sid=None):
        '''
        Compute the output for a specific layer num_l.
        '''
        x = self.first_conv(x)
        for i in range(num_l + 1):
            x = self.cnn_layers[i](x)
            if i < num_l:
                x = self.activation(self.dropout(x))
        return x

    def run_kernel(self, x, layer, num_kernel):
        '''
        Compute the output of a specific kernel num_kernel
        in a specific layer (layer) to input x.
        '''
        # TODO: current assumption is that the network is fully depthwise
        chid = self.args.channel_idx
        ch = self.args.ch_mult

        # input and output filter indices
        out_filt = int(num_kernel/ch) + chid * ch
        inp_filt = num_kernel % ch

        # select specific channel
        x = x[:, chid*ch:(chid+1)*ch, :]

        # deconstruct convolution to get specific kernel output
        x = F.conv1d(x[:, inp_filt:inp_filt + 1, :],
                     layer.weight[
                        out_filt:out_filt + 1, inp_filt:inp_filt + 1, :],
                     layer.bias[out_filt:out_filt + 1],
                     layer.stride,
                     layer.padding,
                     layer.dilation)

        return x

    def kernel_output_all(self, x, num_l, num_f, sid=None):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.layer_output(x, num_l-1, sid)
        x = self.activation(self.dropout(x))
        x = self.run_kernel_multi(x, self.cnn_layers[num_l], num_f)

        return x.detach().cpu()

    def kernel_output(self, x, num_l, num_f):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.kernel_output_all(x, num_l, num_f)
        return -torch.mean(x)

    def plot_welch(self, x, ax, sr=1):
        '''
        Compute and plot (on ax) welch spectra of x.
        '''
        f, Pxx_den = welch(x, self.args.sr_data, nperseg=4*self.args.sr_data)
        ax.plot(f, Pxx_den)
        for freq in self.args.freqs:
            ax.axvline(x=freq, color='red')

    def kernelPFI(self, data, sid=None):
        if not self.kernel_inds:
            for _ in range(len(self.args.dilations)):
                for f in range(self.args.kernel_limit):
                    inds1 = random.randint(0, self.ch-1)
                    inds2 = random.randint(0, self.ch-1)
                    self.kernel_inds.append((inds1, inds2))

        outputs = []
        for num_layer in range(len(self.args.dilations)):
            for num_filter in range(self.args.kernel_limit):
                ind = num_layer*self.args.kernel_limit + num_filter
                x = self.kernel_output_all(data, num_layer, ind, sid)
                outputs.append(x)

        return outputs

    def run_kernel_multi(self, x, layer, num_kernel):
        '''
        Compute the output of a specific kernel num_kernel
        in a specific layer (layer) to input x.
        '''
        # input and output filter indices
        if self.args.kernel_inds:
            out_filt = self.args.kernel_inds[num_kernel][0]
            inp_filt = self.args.kernel_inds[num_kernel][1]
        elif self.kernel_inds:
            out_filt = self.kernel_inds[num_kernel][0]
            inp_filt = self.kernel_inds[num_kernel][1]
        else:
            out_filt = random.randint(0, self.ch-1)
            inp_filt = random.randint(0, self.ch-1)

        # deconstruct convolution to get specific kernel output
        x = F.conv1d(x[:, inp_filt:inp_filt + 1, :],
                     layer.weight[
                        out_filt:out_filt + 1, inp_filt:inp_filt + 1, :],
                     layer.bias[out_filt:out_filt + 1],
                     layer.stride,
                     layer.padding,
                     layer.dilation)

        return x

    def generate_forward(self, inputs, channels):
        '''
        Wrapper around forward function to easily adapt the generate function.
        '''
        return self.forward(inputs)[0].detach().reshape(channels)

    def randomize_kernel_input(self, data):
        '''
        Randomize the input for a specific kernel for kernel_network_FIR.
        '''
        input_data = data.detach().cpu().numpy()
        choosing_data = data.detach().cpu().numpy()
        for c in range(input_data.shape[1]):
            choose_channel = choosing_data[:, c, :].reshape(-1)
            length = choose_channel.shape[0]
            input_data[:, c, :] = np.random.choice(choose_channel, (1, length))
        return torch.Tensor(input_data).cuda()

    def residual(self, data, data_f):
        return data_f

    def kernel_network_FIR(self,
                           folder='kernels_network_FIR',
                           generated_data=None):
        '''
        Get FIR properties for each kernel by running the whole network.
        '''
        self.eval()
        name = folder + 'ch' + str(self.args.channel_idx)
        folder = os.path.join(self.args.result_dir, name)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # data is either drawn from gaussian or passed as argument to this func
        shape = (self.args.num_channels, self.args.generate_length)
        data = np.random.normal(0, self.args.generate_noise, shape)
        if generated_data is not None:
            data = generated_data
        data = torch.Tensor(data).cuda().reshape(1, self.args.num_channels, -1)

        data = self.first_conv(data)

        # loop over whole network
        self.kernel_network_FIR_loop(folder, data)

    def kernel_network_FIR_loop(self, folder, data):
        '''
        Implements loop over the network to get kernel output at each layer.
        '''
        for i, layer in enumerate(self.cnn_layers):
            self.kernel_FIR_plot(folder, data, i, layer)

            # compute output of current layer
            data_f = self.activation(self.dropout(layer(data)))
            data = self.residual(data, data_f)

    def kernel_FIR_plot(self, folder, data, i, layer, name='conv'):
        '''
        Plot FIR response of kernels in current layer (i) to input data.
        '''
        num_plots = self.args.kernel_limit
        fig, axs = plt.subplots(num_plots+1, figsize=(20, num_plots*3))

        multi = self.args.groups == 1
        kernel_func = self.run_kernel_multi if multi else self.run_kernel

        filter_outputs = []
        for k in range(num_plots):
            x = kernel_func(data, layer, k)
            x = x.detach().cpu().numpy().reshape(-1)
            filter_outputs.append(x)

            # compute fft of kernel output
            self.plot_welch(x, axs[k], i)

        filter_outputs = np.array(filter_outputs)
        path = os.path.join(folder, name + str(i) + '.mat')
        savemat(path, {'X': filter_outputs})

        filename = os.path.join(folder, name + str(i) + '.svg')
        fig.savefig(filename, format='svg', dpi=2400)
        plt.close('all')


class WavenetSimpleSembConcat(WavenetSimple):
    '''
    Implements simplified wavenet with concatenated subject embeddings.
    '''
    def loaded(self, args):
        super(WavenetSimpleSembConcat, self).loaded(args)
        self.emb_window = False

    def build_model(self, args):
        self.emb_window = False
        self.shuffle_embeddings = False
        self.inp_ch = args.num_channels + args.embedding_dim
        super(WavenetSimpleSembConcat, self).build_model(args)

    def embed(self, x, sid):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)

        # shuffle embeddings in a window if needed
        if self.emb_window:
            idx = np.random.rand(*sid[:, :, 0].T.shape).argsort(0)
            a = sid[:, :, 0].T.clone()
            out = a[idx, np.arange(a.shape[1])].T

            w = self.emb_window
            out = out.repeat(w[1] - w[0], 1, 1)
            sid[:, :, w[0]:w[1]] = out.permute(1, 2, 0)

        x = torch.cat((x, sid), dim=1)

        return x

    def get_weights(self, grad=False):
        weights = super(WavenetSimpleSembConcat, self).get_weights(grad)
        if self.args.reg_semb:
            weights.append(self.subject_emb.weight)

        return weights

    def forward(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        # shuffle embedding values if needed
        if self.shuffle_embeddings:
            print('This code needs to be checked!')
            subid = int(sid[0].detach().cpu().numpy())
            indices = torch.randperm(self.subject_emb.weight.shape[1])
            w = self.subject_emb.weight.detach()
            w[subid, :] = w[subid, indices]
            self.subject_emb.weight = torch.nn.Parameter(w)

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).forward(x)

    def forward4(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).forward4(x)

    def layer_output(self, x, num_l, sid=None):
        '''
        Compute the output for a specific layer num_l.
        '''
        if sid is None:
            # repeat x 15 times
            x = x.repeat(self.args.subjects, 1, 1)

            # use all 15 embeddings
            sid = torch.LongTensor(np.arange(self.args.subjects)).cuda()

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).layer_output(x, num_l)

    def kernel_network_FIR(self,
                           folder='kernels_network_FIR',
                           generated_data=None):
        '''
        Get FIR properties for each kernel by running the whole network.
        '''
        self.eval()
        name = folder + 'ch' + str(self.args.channel_idx)
        folder = os.path.join(self.args.result_dir, name)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # data is either drawn from gaussian or passed as argument to this func
        shape = (self.args.num_channels, self.args.generate_length)
        data = np.random.normal(0, self.args.generate_noise, shape)
        if generated_data is not None:
            data = generated_data
        data = torch.Tensor(data).cuda().reshape(1, self.args.num_channels, -1)

        # apply subject embedding
        sid = torch.LongTensor([10]).cuda()
        sid = sid.repeat(data.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)
        data = torch.cat((data, sid), dim=1)
        data = self.first_conv(data)

        # loop over whole network
        self.kernel_network_FIR_loop(folder, data)


class WavenetSimpleSembNonlinear1(WavenetSimpleSembConcat):
    def forward(self, x, sid=None):
        '''
        Run a forward pass through the network.
        '''
        x = self.embed(x, sid)

        # only do nonlinear activation after first layer
        x = torch.asinh(self.first_conv(x))

        for layer in self.cnn_layers:
            x = self.activation(self.dropout(layer(x)))

        return self.last_conv(x), x
