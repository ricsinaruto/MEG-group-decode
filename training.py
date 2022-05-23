import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from copy import deepcopy

from scipy.fft import fft, ifft

from torch.nn import MSELoss
from torch.optim import Adam

from loss import Loss
from classifiers_linear import LDA


class Experiment:
    def __init__(self, args, dataset=None):
        '''
        Initialize model and dataset using an Args object.
        '''
        self.args = args
        self.loss = Loss()
        self.val_losses = []
        self.train_losses = []

        # create folder for results
        if os.path.isdir(self.args.result_dir):
            print('Result directory already exists, writing to it.',
                  flush=True)
            print(self.args.result_dir, flush=True)
        else:
            os.makedirs(self.args.result_dir)
            print('New result directory created.', flush=True)
            print(self.args.result_dir, flush=True)

        # save args object
        path = os.path.join(self.args.result_dir, 'args_saved.py')
        os.system('cp ' + args.name + ' ' + path)

        # initialize dataset
        if dataset is not None:
            self.dataset = dataset
        elif args.load_dataset:
            self.dataset = args.dataset(args)
            print('Dataset initialized.', flush=True)

        # load model if path is specified
        if args.load_model:
            if 'model' in args.load_model:
                self.model_path = args.load_model
            else:
                self.model_path = os.path.join(args.load_model, 'model.pt')

            # LDA vs deep learning models
            try:
                self.model = torch.load(self.model_path)
                self.model.loaded(args)
                self.model.cuda()
            except:
                self.model = pickle.load(open(self.model_path, 'rb'))
                self.model.loaded(args)

            self.model_path = os.path.join(self.args.result_dir, 'model.pt')

            print('Model loaded from file.', flush=True)
            #self.args.dataset = self.dataset
        else:
            self.model_path = os.path.join(self.args.result_dir, 'model.pt')
            try:
                self.model = self.args.model(self.args).cuda()
                print('Model initialized with cuda.', flush=True)
            except:  # if cuda not available or not cuda model
                self.model = self.args.model(self.args)
                print('Model initialized without cuda.')

        try:
            # calculate number of total parameters in model
            parameters = [param.numel() for param in self.model.parameters()]
            print('Number of parameters: ', sum(parameters), flush=True)
        except:
            print('Can\'t calculate number of parameters.', flush=True)

    def train(self):
        '''
        Main training loop over epochs and training batches.
        '''
        # initialize optimizer
        optimizer = Adam(self.model.parameters(),
                         lr=self.args.learning_rate,
                         weight_decay=self.args.alpha_norm)

        # start with a pass over the validation set
        best_val = 1000000
        self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()
            self.loss.dict = {}

            # save initial model
            if epoch == 0:
                path = os.path.join(self.args.result_dir, 'model_init.pt')
                torch.save(self.model, path, pickle_protocol=4)
                print('Model saved to result directory.', flush=True)

            # loop over batches
            for i in range(self.dataset.train_batches):
                batch, sid = self.dataset.get_train_batch(i)
                # need to check whether it's an empty batch
                try:
                    if batch.shape[0] < 1:
                        break
                except AttributeError:
                    pass

                losses, _, _, = self.model.loss(batch, i, sid, train=True)

                # optimize model according to the optimization loss
                optkey = [key for key in losses if 'optloss' in key]
                losses[optkey[0]].backward()
                optimizer.step()
                optimizer.zero_grad()
                self.loss.append(losses)

            # print training losses
            if not epoch % self.args.print_freq:
                losses = self.loss.print('trainloss')
                self.train_losses.append([losses[k] for k in losses])

            # run validation pass and save model
            if not epoch % self.args.val_freq:
                losses, _, _ = self.evaluate()
                loss = [losses[k] for k in losses if 'saveloss' in k]
                losses = [losses[k] for k in losses if 'saveloss' not in k]

                self.val_losses.append(losses)

                # only save model if validation loss is best so far
                if loss[0] < best_val:
                    best_val = loss[0]
                    torch.save(self.model, self.model_path, pickle_protocol=4)
                    print('Validation loss improved, model saved.', flush=True)

                    # also compute test loss
                    self.testing()

                # save loss plots if needed
                if self.args.save_curves:
                    self.save_curves()

        # wrap up training, save model and validation loss
        path = self.model_path.strip('.pt') + '_end.pt'
        torch.save(self.model, path, pickle_protocol=4)
        self.model.end()
        self.save_validation()

    def testing(self):
        '''
        Evaluate model on the test set.
        '''
        self.loss.dict = {}
        self.model.eval()

        # loop over test batches
        for i in range(self.dataset.test_batches):
            batch, sid = self.dataset.get_test_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

        losses = self.loss.print('valloss')

        path = os.path.join(self.args.result_dir, 'test_loss.txt')
        with open(path, 'w') as f:
            f.write(str(losses))

    def save_curves(self):
        '''
        Save train and validation loss plots to file.
        '''
        val_losses = np.array(self.val_losses)
        train_losses = np.array(self.train_losses)

        if val_losses.shape[0] > 2:
            val_ratio = int((train_losses.shape[0]-1)/(val_losses.shape[0]-1))
            val_losses = np.repeat(val_losses, val_ratio, axis=0)

            plt.semilogy(train_losses, linewidth=1, label='training losses')
            plt.semilogy(val_losses, linewidth=1, label='validation losses')
            plt.legend()

            path = os.path.join(self.args.result_dir, 'losses.svg')
            plt.savefig(path, format='svg', dpi=1200)
            plt.close('all')

    def evaluate(self):
        '''
        Evaluate model on the validation dataset.
        '''
        self.loss.dict = {}
        self.model.eval()
        outputs = []
        targets = []

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

            outputs.append(output)
            targets.append(target)

        losses = self.loss.print('valloss')
        return losses, torch.cat(outputs), torch.cat(targets)

    def save_validation(self):
        '''
        Save validation loss to file.
        '''
        loss, output, target = self.evaluate()

        # print variance if needed
        #if output is not None and target is not None:
        #    print(torch.std((output-target).flatten()))

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write(str(loss))

    def save_validation_subs(self):
        '''
        Print validation losses separately on each subject's dataset.
        '''
        self.model.eval()
        losses = []

        # don't reduce the loss so we can separate it according to subjects
        mse = MSELoss(reduction='none').cuda()

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss_dict, _, _ = self.model.loss(
                batch, i, sid, train=False, criterion=mse)

            loss = [loss_dict[k] for k in loss_dict if 'valcriterion' in k]
            #loss = torch.mean(loss, (1, 2)).detach()
            loss = loss[0].detach()
            losses.append((sid, loss))

        sid = torch.cat(tuple([loss[0] for loss in losses]))
        loss = torch.cat(tuple([loss[1] for loss in losses]))

        path = os.path.join(self.args.result_dir, 'val_loss_subs.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                sub_loss = torch.mean(loss[sid == i]).item()
                f.write(str(sub_loss) + '\n')

    def PFIemb(self):
        '''
        Permutation Feature Importance (PFI) function for subject embeddings.
        '''
        loss_list = []
        hw = self.args.halfwin
        times = self.dataset.x_val_t.shape[2]

        # slide over the epoch and always permute embeddings within a window
        for i in range(hw-1, times-hw):
            if i > hw-1:
                self.model.wavenet.emb_window = (i-hw, i+hw)

            losses, _, _ = self.evaluate()
            loss = [losses[k] for k in losses if 'Validation accuracy' in k]
            loss_list.append(str(loss[0]))

        name = 'val_loss_PFIemb' + str(hw) + '.txt'
        path = os.path.join(self.args.result_dir, name)
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

    def evaluate_(self, data, og=None):
        '''
        Evaluation helper function for PFI.
        '''
        losses, _, _ = self.evaluate()
        loss = [losses[k] for k in losses if 'Validation accuracy' in k]
        return loss[0]

    def LDA_eval_(self, data, og=None):
        '''
        Evaluation helper for PFI for linear models.
        '''
        acc, _, _ = self.model.eval(data)
        return acc

    def kernelPFI(self, data, og=False):
        '''
        Helper function for PFI to compute kernel output deviations.
        '''
        self.model.eval()
        ch = self.args.num_channels
        num_l = len(self.args.dilations)

        # get output at specific kernels
        outputs_batch = []
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            out = self.model.kernelPFI(batch[:, :ch, :], sid)
            outputs_batch.append(out)

        # concatenate along trial dimension
        outputs = []
        for i in range(len(outputs_batch[0])):
            out = torch.cat([o[i] for o in outputs_batch])
            outputs.append(out)

        if og:
            # set original kernel outputs
            self.kernelPFI_outputs = outputs
            ret = np.zeros(self.args.kernel_limit*num_l)
        else:
            # compute kernel output deviation
            ret = []
            for og, new in zip(self.kernelPFI_outputs, outputs):
                ret.append(torch.linalg.norm(og-new).numpy())
            ret = np.array(ret)

        return ret

    def PFIts(self):
        '''
        Permutation Feature Importance (PFI) function for timesteps.
        '''
        hw = self.args.halfwin
        val_t = self.dataset.x_val_t.clone()
        shuffled_val_t = self.dataset.x_val_t.clone()
        chn = val_t.shape[1] - 1
        times = val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        perm_list = []
        for p in range(self.args.PFI_perms):
            # first permute channels across all timesteps
            idx = np.random.rand(*val_t[:, :chn, 0].T.shape).argsort(0)
            for i in range(times):
                a = shuffled_val_t[:, :chn, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_t[:, :chn, i] = out

            loss_list = [og_loss]
            # slide over the epoch and always permute timesteps within a window
            for i in range(hw, times-hw):
                self.dataset.x_val_t = val_t.clone()
                if i > hw:
                    # either permute inside or outside the window
                    if self.args.PFI_inverse:
                        window = shuffled_val_t[:, :chn, :i-hw].clone()
                        self.dataset.x_val_t[:, :chn, :i-hw] = window
                        window = shuffled_val_t[:, :chn, i+hw:].clone()
                        self.dataset.x_val_t[:, :chn, i+hw:] = window
                    else:
                        window = shuffled_val_t[:, :chn, i-hw:i+hw].clone()
                        self.dataset.x_val_t[:, :chn, i-hw:i+hw] = window

                loss = val_func(self.dataset.x_val_t)
                loss_list.append(loss)

            perm_list.append(np.array(loss_list))

        # save accuracies to file
        path = os.path.join(self.args.result_dir, 'val_loss_PFIts.npy')
        np.save(path, np.array(perm_list))

    def PFIfreq(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        '''
        hw = self.args.halfwin
        chn = self.dataset.x_val_t.shape[1] - 1
        times = self.dataset.x_val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # compute fft
        val_fft = fft(self.dataset.x_val_t[:, :chn, :].cpu().numpy())
        shuffled_val_fft = val_fft.copy()
        samples = [val_fft[0, 0, :].copy()]

        # original loss without shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        perm_list = []
        for p in range(self.args.PFI_perms):
            # shuffle frequency components across channels
            idx = np.random.rand(*val_fft[:, :, 0].T.shape).argsort(0)
            for i in range(times):
                a = shuffled_val_fft[:, :, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_fft[:, :, i] = out

            loss_list = [og_loss]
            # slide over epoch and always permute frequencies within a window
            for i in range(hw, times//2-hw):
                if i > hw:
                    dataset_val_fft = val_fft.copy()
                    win1 = shuffled_val_fft[:, :, i-hw:i+hw+1].copy()
                    dataset_val_fft[:, :, i-hw:i+hw+1] = win1

                    if -i+hw+1 == 0:
                        win2 = shuffled_val_fft[:, :, -i-hw:].copy()
                        dataset_val_fft[:, :, -i-hw:] = win2
                    else:
                        win2 = shuffled_val_fft[:, :, -i-hw:-i+hw+1].copy()
                        dataset_val_fft[:, :, -i-hw:-i+hw+1] = win2

                    samples.append(dataset_val_fft[0, 0, :].copy())

                    # inverse fourier transform
                    data = torch.Tensor(ifft(dataset_val_fft))
                    self.dataset.x_val_t[:, :chn, :] = data

                loss = val_func(self.dataset.x_val_t)
                loss_list.append(loss)

            perm_list.append(np.array(loss_list))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            'val_loss_PFIfreqs' + str(hw*2) + '.npy')
        np.save(path, np.array(perm_list))

        np.save(path + '_samples', np.array(samples))

    def PFIfreq_ch(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        spectral PFI is done separately for each channel.
        '''
        top_chs = self.args.closest_chs
        hw = self.args.halfwin
        chn = self.dataset.x_val_t.shape[1] - 1
        times = self.dataset.x_val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # read a file containing closest channels to each channel location
        path = os.path.join(self.args.result_dir, 'closest' + str(top_chs))
        with open(path, 'rb') as f:
            closest_k = pickle.load(f)

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        # compute fft
        val_fft = fft(self.dataset.x_val_t[:, :chn, :].cpu().numpy())
        shuffled_val_fft = val_fft.copy()

        samples = [val_fft[0, 0, :].copy()]

        perm_list = []
        for p in range(self.args.PFI_perms):
            # shuffle frequency components across channels
            idx = np.random.rand(*val_fft[:, :, 0].T.shape).argsort(0)
            for i in range(times):
                a = shuffled_val_fft[:, :, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_fft[:, :, i] = out

            windows = []
            # slide over epoch and always permute frequencies within a window
            for i in range(hw+1, times//2-hw):

                loss_list = [og_loss]
                for c in range(int(chn/3)):
                    # need to select magnetometer and 2 gradiometers
                    a = np.array(closest_k[c]) * 3
                    chn_idx = np.append(np.append(a, a+1), a+2)

                    dataset_val_fft = val_fft.copy()

                    # instead of shuffling just set to 0
                    dataset_val_fft[:, chn_idx, i-hw:i+hw+1] = 0 + 0j
                    if -i+hw+1 == 0:
                        dataset_val_fft[:, chn_idx, -i-hw:] = 0 + 0j
                    else:
                        dataset_val_fft[:, chn_idx, -i-hw:-i+hw+1] = 0 + 0j

                    samples.append(dataset_val_fft[0, 0, :].copy())

                    # inverse fourier transform
                    data = torch.Tensor(ifft(dataset_val_fft))
                    self.dataset.x_val_t[:, :chn, :] = data

                    loss = val_func(self.dataset.x_val_t)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            'val_loss_PFIfreqs_ch' + str(hw*2) + '.npy')
        np.save(path, np.array(perm_list))

        np.save(path + '_samples', np.array(samples))

    def PFIch(self):
        '''
        Permutation Feature Importance (PFI) function for channels.
        '''
        top_chs = self.args.closest_chs
        val_t = self.dataset.x_val_t.clone()
        shuffled_val_t = self.dataset.x_val_t.clone()
        chn = val_t.shape[1] - 1

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # read a file containing closest channels to each channel location
        path = os.path.join(self.args.result_dir, 'closest' + str(top_chs))
        with open(path, 'rb') as f:
            closest_k = pickle.load(f)

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        # loop over channels and permute channels in vicinity of i-th channel
        perm_list = []
        for p in range(self.args.PFI_perms):
            # first permute timesteps across all channels
            idx = np.random.rand(*val_t[:, 0, :].T.shape).argsort(0)
            for i in range(chn):
                a = shuffled_val_t[:, i, :].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_t[:, i, :] = out

            windows = []
            # loop over time windows for spatiotemporal PFI
            for t in self.args.pfich_timesteps:
                tmin = t[0]
                tmax = t[1]
                loss_list = [og_loss]

                for i in range(int(chn/3)):
                    self.dataset.x_val_t = val_t.clone()

                    # need to select magnetometer and 2 gradiometers
                    a = np.array(closest_k[i]) * 3
                    chn_idx = np.append(np.append(a, a+1), a+2)

                    # shuffle closest k channels
                    if self.args.PFI_inverse:
                        mask = np.ones(chn+1, np.bool)
                        mask[chn_idx] = 0
                        mask[chn] = 0
                        window = shuffled_val_t[:, mask, tmin:tmax].clone()
                        self.dataset.x_val_t[:, mask, tmin:tmax] = window
                    else:
                        window = shuffled_val_t[:, chn_idx, tmin:tmax].clone()
                        self.dataset.x_val_t[:, chn_idx, tmin:tmax] = window

                    loss = val_func(self.dataset.x_val_t)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        name = 'val_loss_PFIch' + str(top_chs) + '.npy'
        path = os.path.join(self.args.result_dir, name)
        np.save(path, np.array(perm_list))

        '''
        # loop over permutations of other channels than i
        # this is to improve noisy results when permuting 1 channel
        chn_idx = np.array([c for c in range(chn) if c != i])
        chn_idx = np.random.choice(chn_idx, size=19, replace=False)
        chn_idx = np.append(chn_idx, i)
        '''

    def kernel_network_FIR(self):
        self.model.kernel_network_FIR()

    def save_embeddings(self):
        # only run this if there are multiple subjects
        if self.args.subjects > 0 and self.args.embedding_dim > 0:
            self.model.save_embeddings()


def main(Args):
    '''
    Main function creating an experiment object and running everything.
    This should be called from launch.py, and it needs an Args object.
    '''
    args = Args()
    dataset = None

    def checklist(x, i):
        # check if an argument is list or not
        return x[i] if isinstance(x, list) else x

    # if list of paths is given, then process everything individually
    for i, d_path in enumerate(args.data_path):
        args_pass = Args()

        args_pass.data_path = d_path
        args_pass.result_dir = checklist(args.result_dir, i)
        args_pass.dump_data = checklist(args.dump_data, i)
        args_pass.load_data = checklist(args.load_data, i)
        args_pass.norm_path = checklist(args.norm_path, i)
        args_pass.pca_path = checklist(args.pca_path, i)
        args_pass.AR_load_path = checklist(args.AR_load_path, i)
        args_pass.load_model = checklist(args.load_model, i)
        args_pass.p_drop = checklist(args.p_drop, i)
        args_pass.load_conv = checklist(args.load_conv, i)
        args_pass.compare_model = checklist(args.compare_model, i)
        args_pass.stft_freq = checklist(args.stft_freq, i)

        if isinstance(args.num_channels[0], list):
            args_pass.num_channels = args.num_channels[i]
        if isinstance(args.subjects_data, list):
            if isinstance(args.subjects_data[0], list):
                args_pass.subjects_data = args.subjects_data[i]

        # skip if subject does not exist
        if not isinstance(d_path, list):
            if not (os.path.isfile(d_path) or os.path.isdir(d_path)):
                print('Skipping ' + d_path, flush=True)
                continue

        # separate loops for cross-validation (handled by args.split)
        num_loops = len(args.split) if isinstance(args.split, list) else 1
        split_len = num_loops

        # separate loops for different training set ratios
        if isinstance(args.max_trials, list):
            num_loops = len(args.max_trials)

        # if only using one dataset across loops, initialze it once
        if args.common_dataset and dataset is None:
            dataset = args_pass.dataset(args_pass)
            args_data = deepcopy(args_pass)

        # inner loops, see above
        for n in range(num_loops):
            args_new = deepcopy(args_pass)

            args_new.max_trials = checklist(args_new.max_trials, n)
            args_new.learning_rate = checklist(args_new.learning_rate, n)
            args_new.batch_size = checklist(args_new.batch_size, n)

            # load learned dimensionality reduction for linear models
            if args_new.load_conv:
                if 'model.pt' not in args_new.load_conv:
                    args_new.load_conv = os.path.join(
                        args_new.load_conv, 'cv' + str(n), 'model.pt')

            # load cross-validation folds accordingly
            if split_len > 1:
                args_new.split = checklist(args.split, n)
                args_new.dump_data = os.path.join(
                    args_new.dump_data + str(n), 'c')
                if args_new.load_data:
                    args_new.load_data = args_new.dump_data
                args_new.result_dir = os.path.join(
                    args_pass.result_dir, 'cv' + str(n))
                if args_new.load_model:
                    paths = args_pass.load_model.split('/')
                    args_new.load_model = os.path.join(
                        '/'.join(paths[:-1]), 'cv' + str(n), paths[-1])

            # else use max_trials for looping logic
            elif isinstance(args.max_trials, list):
                name = 'train' + str(args_new.max_trials)
                args_new.result_dir = os.path.join(args_pass.result_dir, name)

            # set common dataset if given
            if args.common_dataset:
                args_data.load_model = args.load_model[i]
                args_data.result_dir = args.result_dir[i]
                e = Experiment(args_data, dataset)
            else:
                e = Experiment(args_new)

            # only run the functions specified in args
            if Args.func.get('train'):
                e.train()
            if Args.func.get('kernel_network_FIR'):
                e.kernel_network_FIR()
            if Args.func.get('save_validation_subs'):
                e.save_validation_subs()
            if Args.func.get('PFIts'):
                e.PFIts()
            if Args.func.get('PFIch'):
                e.PFIch()
            if Args.func.get('PFIemb'):
                e.PFIemb()
            if Args.func.get('PFIfreq'):
                e.PFIfreq()
            if Args.func.get('PFIfreq_ch'):
                e.PFIfreq_ch()

            e.save_embeddings()

            # delete model and dataset
            del e.model
            del e.dataset
            torch.cuda.empty_cache()
