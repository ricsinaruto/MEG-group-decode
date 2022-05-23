import os
import pickle
import numpy as np

from scipy.io import loadmat, savemat

from mrc_data import MRCData


class CichyData(MRCData):
    '''
    Class for loading the trials from the Cichy dataset.
    '''
    def set_subjects(self, split):
        inds = np.in1d(self.sub_id[split], self.args.subjects_data)
        self.sub_id[split] = self.sub_id[split][:, inds]

        if split == 'train':
            self.x_train_t = self.x_train_t[inds]
        elif split == 'val':
            self.x_val_t = self.x_val_t[inds]
        elif split == 'test':
            self.x_test_t = self.x_test_t[inds]

    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        x_train_ts = []
        x_val_ts = []
        x_test_ts = []

        # load data for each channel
        for index, i in enumerate(chn):
            data = loadmat(args.load_data + 'ch' + str(i) + '.mat')
            x_train_ts.append(np.array(data['x_train_t']))
            x_val_ts.append(np.array(data['x_val_t']))
            try:
                x_test_ts.append(np.array(data['x_test_t']))
            except:
                pass

            if index == 0:
                self.sub_id['train'] = np.array(data['sub_id_train'])
                self.sub_id['val'] = np.array(data['sub_id_val'])
                try:
                    self.sub_id['test'] = np.array(data['sub_id_test'])
                except:
                    pass

        self.x_train_t = np.concatenate(tuple(x_train_ts), axis=1)
        self.x_val_t = np.concatenate(tuple(x_val_ts), axis=1)

        if len(x_test_ts) == 0:
            self.x_test_t = self.x_val_t
            self.sub_id['test'] = self.sub_id['val']
        else:
            self.x_test_t = np.concatenate(tuple(x_test_ts), axis=1)

    def set_common(self, args):
        if not isinstance(args.num_channels, list):
            args.num_channels = list(range(args.num_channels+1))

        num_ch = len(args.num_channels) - 1

        # select wanted subjects
        if args.subjects_data:
            self.set_subjects('train')
            self.set_subjects('val')
            self.set_subjects('test')

        # crop data
        tmin = args.sample_rate[0]
        tmax = args.sample_rate[1]
        self.x_train_t = self.x_train_t[:, :, tmin:tmax]
        self.x_val_t = self.x_val_t[:, :, tmin:tmax]
        self.x_test_t = self.x_test_t[:, :, tmin:tmax]

        args.sample_rate = tmax - tmin

        # select a subset of training trials
        num_trials = np.sum(self.x_train_t[:, num_ch, 0] == 0.0)
        max_trials = int(args.max_trials * num_trials)
        trials = [0] * args.num_classes

        inds = []
        for i in range(self.x_train_t.shape[0]):
            cond = int(self.x_train_t[i, num_ch, 0])
            if trials[cond] < max_trials:
                trials[cond] += 1
                inds.append(i)

        self.x_train_t = self.x_train_t[inds, :, :]

        # whiten data if needed
        if args.group_whiten:
            # reshape for PCA
            x_train = self.x_train_t[:, :num_ch, :].transpose(0, 2, 1)
            x_val = self.x_val_t[:, :num_ch, :].transpose(0, 2, 1)
            x_test = self.x_test_t[:, :num_ch, :].transpose(0, 2, 1)
            x_train = x_train.reshape(-1, num_ch)
            x_val = x_val.reshape(-1, num_ch)
            x_test = x_test.reshape(-1, num_ch)

            # change dim red temporarily
            dim_red = args.dim_red
            args.dim_red = num_ch
            x_train, x_val, x_test = self.whiten(x_train, x_val, x_test)
            args.dim_red = dim_red

            # reshape back to trials
            x_train = x_train.reshape(-1, args.sample_rate, num_ch)
            x_val = x_val.reshape(-1, args.sample_rate, num_ch)
            x_test = x_test.reshape(-1, args.sample_rate, num_ch)
            x_train = x_train.transpose(0, 2, 1)
            x_val = x_val.transpose(0, 2, 1)
            x_test = x_test.transpose(0, 2, 1)

            self.x_train_t[:, :num_ch, :] = x_train
            self.x_val_t[:, :num_ch, :] = x_val
            self.x_test_t[:, :num_ch, :] = x_test

        args.num_channels = args.num_channels[:-1]

        super(CichyData, self).set_common()

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        if self.args.save_data:
            for i in range(self.x_train_t.shape[1]):
                dump = {'x_train_t': self.x_train_t[:, i:i+1:, :],
                        'x_val_t': self.x_val_t[:, i:i+1, :],
                        'x_test_t': self.x_test_t[:, i:i+1, :],
                        'sub_id_train': self.sub_id['train'],
                        'sub_id_val': self.sub_id['val'],
                        'sub_id_test': self.sub_id['test']}

                savemat(self.args.dump_data + 'ch' + str(i) + '.mat', dump)

        # save standardscaler
        path = os.path.join('/'.join(self.args.dump_data.split('/')[:-1]),
                            'standardscaler')
        with open(path, 'wb') as file:
            pickle.dump(self.norm, file)

    def splitting(self, dataset, args):
        split_l = int(args.split[0] * dataset.shape[1])
        split_h = int(args.split[1] * dataset.shape[1])
        x_val = dataset[:, split_l:split_h, :, :]
        x_train = dataset[:, :split_l, :, :]
        x_train = np.concatenate((x_train, dataset[:, split_h:, :, :]),
                                 axis=1)

        return x_train, x_val, x_val

    def load_data(self, args):
        '''
        Load trials for each condition from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if isinstance(args.data_path, list):
            paths = args.data_path
        elif 'sub' in args.data_path:
            paths = [args.data_path]
        else:
            paths = os.listdir(args.data_path)
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if os.path.isdir(p)]
            paths = [p for p in paths if 'opt' not in p]
            paths = [p for p in paths if 'sub' in p]

        channels = len(args.num_channels)
        x_trains = []
        x_vals = []
        x_tests = []
        for path in paths:
            print('Loading ', path, flush=True)
            min_trials = 1000000
            dataset = []

            # loop over 118 conditions
            for c in range(args.num_classes):
                cond_path = os.path.join(path, 'cond' + str(c))
                files = os.listdir(cond_path)
                files = [f for f in files if 'npy' in f]
                if len(files) < min_trials:
                    min_trials = len(files)

                trials = []
                # loop over trials within a condition
                for f in files:
                    trial = np.load(os.path.join(cond_path, f))
                    trials.append(trial)

                dataset.append(np.array(trials))

            # condition with lowest number of trials
            print('Minimum trials: ', min_trials, flush=True)

            # dataset shape: conditions x trials x timesteps x channels
            dataset = np.array([t[:min_trials, :, :] for t in dataset])

            # choose first 306 channels
            dataset = dataset.transpose(0, 1, 3, 2)
            dataset = dataset[:, :, args.num_channels, :]
            self.timesteps = dataset.shape[3]

            # create training and validation splits with equal class numbers
            x_train, x_val, x_test = self.splitting(dataset, args)

            # crop training trials
            max_trials = round(args.max_trials * x_train.shape[1])
            x_train = x_train[:, :max_trials, :, :]

            x_train = x_train.transpose(0, 1, 3, 2).reshape(-1, channels)
            x_val = x_val.transpose(0, 1, 3, 2).reshape(-1, channels)
            x_test = x_test.transpose(0, 1, 3, 2).reshape(-1, channels)

            # standardize dataset along channels
            x_train, x_val, x_test = self.normalize(x_train, x_val, x_test)

            x_trains.append(x_train)
            x_vals.append(x_val)
            x_tests.append(x_test)

        # this is just needed to work together with other dataset classes
        disconts = [[0] for path in paths]
        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, x_tests, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''

        # expand shape to trials
        x = x.transpose(1, 0)
        x = x.reshape(self.args.num_classes, -1, self.timesteps, x.shape[1])
        x = x.transpose(0, 1, 3, 2)

        # downsample data if needed
        resample = int(1000/self.args.sr_data)
        x = x[:, :, :, ::resample]
        timesteps = x.shape[3]
        trials = x.shape[1]

        # create labels, and put them in the last channel of the data
        array = []
        labels = np.ones((trials, 1, timesteps))
        for c in range(x.shape[0]):
            array.append(np.concatenate((x[c, :, :, :], labels * c), axis=1))

        x = np.array(array).reshape(-1, x.shape[2] + 1, timesteps)
        return x
