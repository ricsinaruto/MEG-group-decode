import os
import torch

from torch.nn import Embedding
from scipy.io import loadmat

from wavenets_simple import WavenetSimple
from classifiers_simpleNN import SimpleClassifier, ClassifierModule


class WavenetClassifier(SimpleClassifier):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def loaded(self, args):
        super(WavenetClassifier, self).loaded(args)
        self.wavenet.loaded(args)

    def kernel_network_FIR(self):
        self.wavenet.kernel_network_FIR()

    def analyse_kernels(self):
        self.wavenet.analyse_kernels()

    def kernelPFI(self, data, sid=None):
        return self.wavenet.kernelPFI(data, sid)

    def build_model(self, args):
        if args.wavenet_class:
            self.wavenet = args.wavenet_class(args)
        else:
            self.wavenet = WavenetSimple(args)

        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)

    def forward(self, x, sid=None):
        '''
        Run wavenet on input then feed the output into the classifier.
        '''
        output, x = self.wavenet(x, sid)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        x = self.classifier(x)

        return output, x


class WavenetClassifierSemb(WavenetClassifier):
    '''
    Wavenet Classifier for multi-subject data using subject embeddings.
    '''
    def set_sub_dict(self):
        # this dictionary is needed because
        # subject embeddings and subjects have a different ordering
        self.sub_dict = {0: 10,
                         1: 7,
                         2: 3,
                         3: 11,
                         4: 8,
                         5: 4,
                         6: 12,
                         7: 9,
                         8: 5,
                         9: 13,
                         10: 1,
                         11: 14,
                         12: 2,
                         13: 6,
                         14: 0}

    def __init__(self, args):
        super(WavenetClassifierSemb, self).__init__(args)
        self.set_sub_dict()

    def loaded(self, args):
        super(WavenetClassifierSemb, self).loaded(args)
        self.set_sub_dict()

        # change embedding to an already trained one
        if 'trained_semb' in args.result_dir:
            path = os.path.join(args.load_model, '..', 'sub_emb.mat')
            semb = torch.tensor(loadmat(path)['X']).cuda()
            self.wavenet.subject_emb.weight = torch.nn.Parameter(semb)

    def build_model(self, args):
        self.wavenet = args.wavenet_class(args)

        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)

    def save_embeddings(self):
        self.wavenet.save_embeddings()

    def get_sid(self, sid):
        '''
        Get subject id based on result directory name.
        '''
        ind = int(self.args.result_dir.split('_')[-1].split('/')[0])
        ind = self.sub_dict[ind]

        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def get_sid_exc(self, sid):
        '''
        Get subject embedding of untrained subject
        '''
        ind = int(self.args.result_dir.split('_')[-1].split('/')[0])
        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def get_sid_best(self, sid):
        ind = 8
        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def ensemble_forward(self, x, sid):
        outputs = []
        for i in range(15):
            subid = torch.LongTensor([i]).repeat(*list(sid.shape)).cuda()
            _, out_class = super(WavenetClassifierSemb, self).forward(x, subid)

            outputs.append(out_class.detach())

        outputs = torch.stack(outputs)
        outputs = torch.mean(outputs, dim=0)

        return None, outputs

    def forward(self, x, sid=None):
        if not self.args.keep_sid:
            if 'sub' in self.args.result_dir:
                sid = self.get_sid(sid)
            if 'exc' in self.args.result_dir:
                sid = self.get_sid_exc(sid)
            if 'best' in self.args.result_dir:
                sid = self.get_sid_best(sid)
            if 'ensemble' in self.args.result_dir:
                return self.ensemble_forward(x, sid)

        return super(WavenetClassifierSemb, self).forward(x, sid)
