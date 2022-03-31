import random
import numpy as np
import pickle, torch
from . import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._aug(data_numpy)
        return [data1, data2, data3], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy


class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
