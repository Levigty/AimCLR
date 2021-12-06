import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.pku_read_skeleton import read_xyz

view_dic = {1:"L", 2: "M", 3:"R"}

max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    training_split_file_folder = 'E:/2021/datasets/original_data/pkummd/v1/'
    if benchmark=='xview':
        training_split_file = training_split_file_folder + "cross-view.txt"
    else:
        training_split_file = training_split_file_folder + "cross-subject.txt"
    f = open(training_split_file, "r")
    f.readline()
    training_split = f.readline()
    f.readline()
    testing_split = f.readline()

    training_set = training_split.split(',')
    training_set = [s.strip() for s in training_set]
    training_set = [s for s in training_set if s]

    testing_set = testing_split.split(',')
    testing_set = [s.strip() for s in testing_set]
    testing_set = [s for s in testing_set if s]

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        file_id = int(
            filename[filename.find('F') + 1:filename.find('F') + 4])
        view_id = int(
            filename[filename.find('V') + 1:filename.find('V') + 4])
        action_class = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        split_train_test_name = '0'+filename[filename.find('F') + 1:filename.find('F') + 4] + '-' + view_dic[view_id]

        if (split_train_test_name in training_set) and (split_train_test_name in testing_set):
            raise ValueError()
        istraining = (split_train_test_name in training_set)

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PKU-MMD Part1 Data Converter.')
    parser.add_argument(
        '--data_path', default='E:/2021/datasets/original_data/pkummd/skeleton_pku_v1')
    parser.add_argument('--out_folder', default='E:/2021/datasets/crossclr_pkummd/pku_part1')

    parser.add_argument('--ignored_sample_path',
                        default='E:/2021/datasets/original_data/pkummd/v1/samples_with_missing_skeletons.txt')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)