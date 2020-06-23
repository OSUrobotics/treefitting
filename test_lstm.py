import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision as tv
import torch
from scipy.linalg import svd
import numpy as np
from ipdb import set_trace
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import defaultdict
import pickle
from numbers import Number
import imageio
from copy import deepcopy
from itertools import product, chain
import hashlib
from ipdb import set_trace
import matplotlib
import random

class WordLoader(torch.utils.data.Dataset):

    DATASET_BOUNDS = {
        'training': (0.0, 0.7),
        'validation': (0.7, 0.9),
        'test': (0.9, 1.0),
    }

    @classmethod
    def from_folder(cls, root='/home/main/data/fake_2d_trees/data'):

        mapping = {
            'b': 1,
            'l': 2,
            's': 3,
            't': 4,
            'o': 5,
            'f': 0,
        }

        dataset = cls()

        files = os.listdir(root)
        for file in files:
            with open(os.path.join(root, file), 'rb') as fh:
                data = pickle.load(fh)
            for _, assignments in data['all_labels']:
                # TODO: Add subsequences in here

                valid = True
                category_sequence = []
                for idx, (letter, truth) in enumerate(assignments):
                    category_idx = mapping[letter]
                    category_ser = np.zeros(len(mapping))
                    category_ser[category_idx] = 1

                    if not truth:
                        valid = False

                    category_sequence.append(category_ser)

                data = {
                    'sequence': torch.Tensor(category_sequence),
                    'truth_array': torch.Tensor([1, 0] if valid else [0, 1]),
                    # TODO: Add positional data here based off sequence
                    # TODO: Add edge image data here for 'edge_image'
                }
                dataset.register_data(data)

        return dataset

    def fixed_length_batch_iterator(self, batch_size=4):
        seq = []
        for seq_len, items in self.counts_by_len.items():
            random.shuffle(items)
            for start_mult in range((len(items) - 1) // batch_size):
                seq.append(items[start_mult * batch_size: (start_mult + 1) * batch_size])
        random.shuffle(seq)
        for subseq in seq:
            yield combine_tensor_dicts([self.data[i] for i in subseq])
        raise StopIteration



    def __init__(self, transform=None):

        super(WordLoader, self).__init__()
        self.transform = transform
        self.data = []
        self.counts_by_len = defaultdict(list)

    def register_data(self, item):
        seq_len = item['sequence'].shape[0]
        idx = len(self.data)
        self.data.append(item)
        self.counts_by_len[seq_len].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        return self.data[i]




class Skeletonizer(nn.Module):
    def __init__(self, hidden_size=10, conv_layers=(8, 16, 32), point_dim = 2,
                 image_size=32, image_nodes=64, image_dim=6, dropout=0.2,
                 post_lstm_nodes = 64, num_classes = 6, bidirectional=True):
        super(Skeletonizer, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Set up convolutional layers to process sequence

        all_layers = []
        augmented_conv_layers = [1, *conv_layers]
        for start, end in zip(augmented_conv_layers[:-1], augmented_conv_layers[1:]):
            conv = nn.Conv2d(start, end, 3)
            relu = nn.ReLU()
            pool = nn.MaxPool2d(2, 2)
            all_layers.extend([conv, relu, pool])

        num_pixels = (image_size // (2 ** len(conv_layers))) ** 2

        self.conv = nn.Sequential(*all_layers)
        self.conv_to_features = nn.Sequential(
            nn.Linear(num_pixels, image_nodes),
            nn.Dropout(dropout),
            nn.Linear(image_nodes, image_dim)
        )

        # Set up LSTM - Takes the linear features for each image and any additional linear features, in this case the
        # edge endpoints 2 * point_dim
        self.lstm = nn.LSTM(image_dim + 2 * point_dim, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.hidden_to_classifications = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), post_lstm_nodes),
            nn.Dropout(dropout),
            nn.Linear(post_lstm_nodes, num_classes + 2)
        )

    def forward(self, batch):
        import ipdb
        ipdb.set_trace()

        # Take images, reshape them to 4-D (sequence_len * batch_size, 1, im_size, im_size)
        batch_len, seq_len, x, y = batch['edge_image'].shape
        imgs = batch['edge_image'].view(batch_len * seq_len, 1, x, y)
        image_features = self.conv(imgs)
        image_reps_sequential = self.conv_to_features(image_features.view(batch_len, seq_len, -1))

        # Append point features to linear features, pass them through LSTM
        # Axis is batch x sequence x feature
        all_features_sequential = torch.cat([image_reps_sequential, batch['start'], batch['end']], 2)
        lstm_output = self.lstm(all_features_sequential)[0]

        # First evaluate classifications for each segment
        # Then evaluate classifications for the entire segment as a whole (which will use only the last hidden layer)

        linear_output = self.hidden_to_classifications(lstm_output)
        classifications = linear_output[:,:,:-2]
        correctness = linear_output[:,-1,-2:]

        return {'classification': classifications,
                'correctness': correctness}


def combine_tensor_dicts(list_of_inputs):
    # Assumes dictionary is not nested
    new_dict = {}
    for k in list_of_inputs[0]:
        all_vals = [i[k] for i in list_of_inputs]
        new_dict[k] = torch.stack(all_vals)
    return new_dict





if __name__ == '__main__':
    data = WordLoader.from_folder()
    net = Skeletonizer().double()

    # ex_images = np.random.uniform(0, 1, (6, 1, 64, 64))
    # tens = torch.from_numpy(ex_images).double()
    # blah = net.test_conv(tens)

    for batch in data.fixed_length_batch_iterator(5):
        print(batch)
        break

