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
from itertools import product, chain, permutations
import hashlib
from ipdb import set_trace
import matplotlib
from scipy.spatial import KDTree
import random

class WordLoader(torch.utils.data.Dataset):

    DATASET_BOUNDS = {
        'training': (0.0, 0.7),
        'validation': (0.7, 0.9),
        'test': (0.9, 1.0),
    }

    @classmethod
    def from_folder(cls, root='/home/main/data/fake_2d_trees/data', max_per_tree = 500):

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
            print('Loading {}...'.format(file))
            with open(os.path.join(root, file), 'rb') as fh:
                data = pickle.load(fh)

            graph = data['graph']
            points = data['points']
            tree = KDTree(points, 200)
            pt_subsets = {}
            edge_images_cache = {}

            for _ in range(max_per_tree):
                endpoints, assignments = random.choice(data['all_labels'])

                # Encourage data selection to pick longer sequences
                possible_lens = np.arange(1, len(assignments) + 1)
                wgts = possible_lens / possible_lens.sum()

                chosen_len = possible_lens[np.random.choice(len(possible_lens), p=wgts)]
                possible_starts = np.arange(0, len(assignments) - chosen_len + 1)
                start = possible_starts[np.random.choice(len(possible_starts))]
                end = start + chosen_len

                valid = True
                category_sequence = []
                edge_starts = []
                edge_ends = []
                edge_images = []

                assignments_subsetted = assignments[start:end+1]
                for letter, truth in assignments_subsetted:
                    category_idx = mapping[letter]
                    category_ser = np.zeros(len(mapping))
                    category_ser[category_idx] = 1

                    if not truth:
                        valid = False

                    start_idx = endpoints[0]
                    end_idx = endpoints[1]
                    start = graph.nodes[start_idx]['point']
                    end = graph.nodes[end_idx]['point']

                    edge_starts.append(torch.Tensor(start))
                    edge_ends.append(torch.Tensor(end))
                    category_sequence.append(category_ser)

                    # Construct image
                    r = np.linalg.norm(start-end)/2 * 1.01      # Add small fudge factor
                    edge_pt_indexes = set()
                    for pt in [start, end]:
                        pt_tuple = tuple(pt)
                        subpts = pt_subsets.get(pt_tuple)
                        if subpts is None:
                            subpts = tree.query_ball_point(pt, r)
                            pt_subsets[pt_tuple] = subpts
                        edge_pt_indexes.update(subpts)

                    id_tuple = (tuple(start), tuple(end))
                    edge_points = edge_images_cache.get(id_tuple)
                    if edge_points is None:
                        edge_points = points[list(edge_pt_indexes)]
                        edge_images_cache[id_tuple] = edge_points

                    img = torch.Tensor(convert_to_image(edge_points, start, end, size=(32, 16)))
                    edge_images.append(img)


                data_point = {
                    'sequence': torch.Tensor(category_sequence),
                    'truth_array': torch.Tensor([1, 0] if valid else [0, 1]),
                    'start': torch.stack(edge_starts).double(),
                    'end': torch.stack(edge_ends).double(),
                    'edge_image': torch.stack(edge_images).double(),
                }
                dataset.register_data(data_point)

        return dataset

    def fixed_length_batch_iterator(self, batch_size=4):
        seq = []
        for seq_len, items in self.counts_by_len.items():
            random.shuffle(items)
            for start_mult in range((len(items) - 1) // batch_size + 1):
                seq.append(items[start_mult * batch_size: (start_mult + 1) * batch_size])
        random.shuffle(seq)
        return (combine_tensor_dicts([self.data[i] for i in subseq]) for subseq in seq)
        # for subseq in seq:
        #     yield combine_tensor_dicts([self.data[i] for i in subseq])
        # raise StopIteration



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
    def __init__(self, hidden_size=10, conv_layers=(8, 32), point_dim = 2,
                 image_size=(32, 16), image_nodes=64, image_dim=6, dropout=0.3,
                 post_lstm_nodes = 64, num_classes = 6, bidirectional=True):
        super(Skeletonizer, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Set up convolutional layers to process sequence

        all_layers = []
        augmented_conv_layers = [1, *conv_layers]
        for start, end in zip(augmented_conv_layers[:-1], augmented_conv_layers[1:]):
            conv = nn.Conv2d(start, end, 3, padding=1)
            relu = nn.ReLU()
            pool = nn.MaxPool2d(2, 2)
            all_layers.extend([conv, relu, pool])

        num_pixels = image_size[0] * image_size[1] // (2 ** (len(conv_layers) * 2))

        self.conv = nn.Sequential(*all_layers)
        self.conv_to_features = nn.Sequential(
            nn.Linear(num_pixels * conv_layers[-1], image_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_nodes, image_dim),
            nn.ReLU(),      # Is this one necessary?
        )

        # Set up LSTM - Takes the linear features for each image and any additional linear features, in this case the
        # edge endpoints 2 * point_dim
        self.lstm = nn.LSTM(image_dim + 2 * point_dim, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.hidden_to_classifications = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), post_lstm_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(post_lstm_nodes, num_classes + 2)
        )

        stringify = lambda x: '-'.join([str(y) for y in x])
        self.model_name = 'hs{}_cl{}_{}d_im{}_imnode{}_imdim{}_lstmnode{}_{}class{}'.format(
            hidden_size, stringify(conv_layers), point_dim, stringify(image_size), image_nodes, image_dim,
            post_lstm_nodes, num_classes, '_bidir' if bidirectional else ''
        )

    def load_model(self):
        try:
            with open(self.model_name + '.model', 'rb') as fh:
                state_dict = torch.load(fh)
            self.load_state_dict(state_dict)
            print('Loaded existing model weights!')
        except:
            print('No existing model weights.')

    def save_model(self, suffix=''):
        name = self.model_name + suffix + '.model'
        torch.save(self.state_dict(), name)
        print('Model {} saved!'.format(name))


    def forward(self, batch):

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


def convert_to_image(points, start, end, size=(32, 16)):
    # Equivalent of project_point_onto_plane

    x_axis = end - start
    r = np.linalg.norm(start - end) / 2

    t = np.arctan2(x_axis[1], x_axis[0])
    tf_mat = np.array([
        [np.cos(t), -np.sin(t), start[0]],
        [np.sin(t), np.cos(t), end[1]],
        [0, 0, 1]
    ])

    pts_homog = np.ones((points.shape[0], 3))
    pts_homog[:,0:2] = points
    new_pts = np.linalg.inv(tf_mat).dot(pts_homog.T).T[:,0:2] / r   # x from 0 to 1, y from -0.5 to 0.5
    x_lims = np.linspace(-0.5, 1.5, size[0] + 1)
    y_lims = np.linspace(-0.5, 0.5, size[1] + 1)
    img = np.histogram2d(new_pts[:,0], new_pts[:,1], bins=[x_lims, y_lims])[0]
    return img


def combine_tensor_dicts(list_of_inputs):
    # Assumes dictionary is not nested
    new_dict = {}
    for k in list_of_inputs[0]:
        all_vals = [i[k] for i in list_of_inputs]
        new_dict[k] = torch.stack(all_vals)
    return new_dict



def train_net(max_epochs=5, no_improve_threshold=999, lr=1e-4, batch=4, load=False):
    """

    :param max_epochs: Max number of epochs to train
    :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
    :return:
    """
    train_data = WordLoader.from_folder()
    val_data = WordLoader.from_folder('/home/main/data/fake_2d_trees/data_validation')

    net = Skeletonizer().double()
    if load:
        net.load_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    last_val_accuracy = 0
    best_acc = 0
    not_improved_streak = 0


    total = len(train_data) // batch

    for epoch in range(1, max_epochs + 1):
        print('Starting Epoch {}...'.format(epoch))
        for iter, data in enumerate(train_data.fixed_length_batch_iterator(batch)):
            if not iter % 100:
                print('{}/{}'.format(iter, total))

            optimizer.zero_grad()

            outputs = net(data)
            batch_size, sequence_size, feature_size = outputs['classification'].shape
            category_prediction = outputs['classification'].view(-1, 6)
            is_valid_prediction = outputs['correctness']


            cat_true = torch.max(data['sequence'].view(-1, 6), 1)[1]
            is_valid_true = torch.max(data['truth_array'], 1)[1]

            category_loss = criterion(category_prediction, cat_true) * batch * (sequence_size) ** 0.5
            truth_loss = criterion(is_valid_prediction, is_valid_true) * batch

            category_loss.backward(retain_graph=True)
            truth_loss.backward()
            # category_loss.backward()

            optimizer.step()

        # train_acc, _ = eval_net(net, train_loader)
        acc_rez = eval_net(net, val_data)
        acc = (acc_rez['category_accuracy'] + acc_rez['valid_accuracy']) / 2

        if acc > last_val_accuracy:
            not_improved_streak = 0
        else:
            not_improved_streak += 1
            print('Accuracy has not improved for {} epochs'.format(not_improved_streak))
            if not_improved_streak >= no_improve_threshold:
                print('Stopping training')
                break

        print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, acc*100))
        print('({:.2f}% valid, {:.2f}% category)'.format(acc_rez['valid_accuracy'] * 100, acc_rez['category_accuracy'] * 100))


        if acc > best_acc:
            best_acc = acc
            net.save_model()
            print('Saved best model!')

        last_val_accuracy = acc

        if not (epoch % 10):
            print('--Current best validation acc is {:.2f}%'.format(best_acc * 100))


def eval_net(net, dataset):
    net.eval()

    total = 0
    valid_correct = 0
    category_correct = 0

    predictions = []
    reals = []

    for data in dataset.fixed_length_batch_iterator(4):
        with torch.no_grad():

            outputs = net(data)
            category_prediction = torch.max(outputs['classification'].view(-1, 6), 1)[1]
            is_valid_prediction = torch.max(outputs['correctness'], 1)[1]

            cat_true = torch.max(data['sequence'].view(-1, 6), 1)[1]
            is_valid_true = torch.max(data['truth_array'], 1)[1]

            total += len(data)

            cat_acc = (category_prediction == cat_true).float().mean().item()
            valid_acc = (is_valid_prediction == is_valid_true).float().mean().item()

            valid_correct += valid_acc * len(data)
            category_correct += cat_acc * len(data)

            predictions.extend(category_prediction.numpy())
            reals.extend(cat_true.numpy())



    net.train()

    from collections import Counter
    print('Counts by prediction:\n{}'.format(Counter(predictions)))
    print('Counts by truth:\n{}'.format(Counter(reals)))

    rez = {
        'category_accuracy': category_correct / total,
        'valid_accuracy': valid_correct / total,
        'total': total
    }

    return rez










if __name__ == '__main__':
    train_net(1000, 1000, load=False, lr=1e-3)
