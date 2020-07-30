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
from copy import deepcopy
import hashlib


class RealTreeClassifier(nn.Module):

    def __init__(self, data_params=None):

        defaults = {
            'point_dim': 0,

            'raster_size': (128, 128),
            # 'raster_convs': (4, 'bn', -4, 8, 'bn', -4, 16, 'bn'),
            'raster_convs': (16, 16, 'bn', -2, 16, 16, 'bn', -2, 32, 32, 'bn', -2, 32, 32, 'bn', -2, 64, 64),
            'raster_fc': (64,),
            'raster_vec_size': 6,

            'local_size': (32, 16),
            # 'local_convs': (2, 'bn', -2, 4, 'bn', -2, 8, 'bn'),
            'local_convs': (16, 16, 'bn', -2, 32, 32, 'bn', -2, 64, 64),
            'local_fc': (64,),
            'local_vec_size': 6,

            'final_fc': (128, 0.25, 128, 0.25),
            'num_classes': 5,
        }

        settings = defaults.copy()
        self.settings = settings
        self.num_classes = settings['num_classes']
        if data_params is not None:
            settings.update(data_params)

        super(RealTreeClassifier, self).__init__()

        self.full_name = '__'.join(['{}_{}'.format(k,settings[k]) for k in sorted(settings)])
        self.name = hashlib.md5(self.full_name.encode()).hexdigest()

        # Create Conv and FC layers for global raster
        self.raster_conv, raster_shrinkage, final_channels = self.process_conv_structure(settings['raster_convs'], 2)
        self.raster_flat_size = np.prod(settings['raster_size']) // raster_shrinkage ** 2 * final_channels
        self.raster_fc = self.process_fc_structure(settings['raster_fc'], self.raster_flat_size, settings['raster_vec_size'])

        # Create Conv and FC layers for local raster
        self.local_conv, local_shrinkage, final_channels = self.process_conv_structure(settings['local_convs'], 1)
        self.local_flat_size = np.prod(settings['local_size']) // local_shrinkage ** 2 * final_channels
        self.local_fc = self.process_fc_structure(settings['local_fc'], self.local_flat_size, settings['local_vec_size'])

        # Create FC which connects layers together plus adding in point dimensions if necessary
        linear_start = settings['local_vec_size'] + settings['raster_vec_size'] + settings['point_dim'] * 2
        linear_end = settings['num_classes'] + 2
        self.final_fc = self.process_fc_structure(settings['final_fc'], linear_start, linear_end)

    @classmethod
    def process_conv_structure(cls, conv_tuple, starting_channels):

        all_modules = []
        last_channels = starting_channels
        shrinkage = 1
        for val in conv_tuple:
            if val == 'bn':
                all_modules.append(nn.BatchNorm2d(last_channels))
            elif val > 0:
                all_modules.append(nn.Conv2d(last_channels, val, 3, padding=1))
                all_modules.append(nn.ReLU())
                last_channels = val
            else:   # Maxpooling indicated by negative
                val = abs(val)
                all_modules.append(nn.MaxPool2d(val, val))
                shrinkage *= val
        return nn.Sequential(*all_modules), shrinkage, last_channels

    @classmethod
    def process_fc_structure(cls, fc_tuple, starting_vals, ending_vals):
        all_modules = []
        last_layer = starting_vals
        for val in fc_tuple:
            if isinstance(val, float):
                all_modules.append(nn.Dropout(val))
            else:
                all_modules.append(nn.Linear(last_layer, val))
                all_modules.append(nn.ReLU())
                last_layer = val
        all_modules.append(nn.Linear(last_layer, ending_vals))
        return nn.Sequential(*all_modules)


    def forward(self, x):

        # Get edge classification
        local = x['local_image']
        local = local.view(-1, 1, *local.shape[-2:])
        local = self.local_conv(local).view(local.shape[0], -1)
        local = self.local_fc(local)

        rast = x['global_image'].permute(0, 3, 1, 2)
        rast = self.raster_conv(rast).view(rast.shape[0], -1)
        rast = self.raster_fc(rast)

        # Combine into final
        if self.settings['point_dim']:
            final = torch.cat([local, rast, x['start'], x['end']], 1)
        else:
            final = torch.cat([local, rast], 1)
        final = self.final_fc(final)
        return final

    def load(self, suffix=''):
        full_name = 'models/{}{}.model'.format(self.name, '_{}'.format(suffix) if suffix else '')
        with open(full_name, 'rb') as fh:
            state_dict = torch.load(fh)
        self.load_state_dict(state_dict)

    def save(self, suffix=''):
        full_name = 'models/{}{}.model'.format(self.name, '_{}'.format(suffix) if suffix else '')
        torch.save(self.state_dict(), full_name)
        print('Saved model to {}'.format(full_name))

    def guess_all(self, data_loader):
        self.eval()
        with torch.no_grad():
            all_rez = []

            for data_batch in data_loader:
                rez = torch.sigmoid(self(data_batch))
                all_rez.append(rez.numpy())

        return np.concatenate(all_rez)


class TreeDataset(torch.utils.data.Dataset):

    NUMERIC_KEYS = [
        'classification',
        'connected',
        'global_image',
        'local_image',
        'start',
        'end',
    ]


    @classmethod
    def from_superpoint_graph(cls, graph):
        dataset = cls()

        for a, b in graph.edges:
            e = graph.edges[a,b]
            data_dict = {
                'global_image': e['global_image'],
                'local_image': e['local_image'],
                'start': graph.nodes[a]['point'],
                'end': graph.nodes[b]['point']

            }
            convert_dict_to_torch(data_dict)

            dataset.data.append(data_dict)
            dataset.ids.append((a,b))

        return dataset


    @classmethod
    def from_directory(cls, root, include_auxiliary=True, set_type='training'):

        dataset = cls()

        default_props = {
            'training': [0.0, 0.7],
            'validation': [0.7, 0.9],
            'testing': [0.9, 1.0],
        }
        if isinstance(set_type, str):
            proportions = default_props[set_type]
        elif isinstance(set_type, list):
            proportions = set_type      # Pass in your own bounds, like [0, 0.5]
        else:
            raise ValueError("Did not understand set_type {}".format(set_type))

        all_files = sorted([file for file in os.listdir(root) if file.endswith('.tree')])

        np.random.seed(0)
        rand = np.random.uniform(size=len(all_files))
        to_select = np.where((proportions[0] <= rand) & (rand < proportions[1]))[0]
        files_to_load = [all_files[i] for i in to_select]

        if include_auxiliary:

            new_files = []
            aux_folder = os.path.join(root, 'auxiliary')
            aux_files = os.listdir(aux_folder)

            for file in files_to_load:
                valid = list(filter(lambda x: x.startswith(file), aux_files))
                new_files.extend(map(lambda x: os.path.join('auxiliary', x), valid))

            files_to_load.extend(new_files)

        for i, file in enumerate(files_to_load):

            with open(os.path.join(root, file), 'rb') as fh:
                data_dict = pickle.load(fh)
            data_dict = {k: data_dict[k] for k in cls.NUMERIC_KEYS}
            convert_dict_to_torch(data_dict)
            dataset.data.append(data_dict)

        np.random.seed(None)

        dataset.ids = files_to_load
        return dataset

    def __init__(self, transform=None):

        super(TreeDataset, self).__init__()
        self.transform = transform
        self.data = []
        self.ids = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def convert_dict_to_torch(data_dict, add_dim=False, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()
    for key, val in data_dict.items():
        if key in ignore_keys:
            continue
        if isinstance(val, np.ndarray):
            data_dict[key] = torch.from_numpy(val)
            if add_dim:
                data_dict[key] = data_dict[key].view(1, *data_dict[key].shape)
        elif isinstance(val, dict):
            convert_dict_to_torch(val, add_dim=add_dim)
        elif isinstance(val, Number):
            data_dict[key] = torch.Tensor([val]).double()
        else:
            raise TypeError('Reached unknown nested type {}!'.format(type(val).__name__))



def train_net(max_epochs=5, no_improve_threshold=999, lr=1e-4, load=False):
    """

    :param max_epochs: Max number of epochs to train
    :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
    :return:
    """
    net = RealTreeClassifier().double()

    train_data = TreeDataset.from_directory('/home/main/data/tree_edge_data', True, [0, 0.8])
    train_loader = torch_data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)

    val_data = TreeDataset.from_directory('/home/main/data/tree_edge_data', True, [0.8, 1.0])
    val_loader = torch_data.DataLoader(val_data, batch_size=10, shuffle=False, num_workers=0)



    if load:
        try:
            net.load()
        except FileNotFoundError:
            print('No model currently exists')

    criterion = nn.CrossEntropyLoss()
    criterion_cat = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    last_val_accuracy = 0
    best_acc = 0
    not_improved_streak = 0

    total = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        print('Starting Epoch {}...'.format(epoch))
        net.train()
        for iter, data in enumerate(train_loader):

            if not iter % 100:
                print('{}/{}'.format(iter, total))

            optimizer.zero_grad()

            outputs = net(data)
            category_prediction = outputs[:,:net.num_classes]
            connect_prediction = outputs[:,net.num_classes:net.num_classes+2]

            connect_true = torch.max(data['connected'], 1)[1]
            category_true = torch.max(data['classification'], 1)[1]
            # First compute connectedness estimates
            connected_loss = criterion(connect_prediction, connect_true)

            # Next, compute loss on categories, but only for values where connected
            is_connected = data['connected'][:,1].bool()
            num_connected = is_connected.sum()
            if num_connected:
                # TODO: Try changing the subsetting to a torch.where and see if that fixes the training? Though not necessary now


                category_loss = criterion_cat(category_prediction[is_connected], category_true[is_connected]) / num_connected
                connected_loss.backward(retain_graph=True)
                category_loss.backward()
            else:
                connected_loss.backward()

            optimizer.step()


        # train_acc, _ = eval_net(net, train_loader)
        with torch.no_grad():
            acc_rez = eval_net(net, val_loader)

        conn_acc = acc_rez['connected_accuracy']
        cat_acc = acc_rez['category_accuracy']
        acc_by_label = acc_rez['per_category_accuracy']
        acc = 2/3 * conn_acc + 1/3 * cat_acc


        if acc > last_val_accuracy:
            not_improved_streak = 0
        else:
            not_improved_streak += 1
            print('Accuracy has not improved for {} epochs'.format(not_improved_streak))
            if not_improved_streak >= no_improve_threshold:
                print('Stopping training')
                break

        # print('Epoch {} training accuracy:   {:.2f}%'.format(epoch, train_acc * 100))

        print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, acc*100))
        print('({:.2f}% connectedness, {:.2f}% category)'.format(conn_acc * 100, cat_acc * 100))


        if acc > best_acc:
            best_acc = acc
            net.save()

        last_val_accuracy = acc

        if not (epoch % 10):
            print('--Current best validation acc is {:.2f}%'.format(best_acc * 100))

        print('Accuracy by label:')
        for label, label_acc in acc_by_label.items():
            print('\t{}: {:.2f}% (out of {})'.format(label, label_acc * 100, acc_rez['per_category_count'][label]))

def eval_net(net, dataloader):
    net.eval()

    connect_total = 0
    connect_correct = 0

    category_total = 0
    category_correct = 0

    total_per = defaultdict(lambda: 0)
    correct_per = defaultdict(lambda: 0)

    for data in dataloader:

        outputs = net(data)

        category_prediction = torch.max(outputs[:, :net.num_classes], 1)[1]
        connect_prediction = torch.max(outputs[:, net.num_classes:net.num_classes+2], 1)[1]

        connect_true = torch.max(data['connected'], 1)[1]
        category_true = torch.max(data['classification'], 1)[1]

        connect_total += len(connect_true)
        connect_correct += (connect_prediction == connect_true).sum().item()

        real_connect = data['connected'][:,1].bool()
        category_prediction = category_prediction[real_connect]
        category_true = category_true[real_connect]

        category_total += len(category_prediction)
        category_correct += (category_prediction == category_true).sum().item()

        for label, prediction in zip(category_true, category_prediction):
            label = label.item()
            prediction = prediction.item()
            total_per[label] += 1
            if prediction == label:
                correct_per[label] += 1
    net.train()

    final = {k: correct_per.get(k, 0) / total_per[k] for k in total_per}

    rez = {
        'connected_accuracy': connect_correct / connect_total,
        'category_accuracy': category_correct / category_total,
        'per_category_accuracy': final,
        'per_category_count': total_per,
    }

    return rez

if __name__ == '__main__':

    train_net(500, 500, lr=1e-4, load=False)

