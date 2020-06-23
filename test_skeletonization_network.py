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



class SyntheticTreeClassifier(nn.Module):

    def __init__(self, data_params=None, raster_vec_size=6, local_vec_size=6):
        if data_params is None:
            data_params = {}
        super(SyntheticTreeClassifier, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)

        # Convert raster to a 6-vec
        self.raster_conv_1 = nn.Conv2d(1, 2, 5, padding=2)
        self.raster_conv_2 = nn.Conv2d(2, 4, 5, padding=2)
        self.raster_conv_3 = nn.Conv2d(4, 8, 5, padding=2)
        self.flat_raster_size = (data_params.get('raster_grid_size', 128) // 16) ** 2 * 8
        self.raster_fc = nn.Linear(self.flat_raster_size, 64)
        self.raster_vec = nn.Linear(64, raster_vec_size)

        # Convert edge info to a 6-vec
        self.local_conv_1 = nn.Conv2d(1, 2, 3, padding=1)
        self.local_conv_2 = nn.Conv2d(2, 4, 3, padding=1)
        self.local_conv_3 = nn.Conv2d(4, 8, 3, padding=1)
        self.flat_local_size = (data_params.get('local_grid_size', 32) // 4) ** 2 * 8
        self.local_fc = nn.Linear(self.flat_local_size, 64)
        self.local_vec = nn.Linear(64, local_vec_size)

        # Combine both along with linear info
        self.combination_fc = nn.Linear(local_vec_size + raster_vec_size + 4, 128)
        self.output = nn.Linear(128, data_params.get('num_classes', 4) + 2)

    def forward(self, x):

        # Get edge classification
        local = x['edge_image']
        local = local.view(-1, 1, *local.shape[-2:])
        local = self.pool(nn_func.relu(self.local_conv_1(local)))
        local = self.pool(nn_func.relu(self.local_conv_2(local)))
        local = nn_func.relu(self.local_conv_3(local)).view(-1, self.flat_local_size)
        local = nn_func.relu(self.local_fc(local))
        local = self.local_vec(local)

        rast = x['global_image']
        rast = rast.view(-1, 1, *rast.shape[-2:])
        rast = self.pool_4(nn_func.relu(self.raster_conv_1(rast)))
        rast = self.pool_4(nn_func.relu(self.raster_conv_2(rast)))
        rast = nn_func.relu(self.raster_conv_3(rast))
        flattened_rast = rast.view(-1, self.flat_raster_size)
        rast = nn_func.relu(self.raster_fc(flattened_rast))
        rast = self.raster_vec(rast)

        # Combine into final
        final = torch.cat([local, rast, x['node_a'], x['node_b']], 1)
        final = nn_func.relu(self.combination_fc(final))
        final = self.output(final)

        return final

    def guess_from_export_dataset(self, dataset):

        val_loader = torch_data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        edge_indexes = []
        numpy_rez = []
        self.eval()

        for data in val_loader:
            rez = self.forward(data)
            edge_indexes.extend(data['edge_id'].numpy())
            numpy_rez.append(torch.sigmoid(rez).detach().numpy())

        self.train()

        return np.concatenate(numpy_rez), edge_indexes


class TreeDataset(torch.utils.data.Dataset):

    @classmethod
    def from_dict(cls, data_dict):
        dataset = cls()
        dataset.process_dict(deepcopy(data_dict))
        return dataset

    @classmethod
    def from_directory(cls, root, set_type='training'):

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

        np.random.seed(0)

        all_files = list(filter(lambda x: x.endswith('.info'), sorted(os.listdir(root))))
        rand = np.random.uniform(size=len(all_files))
        to_select = np.where((proportions[0] <= rand) & (rand < proportions[1]))[0]
        files_to_load = [all_files[i] for i in to_select]

        for i, file in enumerate(files_to_load):

            with open(os.path.join(root, file), 'rb') as fh:
                data_dict = pickle.load(fh)
                dataset.process_dict(data_dict)

        np.random.seed(None)
        return dataset

    def __init__(self, transform=None):

        super(TreeDataset, self).__init__()
        self.transform = transform
        self.global_info = {}       # i
        self.superpoint_info = {}   # i, a
        self.edge_info = []         # (i, a, b), data

    def __len__(self):
        return len(self.edge_info)

    def __getitem__(self, i):

        (data_i, a_i, b_i), data = self.edge_info[i]
        return {
            'edge_id': i,
            'edge': torch.Tensor([a_i, b_i]),
            'global_image': self.global_info[data_i]['image'],
            'node_a': self.superpoint_info[data_i, a_i]['location'],
            'node_b': self.superpoint_info[data_i, b_i]['location'],
            'edge_image': data['image'],
            'category': data['category'],
            'connected': data['connected']
        }

    def process_dict(self, data_dict):
        # Process global data
        i = len(self.global_info)
        global_dict = data_dict['global']
        convert_dict_to_torch(global_dict)
        self.global_info[i] = global_dict

        # Process superpoint-based information
        superpoint_info = data_dict['superpoints']
        convert_dict_to_torch(superpoint_info)
        for node_id, data in superpoint_info.items():
            self.superpoint_info[i, node_id] = data

        # Process edge-based information
        edge_data = data_dict['edges']
        for edge_dict in edge_data:
            node_a, node_b = edge_dict.pop('edge')
            convert_dict_to_torch(edge_dict)
            edge = (i, node_a, node_b)
            self.edge_info.append([edge, edge_dict])


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
    train_data = TreeDataset.from_directory('experimental_training_data', 'training')
    train_loader = torch_data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)

    val_data = TreeDataset.from_directory('experimental_training_data', 'validation')
    val_loader = torch_data.DataLoader(val_data, batch_size=10, shuffle=False, num_workers=0)


    net = SyntheticTreeClassifier().double()
    if load:
        with open('synthetic_best.model', 'rb') as fh:
            state_dict = torch.load(fh)
        net.load_state_dict(state_dict)
        print('Loaded existing model weights!')

    criterion = nn.CrossEntropyLoss()
    criterion_cat = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    last_val_accuracy = 0
    best_acc = 0
    not_improved_streak = 0

    total = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        print('Starting Epoch {}...'.format(epoch))
        for iter, data in enumerate(train_loader):

            if not iter % 100:
                print('{}/{}'.format(iter, total))

            optimizer.zero_grad()

            outputs = net(data)
            category_prediction = outputs[:,:4]
            connect_prediction = outputs[:,4:6]

            connect_true = torch.max(data['connected'], 1)[1]
            category_true = torch.max(data['category'], 1)[1]
            # First compute connectedness estimates
            connected_loss = criterion(connect_prediction, connect_true)

            # Next, compute loss on categories, but only for values where connected
            is_connected = data['connected'][:,0].bool()
            num_connected = is_connected.sum()
            if num_connected:
                category_loss = criterion_cat(category_prediction[is_connected], category_true[is_connected]) / num_connected
                connected_loss.backward(retain_graph=True)
                category_loss.backward()
            else:
                connected_loss.backward()

            optimizer.step()


        # train_acc, _ = eval_net(net, train_loader)
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
            torch.save(net.state_dict(), 'synthetic_best.model')
            print('Saved best model!')

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

        category_prediction = torch.max(outputs[:, :4], 1)[1]
        connect_prediction = torch.max(outputs[:, 4:6], 1)[1]

        connect_true = torch.max(data['connected'], 1)[1]
        category_true = torch.max(data['category'], 1)[1]

        connect_total += len(connect_true)
        connect_correct += (connect_prediction == connect_true).sum().item()

        real_connect = data['connected'][:,0].bool()
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

    train_net(500, 500, load=True)

