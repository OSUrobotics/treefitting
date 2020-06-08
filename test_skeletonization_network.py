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

        # Convert local info to a 6-vec
        self.local_conv_1 = nn.Conv2d(1, 2, 3, padding=1)
        self.local_conv_2 = nn.Conv2d(2, 4, 3, padding=1)
        self.local_conv_3 = nn.Conv2d(4, 8, 3, padding=1)
        self.flat_local_size = (data_params.get('local_grid_size', 32) // 4) ** 2 * 8
        self.local_fc = nn.Linear(self.flat_local_size, 64)
        self.local_vec = nn.Linear(64, local_vec_size)

        # Combine both along with linear info
        self.combination_fc = nn.Linear(raster_vec_size + local_vec_size * 2 + 6, 128)
        self.output = nn.Linear(128, data_params.get('num_classes', 5))

    def forward(self, x):

        # Get classifications for node_a and node_b
        rez = []
        for local_info in [x['node_a'], x['node_b']]:
            local = local_info['image']
            local = local.view(-1, 1, *local.shape[-2:])
            local = self.pool(nn_func.relu(self.local_conv_1(local)))
            local = self.pool(nn_func.relu(self.local_conv_2(local)))
            local = nn_func.relu(self.local_conv_3(local)).view(-1, self.flat_local_size)
            local = nn_func.relu(self.local_fc(local))
            local = self.local_vec(local)
            rez.append(local)
        a, b = rez

        rast = x['global_image']
        rast = rast.view(-1, 1, *rast.shape[-2:])
        rast = self.pool_4(nn_func.relu(self.raster_conv_1(rast)))
        rast = self.pool_4(nn_func.relu(self.raster_conv_2(rast)))
        rast = nn_func.relu(self.raster_conv_3(rast))
        flattened_rast = rast.view(-1, self.flat_raster_size)
        rast = nn_func.relu(self.raster_fc(flattened_rast))
        rast = self.raster_vec(rast)

        # Combine into final
        final = torch.cat([rast, a, b, x['node_a']['location'], x['node_b']['location'],
                           x['node_a']['scale'].double(), x['node_b']['scale'].double()], 1)
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

    def __init__(self, set_type='training', root='experimental_training_data', transform=None):

        self.transform = transform
        default_props = {
            'training': [0.0, 0.7],
            'validation': [0.7, 0.9],
            'testing': [0.9, 1.0],
        }
        self.all_edges = []

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
        self.all_data = []
        for i, file in enumerate(files_to_load):
            with open(os.path.join(root, file), 'rb') as fh:
                data_dict = pickle.load(fh)
            edges = data_dict.pop('edges')
            self.all_edges.extend([(i, a_i, b_i) for a_i, b_i in edges])
            convert_dict_to_torch(data_dict)
            self.all_data.append(data_dict)

        np.random.seed(None)

    def __len__(self):
        return len(self.all_edges)

    def __getitem__(self, i):
        data_i, (a_i, b_i), truth = self.all_edges[i]
        dataset = self.all_data[data_i]
        return {
            'edge_id': i,
            'global_image': dataset['global']['image'],
            'node_a': dataset['superpoints'][a_i],
            'node_b': dataset['superpoints'][b_i],
            'truth': truth
        }

class TreeDatasetFromExportDict(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.data_dict = deepcopy(data_dict)
        self.edges = self.data_dict.pop('edges')
        convert_dict_to_torch(self.data_dict)

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, i):
        (a_i, b_i), truth = self.edges[i]
        return {
            'edge_id': i,
            'global_image': self.data_dict['global']['image'],
            'node_a': self.data_dict['superpoints'][a_i],
            'node_b': self.data_dict['superpoints'][b_i],
            'truth': truth
        }


def convert_dict_to_torch(data_dict, add_dim=False):
    for key, val in data_dict.items():
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



def train_net(max_epochs=5, no_improve_threshold=999, lr=1e-4):
    """

    :param max_epochs: Max number of epochs to train
    :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
    :return:
    """
    train_data = TreeDataset('training')
    train_loader = torch_data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)

    val_data = TreeDataset('validation')
    val_loader = torch_data.DataLoader(val_data, batch_size=10, shuffle=False, num_workers=0)


    net = SyntheticTreeClassifier().double()
    criterion = nn.CrossEntropyLoss()
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

            classification = torch.max(data['truth'], 1)[1]
            optimizer.zero_grad()

            outputs = net(data)

            loss = criterion(outputs, classification)
            loss.backward()
            optimizer.step()


        train_acc, _ = eval_net(net, train_loader)
        acc, acc_by_label = eval_net(net, val_loader)

        if acc > last_val_accuracy:
            not_improved_streak = 0
        else:
            not_improved_streak += 1
            print('Accuracy has not improved for {} epochs'.format(not_improved_streak))
            if not_improved_streak >= no_improve_threshold:
                print('Stopping training')
                break

        print('Epoch {} training accuracy:   {:.2f}%'.format(epoch, train_acc * 100))
        print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, acc*100))

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), 'synthetic_best.model')
            print('Saved best model!')

        last_val_accuracy = acc

        if not (epoch % 10):
            print('--Current best validation acc is {:.2f}%'.format(best_acc * 100))

        print('Accuracy by label:')
        for label, label_acc in acc_by_label.items():
            print('\t{}: {:.2f}%'.format(label, label_acc * 100))


def eval_net(net, dataloader):
    net.eval()

    total = 0
    correct = 0

    total_per = defaultdict(lambda: 0)
    correct_per = defaultdict(lambda: 0)

    for data in dataloader:

        classification = torch.max(data['truth'], 1)[1]
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        total += classification.size(0)
        correct += (predicted == classification.data).sum().item()

        for label, prediction in zip(classification.data, predicted):
            label = label.item()
            prediction = prediction.item()
            total_per[label] += 1
            if prediction == label:
                correct_per[label] += 1
    net.train()

    final = {k: correct_per.get(k, 0) / total_per[k] for k in total_per}

    return correct / total, final

if __name__ == '__main__':

    train_net(500, 500)

