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
from copy import deepcopy
from utils import project_point_onto_plane, project_points_onto_normal

def convert_pc_to_grid(pc, reference_point, grid_size=16, v=None, return_scale=False):
    """
    Takes a point cloud and projects it into the plane defined by its two most significant SVD components
    :param pc: An Nx3 point cloud
    :param reference_point: A 3-vector representing the "center" of the image
    :param grid_size: An integer or tuple of integers for the image grid size
    :param v: A v matrix produced from an SVD. Set to None for it to be automatically computed.
    :return: A normalized grid defined by the grid size
    """

    if v is None:
        centroid = pc.mean(axis=0)
        _, _, v = svd(pc - centroid)


    proj_2d = project_point_onto_plane(reference_point, v[0,:], v[1,:], pc)
    min_x = proj_2d[:,0].min()
    max_x = proj_2d[:,0].max()
    min_y = proj_2d[:,1].min()
    max_y = proj_2d[:,1].max()


    scaling = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
    bounds = np.linspace(-scaling, scaling, grid_size+1)
    output = np.histogram2d(proj_2d[:,0], proj_2d[:,1], bounds)[0]
    output = output / np.max(output)
    if return_scale:
        return output, scaling
    else:
        return output



class NewCloudClassifier(nn.Module):

    @classmethod
    def from_data_file(cls, data_file_or_loc, load_model=None, *args, **kwargs):

        if isinstance(data_file_or_loc, str):
            with open(data_file_or_loc, 'rb') as fh:
                data = pickle.load(fh)
                convert_dict_to_torch(data)
        else:
            data = data_file_or_loc

        data_params = {
            'raster_grid_size': data['raster_info']['raster'].shape[0],
            'local_grid_size': data['image_feature'].shape[0],
            'linear_features': data['linear_features'].numel(),
            'num_classes': data['classification'].numel()
        }

        net = cls(data_params, *args, **kwargs)
        if load_model:
            try:
                net.load_state_dict(torch.load(load_model))
            except FileNotFoundError:
                print('Could not find "{}", loading empty model'.format(load_model))

        return net.double()



    def __init__(self, data_params, raster_vec_size=6, local_vec_size=6):
        super(NewCloudClassifier, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)

        # Convert raster to a 6-vec
        self.raster_conv_1 = nn.Conv2d(1, 2, 5, padding=2)
        self.raster_conv_2 = nn.Conv2d(2, 4, 5, padding=2)
        self.raster_conv_3 = nn.Conv2d(4, 8, 5, padding=2)
        self.flat_raster_size = (data_params['raster_grid_size'] // 16) ** 2 * 8
        self.raster_fc = nn.Linear(self.flat_raster_size + 2, 64)
        self.raster_vec = nn.Linear(64, raster_vec_size)

        # Convert local info to a 6-vec
        self.local_conv_1 = nn.Conv2d(1, 2, 3, padding=1)
        self.local_conv_2 = nn.Conv2d(2, 4, 3, padding=1)
        self.local_conv_3 = nn.Conv2d(4, 8, 3, padding=1)
        self.flat_local_size = (data_params['local_grid_size'] // 4) ** 2 * 8
        self.local_fc = nn.Linear(self.flat_local_size, 64)
        self.local_vec = nn.Linear(64, local_vec_size)

        # Combine both along with linear info
        self.combination_fc = nn.Linear(raster_vec_size + local_vec_size + data_params['linear_features'], 64)
        self.output = nn.Linear(64, data_params['num_classes'])

    def forward(self, x):



        rast = x['raster_info']['raster']
        rast = rast.view(-1, 1, *rast.shape[-2:])
        rast = self.pool_4(nn_func.relu(self.raster_conv_1(rast)))
        rast = self.pool_4(nn_func.relu(self.raster_conv_2(rast)))
        rast = nn_func.relu(self.raster_conv_3(rast))
        flattened_rast = rast.view(-1, self.flat_raster_size)
        combined_rast = torch.cat([flattened_rast, x['raster_info']['raster_location']], 1)
        rast = nn_func.relu(self.raster_fc(combined_rast))
        rast = self.raster_vec(rast)

        local = x['image_feature']
        local = local.view(-1, 1, *local.shape[-2:])
        local = self.pool(nn_func.relu(self.local_conv_1(local)))
        local = self.pool(nn_func.relu(self.local_conv_2(local)))
        local = nn_func.relu(self.local_conv_3(local)).view(-1, self.flat_local_size)
        local = nn_func.relu(self.local_fc(local))
        local = self.local_vec(local)

        # Combine into final
        final = torch.cat([rast, local, x['linear_features']], 1)
        final = nn_func.relu(self.combination_fc(final))
        final = self.output(final)

        return final

    def guess_from_superpoint_export(self, x):
        x = deepcopy(x)

        convert_dict_to_torch(x, add_dim=True)
        self.eval()
        rez = self.forward(x)
        self.train()
        return torch.sigmoid(rez).view(-1).detach().numpy()



class CloudClassifier(nn.Module):

    @classmethod
    def from_model(cls, model = 'best_model.model'):
        net = cls()
        net.load_state_dict(torch.load(model))
        return net

    def __init__(self):
        super(CloudClassifier, self).__init__()
        self.conv_1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv_2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv_3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv_4 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_1 = nn.Linear(8 * 8 * 32, 128)
        self.fc_2 = nn.Linear(128, 3)

    def forward(self, x):

        x = nn_func.relu(self.conv_1(x))
        x = nn_func.relu(self.conv_2(x))
        # x = nn_func.relu(self.conv_2(x))
        x = self.pool(x)
        x = nn_func.relu(self.conv_3(x))
        x = nn_func.relu(self.conv_4(x))
        x = x.view(-1, 8 * 8 * 32)
        x = nn_func.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

    def guess_from_array(self, array):

        array = torch.tensor(array).float().view(-1,1,16,16)
        guesses = self.forward(array).data.numpy()
        return 1 / (1 + np.exp(-guesses[0]))


class TreeDataset(torch.utils.data.Dataset):

    def __init__(self, set_type='training', root='training_data', transform=None):

        self.transform = transform
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

        all_files = list(filter(lambda x: x.endswith('.pt'), sorted(os.listdir(root))))
        rand = np.random.uniform(size=len(all_files))
        to_select = np.where((proportions[0] <= rand) & (rand < proportions[1]))[0]
        files_to_load = [all_files[i] for i in to_select]
        self.all_data = []
        for file in files_to_load:
            with open(os.path.join(root, file), 'rb') as fh:
                data_dict = pickle.load(fh)
                convert_dict_to_torch(data_dict)
                self.all_data.append(data_dict)

        np.random.seed(None)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, i):
        data = self.all_data[i]
        if self.transform:
            self.transform(data)

        return data



def convert_dict_to_torch(data_dict, add_dim=False):
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            data_dict[key] = torch.from_numpy(val)
            if add_dim:
                data_dict[key] = data_dict[key].view(1, *data_dict[key].shape)
        elif isinstance(val, dict):
            convert_dict_to_torch(val, add_dim=add_dim)
        else:
            raise TypeError('Reached unknown nested type {}!'.format(type(val).__name__))



def train_net(max_epochs=5, no_improve_threshold=999, lr=1e-4):
    """

    :param max_epochs: Max number of epochs to train
    :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
    :return:
    """
    train_data = TreeDataset('training')
    train_loader = torch_data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

    val_data = TreeDataset('validation')
    val_loader = torch_data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)


    net = NewCloudClassifier.from_data_file(train_data[0]).double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    last_val_accuracy = 0
    best_acc = 0
    not_improved_streak = 0

    for epoch in range(1, max_epochs + 1):
        print('Starting Epoch {}...'.format(epoch))
        for data in train_loader:

            classification = torch.max(data['classification'], 1)[1]
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
            torch.save(net.state_dict(), 'best_new_model.model')
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

        classification = torch.max(data['classification'], 1)[1]
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

# def train_net(max_epochs=5, no_improve_threshold=1, lr=1e-4):
#     """
#
#     :param max_epochs: Max number of epochs to train
#     :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
#     :return:
#     """
#
#     net = CloudClassifier()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#
#
#     tf = tv.transforms.Compose([
#         tv.transforms.Grayscale(),
#         tv.transforms.RandomHorizontalFlip(),
#         tv.transforms.ToTensor(),
#         # tv.transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     train_data = tv.datasets.ImageFolder(root=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data'),
#                                          transform=tf)
#     train_loader = torch_data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)
#
#     val_data = tv.datasets.ImageFolder(
#         root=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'validation_data'), transform=tf)
#     val_loader = torch_data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=1)
#
#     last_val_accuracy = 0
#     best_acc = 0
#     not_improved_streak = 0
#
#     for epoch in range(1, max_epochs + 1):
#         print('Starting Epoch {}...'.format(epoch))
#         for data in train_loader:
#
#
#             set_trace()
#             images, labels = data
#             images, labels = Variable(images), Variable(labels)
#
#             optimizer.zero_grad()
#
#
#             outputs = net(images)
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#         train_acc, _ = eval_net(net, train_loader)
#         acc, acc_by_label = eval_net(net, val_loader)
#
#         if acc > last_val_accuracy:
#             not_improved_streak = 0
#         else:
#             not_improved_streak += 1
#             print('Accuracy has not improved for {} epochs'.format(not_improved_streak))
#             if not_improved_streak >= no_improve_threshold:
#                 print('Stopping training')
#                 break
#
#         print('Epoch {} training accuracy:   {:.2f}%'.format(epoch, train_acc * 100))
#         print('Epoch {} validation accuracy: {:.2f}%'.format(epoch, acc*100))
#
#         if acc > best_acc:
#             best_acc = acc
#             torch.save(net.state_dict(), 'best_model.model')
#             print('Saved best model!')
#
#         last_val_accuracy = acc
#
#         if not (epoch % 10):
#             print('--Current best validation acc is {:.2f}%'.format(best_acc * 100))
#
#         print('Accuracy by label:')
#         for label, label_acc in acc_by_label.items():
#             print('\t{}: {:.2f}%'.format(label, label_acc * 100))

# def eval_net(net, dataloader):
#     net.eval()
#
#     total = 0
#     correct = 0
#
#     total_per = defaultdict(lambda: 0)
#     correct_per = defaultdict(lambda: 0)
#
#     first = True
#     for data in dataloader:
#
#         images, labels = data
#         images = Variable(images)
#         labels = Variable(labels)
#         outputs = net(images)
#         if first:
#             # print(outputs)
#             first = False
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels.data).sum().item()
#
#         for label, prediction in zip(labels.data, predicted):
#             label = label.item()
#             prediction = prediction.item()
#             total_per[label] += 1
#             if prediction == label:
#                 correct_per[label] += 1
#     net.train()
#
#     final = {k: correct_per.get(k, 0) / total_per[k] for k in total_per}
#
#     return correct / total, final


if __name__ == '__main__':

    train_net(500, 500)

    # import sys
    # try:
    #     lr = float(sys.argv[1])
    # except IndexError:
    #     lr = 1e-4
    # train_net(500, 500)