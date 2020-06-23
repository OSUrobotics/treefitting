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
from itertools import product
import hashlib
from ipdb import set_trace
import matplotlib


class SkeletonAutoencoder(nn.Module):

    def __init__(self, encoder_dim, encoder_layers):
        encoder_str = 'd{}_'.format(encoder_dim) + '_'.join([str(layer) for layer in encoder_layers])
        super(SkeletonAutoencoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers
        self.name = encoder_str
        self.final_side = int(32 / (2 ** len(encoder_layers)))

        self.pool = nn.MaxPool2d(2, 2)
        # self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        all_layers = [1] + encoder_layers

        convs = []
        deconvs = []
        for start_channels, end_channels in zip(all_layers[:-1], all_layers[1:]):
            convs.append([nn.Conv2d(start_channels, end_channels, 3, padding=1), self.relu, self.pool])
            deconvs.append([self.unpool, nn.Conv2d(end_channels, start_channels, 3, padding=1), self.relu])
        flatten = lambda l: [item for sl in l for item in sl]
        convs = flatten(convs)
        deconvs = flatten(deconvs[::-1])
        deconvs[-1] = self.sigmoid      # We don't want to RELU the last one

        self.convs = nn.Sequential(*convs)
        self.deconvs = nn.Sequential(*deconvs)

        total_features = self.final_side ** 2 * all_layers[-1]
        self.encode_fc = nn.Linear(total_features, encoder_dim)
        self.decode_fc = nn.Linear(encoder_dim, total_features)

    def load(self):
        try:
            with open('{}.model'.format(self.name), 'rb') as fh:
                state_dict = torch.load(fh)
            self.load_state_dict(state_dict)
            print('Loaded existing model weights!')
        except FileNotFoundError:
            print('No state dict found!')

    def encode(self, x):

        x = self.convs(x)
        x = self.encode_fc(x.view(x.shape[0], -1))
        return x

    def decode(self, x):
        # set_trace()
        x = self.decode_fc(x)
        x = x.view(x.shape[0], -1, self.final_side, self.final_side)
        x = self.deconvs(x)

        # for layer in self.deconvs:
        #     new = layer(x)
        #     # if new.max() < 1e-5:
        #     #     set_trace()
        #
        #     if (new <= 0).all():
        #         print('All negative!')
        #         set_trace()

            # x = new

        return x

    def forward(self, x):

        inputs = x['input']
        inputs = inputs.view(-1, 1, *inputs.shape[-2:])

        encoded = self.encode(inputs)
        decoded = self.decode(encoded)

        # Reshape output to be mono-channel
        return decoded.view(inputs.shape[0], *inputs.shape[2:])

    def from_numpy_array(self, x):
        self.eval()
        with torch.no_grad():

            input = {'input': torch.from_numpy(x).view(1, *x.shape)}
            return self(input).view(*x.shape).detach().numpy()


class ImageLoader(torch.utils.data.Dataset):

    DATASET_BOUNDS = {
        'training': (0.0, 0.7),
        'validation': (0.7, 0.9),
        'test': (0.9, 1.0),
    }

    @classmethod
    def from_folders(cls, training_root, truth_root, bounds='training'):

        dataset = cls()

        if isinstance(bounds, str):
            bounds = cls.DATASET_BOUNDS[bounds]

        low, high = bounds

        truths = {}
        truth_files = os.listdir(truth_root)
        for file in truth_files:
            file_root = file.split('.')[0]
            if not (low <= str_to_float(file_root) < high):
                continue

            img = load_image(os.path.join(truth_root, file), invert=False)
            truths[file_root] = img

        print('Loaded {} truth images'.format(len(truths)))

        training_files = os.listdir(training_root)
        for file in training_files:
            file_root = file.split('.')[0]
            truth_id, rot, flip, _ = file_root.split('_')
            if truth_id not in truths:
                continue

            rot = int(rot)
            flip = bool(int(flip))

            truth_img = np.ascontiguousarray(flip_and_rot_image(truths[truth_id], flip, rot))
            input_img = np.ascontiguousarray(load_image(os.path.join(training_root, file)))

            dataset.add_item(input_img, truth_img)

        return dataset

    def __init__(self, transform=None):

        super(ImageLoader, self).__init__()
        self.transform = transform

        self.inputs = []
        self.truths = {}

    def add_item(self, input_img, truth_img=None):

        if isinstance(input_img, np.ndarray):
            input_img = torch.from_numpy(input_img).double()

        if truth_img is None:
            truth_img = torch.zeros(input_img.shape).double()
        elif isinstance(truth_img, np.ndarray):
            truth_img = torch.from_numpy(truth_img).double()


        self.inputs.append({
            'input': input_img,
            'truth': truth_img
        })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):

        return self.inputs[i]

class ImageLoaderPaired(ImageLoader):
    @classmethod
    def from_folder(cls, root, bounds='training'):

        dataset = cls()

        if isinstance(bounds, str):
            bounds = cls.DATASET_BOUNDS[bounds]

        low, high = bounds

        all_files = [file for file in os.listdir(root) if file.endswith('_t.png')]
        for file in all_files:
            file_root = file.split('_t')[0]
            if not (low <= str_to_float(file_root) < high):
                continue

            input_file = os.path.join(root, file_root + '.png')
            base_truth_image = load_image(os.path.join(root, file))
            base_input_image = load_image(os.path.join(root, input_file))

            for flip, rot in product([False, True], [0, 1, 2, 3]):

                truth_img = np.ascontiguousarray(flip_and_rot_image(base_truth_image, flip, rot))
                input_img = np.ascontiguousarray(flip_and_rot_image(base_input_image, flip, rot))

                dataset.add_item(input_img, truth_img)

        return dataset



def str_to_float(x):
    hexdigest = hashlib.md5(x.encode('utf8')).hexdigest()[:8]
    np.random.seed(int(hexdigest, base=16))
    val = np.random.uniform()
    np.random.seed(None)
    return val

def train_net(max_epochs=5, stop_after=12, lr=1e-4, load=False, suffix=''):
    """

    :param max_epochs: Max number of epochs to train
    :param no_improve_threshold: If the validation error does not get better after X epochs, stop the training
    :return:
    """
    train_folder = '/home/main/data/autoencoder_test_training'
    truth_folder = '/home/main/data/autoencoder_test_truth'
    real_folder = '/home/main/data/autoencoder_real_tree'

    train_data = ImageLoader.from_folders(train_folder, truth_folder, 'training')
    train_data_real = ImageLoaderPaired.from_folder(real_folder, 'training')
    train_loader = torch_data.DataLoader(torch_data.ConcatDataset([train_data, train_data_real]), batch_size=10, shuffle=True, num_workers=0)

    val_data = ImageLoader.from_folders(train_folder, truth_folder, 'validation')
    val_data_real = ImageLoaderPaired.from_folder(real_folder, 'validation')
    val_loader = torch_data.DataLoader(torch_data.ConcatDataset([val_data, val_data_real]), batch_size=10, shuffle=False, num_workers=0)

    # test_data = ImageLoader.from_folders(train_folder, truth_folder, 'test')
    test_data_real = ImageLoaderPaired.from_folder(real_folder, 'test')
    test_loader = torch_data.DataLoader(test_data_real, batch_size=15, shuffle=True, num_workers=0)

    net = SkeletonAutoencoder(30, [32, 64, 128]).double()

    if load:
        net.load()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_loss = np.inf
    not_improved_streak = 0

    total = len(train_loader)
    epoch = 0
    output_test_diagnostics(net, test_loader, suffix='d{}_{}'.format(net.encoder_dim, 'd{}_{}{}'.format(net.encoder_dim, suffix, epoch)))
    for epoch in range(1, max_epochs + 1):
        print('Starting Epoch {}...'.format(epoch))
        for iter, data in enumerate(train_loader):

            if not iter % 100:
                print('{}/{}'.format(iter, total))

            optimizer.zero_grad()

            outputs = net(data)
            total_loss = criterion(outputs, data['truth']) * outputs.shape[0]
            total_loss.backward()

            optimizer.step()


        val_loss = eval_net(net, val_loader)
        if val_loss < best_loss:
            not_improved_streak = 0
            best_loss = val_loss
            print('Loss improved! Saving model and outputting diagnostics...')
            torch.save(net.state_dict(), '{}.model'.format(net.name))
            output_test_diagnostics(net, test_loader, suffix='d{}_{}'.format(net.encoder_dim, 'd{}_{}{}'.format(net.encoder_dim, suffix, epoch)))

        else:
            not_improved_streak += 1
            print('Loss has not improved for {} epochs'.format(not_improved_streak))
            if not_improved_streak >= stop_after:
                return

        print('\tEpoch {} validation loss:  {:.8f}'.format(epoch, val_loss))
        print('\tBest loss was at epoch {}: {:.8f}'.format(epoch - not_improved_streak, best_loss))


def eval_net(net, dataloader):
    net.eval()
    criterion = nn.MSELoss()
    total = 0
    running_loss = 0

    for data in dataloader:
        outputs = net(data)
        loss = float(criterion(outputs, data['truth'])) * outputs.shape[0]
        running_loss += loss
        total += outputs.shape[0]

    net.train()


    return running_loss / total

def output_test_diagnostics(net, loader, suffix=''):
    # Num to plot will be based on batch size of loader

    predicted = np.zeros((0, 0, 0))
    truth = np.zeros((0, 0, 0))
    inputs = np.zeros((0, 0, 0))

    for data in loader:
        predicted = net(data).detach().numpy()
        inputs = data['input'].numpy()
        truth = data['truth'].numpy()
        break

    for arr in [predicted, inputs, truth]:
        arr[arr > 1.0] = 1.0
        arr[arr < 0.0] = 0.0




    plt.clf()
    n = len(predicted)
    plt.figure(figsize=(6, 2 * n))
    for i, imgs in enumerate(zip(inputs, predicted, truth)):

        for j, img in enumerate(imgs):
            ax = plt.subplot(n, 3, i*3 + (j+1))
            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    if suffix:
        suffix = '_' + suffix

    plt.savefig('/home/main/data/autoencoder_diagnostics/results{}.png'.format(suffix))
    plt.close()
    plt.clf()


"""
DATA GENERATION
"""

def load_image(name, invert=False, as_torch=False, negative_domain=False):
    img = np.array(imageio.imread(name))
    if len(img.shape) > 2:
        img = img.mean(axis=2)

    img = img / 255

    if invert:
        img = 1 - img

    if negative_domain:
        img = 2 * img - 1

    if as_torch:
        img = torch.from_numpy(img).double()

    return img


def flip_and_rot_image(img, flip=False, rot=0):
    return np.rot90(img.copy()[:, ::-1 if flip else 1], rot)

def sample_xys_from_image(image, n, flip=False):

    weights = np.reshape(image, (-1,))
    weights = weights / weights.sum()
    choices = np.random.choice(len(weights), n, p=weights)

    ys = choices % image.shape[1]
    xs = choices // image.shape[1]
    if flip:
        xs, ys = ys, xs

    xys = np.array([xs, ys]).T.astype(np.float)
    return xys


def generate_noised_image(name, points_range=(200, 1000), noise_range=(0,3), white_noise_range=(0,0.02),
                          black_noise_range=(0, 0.1), num_images=80, output_folder=None):

    file_root = os.path.split(name)[-1].split('.')[0]
    new_name = file_root + '_{}_{}_{}.png'

    img = load_image(name, invert=False)
    for i in range(num_images):
        rot = np.random.randint(0, 4)
        flip = np.random.randint(0, 2)
        new_img = flip_and_rot_image(img, flip, rot)
        xys = sample_xys_from_image(new_img, np.random.randint(*points_range))
        gaussian_noise_level = np.random.uniform(*noise_range)
        xys += np.random.normal(0, gaussian_noise_level, size=xys.shape)
        nx, ny = new_img.shape
        hist = np.histogram2d(xys[:,0], xys[:,1], bins=[np.linspace(0, nx, nx+1), np.linspace(0, ny, ny+1)])[0]
        hist = hist / hist.max()

        white_noise_threshold = np.random.uniform(*white_noise_range)
        black_noise_threshold = np.random.uniform(*black_noise_range)

        white_noise_mask = np.random.uniform(size=new_img.shape) < white_noise_threshold

        hist[white_noise_mask] = np.random.uniform(hist.mean(), 1, size=white_noise_mask.sum())
        hist[np.random.uniform(size=new_img.shape) < black_noise_threshold] = 0

        if output_folder is None:
            plt.imshow(hist)
            plt.show()
        else:

            file_name = new_name.format(rot, flip, i)
            imageio.imwrite(os.path.join(output_folder, file_name), hist)


if __name__ == '__main__':
    root = '/home/main/data/autoencoder_test_truth'
    new = '/home/main/data/autoencoder_test_training'

    train_net(5000, 12, lr=5e-5, load=True, suffix='testing_real_data_')



    # # Data generation
    #
    #
    # files = os.listdir(root)
    # for file in files:
    #     if not file.endswith('.png'):
    #         continue
    #     full_path = os.path.join(root, file)
    #     generate_noised_image(full_path, output_folder=new)








