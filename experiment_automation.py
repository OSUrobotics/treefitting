import os
import pickle
import MainWindow
import pymesh
from utils import compute_filter
import numpy as np
from tree_model import TreeModel
from time import time
from contextlib import contextmanager


base = '/home/main/data/point_clouds/bag_{}/cloud_final.ply'

@contextmanager
def timeit(info_dict, key):
    start = time()
    try:
        yield
    finally:
        end = time()
        info_dict[key] = end - start

def get_trusted_configs():

    all_configs = set(os.listdir('config'))

    total = 0
    trusted = []

    for val in range(0, 84):
        to_load = base.format(val)
        if not os.path.exists(to_load):
            continue
        total += 1
        hash = MainWindow.PointCloudViewerGUI.get_file_hash(to_load)
        all_configs.remove(hash)
        config_path = os.path.join('config', hash, 'config.pickle')
        try:
            with open(config_path, 'rb') as fh:
                config = pickle.load(fh)
            if config.get('trusted', False):
                trusted.append(val)
        except FileNotFoundError:
            continue
    print('Out of {} configs, {} were marked as trusted'.format(total, len(trusted)))
    return trusted

def load_file_pc_and_config(file):
    if isinstance(file, int):
        file = base.format(file)

    pc = pymesh.load_mesh(file).vertices
    hash = MainWindow.PointCloudViewerGUI.get_file_hash(file)
    with open(os.path.join('config', hash, 'config.pickle'), 'rb') as fh:
        config = pickle.load(fh)
    return pc, config



def end_to_end_test(file, params=None, downsampling=50000, superpoint_radius=0.10):
    info = {}

    pc, config = load_file_pc_and_config(file)
    initial_filter = compute_filter(pc, config['bounds'], config['polygons'])
    indexes = np.where(initial_filter)[0]
    if len(indexes) > downsampling:
        to_keep = indexes[np.random.choice(len(indexes), downsampling, replace=False)]
    final_filter = np.zeros(len(pc), dtype=np.bool)
    final_filter[to_keep] = True
    info['filter'] = final_filter

    pc = pc[final_filter]
    tree = TreeModel.from_point_cloud(pc.copy(), params=params)
    del pc

    with timeit(info, '1: superpoints'):
        tree.load_superpoint_graph(radius=superpoint_radius)

    with timeit(info, '2: edge classification'):
        tree.classify_edges()

    with timeit(info, '3: skeletonization'):
        tree.skeletonize()
        tree.find_side_branches()

    info['tree'] = tree
    info['config'] = config

    return info





if __name__ == '__main__':
    rez_folder = '/home/main/data/skeletonization_results'
    training_set = [1, 5, 7, 15, 29, 31, 52, 57, 62, 68, 73, 76]

    to_run = 5
    info = end_to_end_test(to_run, None)
    file_name_base = 'skeleton_{}_{}.pickle'
    i = 1
    while True:
        file_name = file_name_base.format(to_run, i)
        file_path = os.path.join(rez_folder, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as fh:
                pickle.dump(info, fh)
            break
        i += 1
