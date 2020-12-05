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
                if config.get('trunk') is None:
                    print('{} had no trunk!'.format(val))
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

    # HACK!
    if 'martins_clouds' in file:
        pc = pc @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])


    initial_filter = compute_filter(pc, config['bounds'], config['polygons'])
    indexes = np.where(initial_filter)[0]
    if len(indexes) > downsampling:
        to_keep = indexes[np.random.choice(len(indexes), downsampling, replace=False)]
    else:
        to_keep = indexes
    final_filter = np.zeros(len(pc), dtype=np.bool)
    final_filter[to_keep] = True
    info['filter'] = final_filter

    pc = pc[final_filter]
    tree = TreeModel.from_point_cloud(pc.copy(), params=params, trunk=config.get('trunk', None))
    del pc

    with timeit(info, '1: superpoints'):
        tree.load_superpoint_graph(radius=superpoint_radius)

    with timeit(info, '2: edge classification'):
        tree.classify_edges()

    with timeit(info, '3: skeletonization'):
        tree.skeletonize()
        tree.thinned_tree.find_side_branches()

    info['tree'] = tree
    info['config'] = config
    info['config']['source'] = file

    return info





if __name__ == '__main__':
    rez_folder = '/home/main/data/skeletonization_results'
    # training_set = [1, 5, 7, 15, 29, 31, 52, 57, 62, 68, 73, 76]
    # import random
    # random.shuffle(training_set)
    #
    # for to_run in training_set:
    #     info = end_to_end_test(to_run, None)
    #     file_name_base = 'skeleton_{}_{}.pickle'
    #     i = 1
    #     while True:
    #         file_name = file_name_base.format(to_run, i)
    #         file_path = os.path.join(rez_folder, file_name)
    #         if not os.path.exists(file_path):
    #             with open(file_path, 'wb') as fh:
    #                 pickle.dump(info, fh)
    #             break
    #         i += 1

    file_path = '/home/main/data/point_clouds/martins_clouds/07366.ply'
    info = end_to_end_test(file_path)
    to_save = os.path.join(rez_folder, 'martins_cloud.pickle')
    with open(to_save, 'wb') as fh:
        pickle.dump(info, fh)

    # from itertools import product
    # param_settings = {
    #     'angle_coeff': [0.25, 0.5, 0.75],
    #     'angle_min_degrees': [30.0, 60.0, 90.0],
    #     'angle_power': [1, 2],
    #     'elev_coeff': [0.1, 0.3, 0.5],
    #     'elev_min_degrees': [30, 50, 70],
    #     'elev_power': [1],
    #     'force_fixed_seed': [True],
    #     'pop_size': [300],
    # }
    #
    # test_tree = 15
    #
    # keys = list(param_settings.keys())
    #
    # print(keys)
    # assert False
    #
    # vals = [param_settings[k] for k in keys]
    # file_name_base = 'calibration_{}_{}.pickle'
    # for i, all_settings in enumerate(product(*vals)):
    #     params = {k: s for k, s in zip(keys, all_settings)}
    #     info = end_to_end_test(test_tree, params=params)
    #     file_name = file_name_base.format(test_tree, i)
    #     file_path = os.path.join(rez_folder, file_name)
    #     with open(file_path, 'wb') as fh:
    #         pickle.dump(info, fh)



