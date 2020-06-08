import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import pickle
from copy import deepcopy

CATEGORY_TRUNK = 0
CATEGORY_SUPPORT = 1
CATEGORY_LEADER = 2
CATEGORY_SIDE_BRANCH = 3
CATEGORY_FALSE_CONNECTION = 4

COLORS = {
    4: (0.1, 0.1, 0.1, 0.4),
    0: (0.9, 0, 0.9, 1.0),
    1: (0.1, 0.9, 0.1, 1.0),
    2: (0.1, 0.1, 0.9, 1.0),
    3: (0.9, 0.1, 0.1, 1.0),
}

def generate_random_tree():

    nodes = nx.Graph()
    sample_dist = 0.15

    trunk_height = np.random.uniform(0.3, 0.5)
    left_support_length = np.random.uniform(0.4, 0.7)
    right_support_length = np.random.uniform(0.4, 0.7)


    # total_span = left_support_length + right_support_length
    # leader_locs = np.linspace(-left_support_length, right_support_length, num_leaders + 2) + np.random.uniform(-0.02, 0.02, num_leaders + 2)


    trunk = LinearGenerator(0.01, trunk_height)
    trunk_nodes = trunk.generate_uniform_samples(sample_dist, flush=True)
    nodes_as_list = [tuple(x) for x in trunk_nodes]
    nodes.add_nodes_from(nodes_as_list)
    nodes.add_edges_from(zip(nodes_as_list[:-1], nodes_as_list[1:]), category=CATEGORY_TRUNK)

    left_support = ExponentialGenerator(left_support_length, np.random.uniform(0, 0.4), flip=True, start_pt = trunk_nodes[-1])
    right_support = ExponentialGenerator(right_support_length, np.random.uniform(0, 0.4), flip=False, start_pt = trunk_nodes[-1])

    left_nodes = left_support.generate_uniform_samples(sample_dist, flush=True)
    right_nodes = right_support.generate_uniform_samples(sample_dist, flush=True)

    add_branch_nodes_to_graph(nodes, left_nodes, CATEGORY_SUPPORT)
    add_branch_nodes_to_graph(nodes, right_nodes, CATEGORY_SUPPORT)

    all_support_nodes = list(left_nodes) + list(right_nodes)[1:]
    num_leaders = np.random.randint(len(all_support_nodes) // 2, len(all_support_nodes)) - 2

    leader_indexes = np.random.choice(len(all_support_nodes), num_leaders + 2, replace=False)
    leader_locs = sorted([all_support_nodes[i] for i in leader_indexes], key=lambda x: x[0])

    all_leaders = []
    leader_heights = np.random.uniform(0.5, 1.0, num_leaders)
    leader_nodes = []
    for i, leader_loc in enumerate(leader_locs[1:-1]):
        all_leaders.append(LinearGenerator(0.001, leader_heights[i], start_pt=leader_loc))
        current_leader_nodes = all_leaders[-1].generate_uniform_samples(sample_dist, flush=True)
        leader_nodes.append(current_leader_nodes)
        add_branch_nodes_to_graph(nodes, current_leader_nodes, CATEGORY_LEADER)

    # Add horizontal branches
    branches = []
    for i, leader_node_set in enumerate(leader_nodes):
        for flip in [False, True]:
            # Generate branches on left and right
            leader_node_set = leader_node_set[1:]
            num_branches = np.random.randint(0, min(4, len(leader_node_set)))
            branches_locs = [leader_node_set[i] for i in np.random.choice(len(leader_node_set), num_branches, replace=False)]

            if flip:
                max_len = leader_locs[i + 1][0] - leader_locs[i][0]

            else:
                max_len = leader_locs[i + 2][0] - leader_locs[i + 1][0]
            max_len *= 0.4
            for branch_x_loc, branch_y_loc in branches_locs:
                branches.append(LinearGenerator(max_len, np.random.uniform(0, 0.05), flip=flip,
                                                start_pt=np.array([branch_x_loc, branch_y_loc])))
                add_branch_nodes_to_graph(nodes, branches[-1].generate_uniform_samples(sample_dist, flush=True), CATEGORY_SIDE_BRANCH)

    return nodes


def perturb_graph(graph, noise_stdev):
    all_nodes = np.array(graph.nodes)
    new_nodes = all_nodes + np.random.normal(0, noise_stdev, all_nodes.shape)
    map = {tuple(old): tuple(new) for old, new in zip(all_nodes, new_nodes)}
    new_graph = nx.Graph()
    new_graph.add_nodes_from([tuple(x) for x in new_nodes])
    for a, b, cat in graph.edges.data('category'):
        a = tuple(a)
        b = tuple(b)
        new_graph.add_edge(map[a], map[b], category=cat)

    return new_graph

def plot_graph(graph, pts=None):
    plt.clf()
    if pts is not None:
        plt.scatter(pts[:,0], pts[:,1])
    nodes = np.array(graph.nodes)
    for edge in graph.edges:
        a, b = edge
        data = graph.edges[edge]
        cat = data['category']
        correct = data.get('correct', True)
        if not correct:
            plt.plot([a[0], b[0]], [a[1], b[1]], color='yellow', linewidth=6)
        plt.plot([a[0], b[0]], [a[1], b[1]], color=COLORS[cat], linewidth=2 if correct else 4)


    plt.scatter(nodes[:,0], nodes[:,1], marker='x')
    plt.axis('equal')
    plt.show()


def add_branch_nodes_to_graph(graph, pts, branch_type, lenience=0.01):
    pts = deepcopy(pts)
    if tuple(pts[0]) not in graph.nodes:
        existing_pts = np.array(graph.nodes)
        dists = np.linalg.norm(existing_pts - np.array(pts[0]), axis=1)
        if (dists > lenience).all():
            raise Exception('No node found!')
        to_connect = tuple(existing_pts[np.argmin(dists)])
        pts[0] = to_connect
    pts = [tuple(x) for x in pts]
    graph.add_nodes_from(pts)
    for start, end in zip(pts[:-1], pts[1:]):
        graph.add_edge(start, end, category=branch_type)

def sample_points_from_graph(graph, num=10000, noise=0.005, extreme_noise=0.03, extreme_noise_p = 0.05):
    edges = np.array(graph.edges)
    dists = np.linalg.norm(edges[:,1]-edges[:,0], axis=1)
    dists /= sum(dists)

    edge_choices = np.random.choice(len(edges), num, replace=True, p=dists)
    chosen_edges = edges[edge_choices]
    sampled_pts = chosen_edges[:,0] + ((chosen_edges[:,1] - chosen_edges[:,0]).T * np.random.uniform(size=num)).T

    extreme_noise_pts = np.random.normal(0, extreme_noise, size=sampled_pts.shape)
    extreme_noise_pts[np.random.uniform(size=num) > extreme_noise_p] = 0
    sampled_pts += np.random.normal(0, noise, size=sampled_pts.shape) + extreme_noise_pts

    return sampled_pts

def reconnect_graph(graph, points, connection_dist):
    all_nodes = list(np.array(graph.nodes))
    points = points.copy()
    for node in all_nodes:
        dists = np.linalg.norm(points - node, axis=1)
        points = points[dists > connection_dist]

    while points.size:
        new_node_idx = np.random.choice(len(points))
        new_node = points[new_node_idx]
        dists = np.linalg.norm(points - new_node, axis=1)
        points = points[dists > connection_dist]
        all_nodes.append(new_node)
        graph.add_node(tuple(new_node))

    all_nodes = np.array(all_nodes)
    for node in all_nodes:
        dists = np.linalg.norm(all_nodes - node, axis=1)
        to_connect = all_nodes[dists <= connection_dist]
        node = tuple(node)
        for connect_node in to_connect:
            connect_node = tuple(connect_node)
            if (node, connect_node) not in graph.edges:
                graph.add_edge(node, connect_node, category=CATEGORY_FALSE_CONNECTION)





# TODO: Include density component
class TreeComponent:
    def __init__(self, span, height, flip=False, start_pt=None, width=0.01, density=1.0):
        if start_pt is None:
            start_pt = np.array([0, 0])

        self.start_pt = start_pt
        self.span = span
        self.height = height
        self.flip = flip
        self.arclen_parametrization = None
        self.width = width
        self.density = density

    def generating_func(self, x_val):
        raise NotImplementedError

    def get_arclen_parametrization(self, interval=0.0001):
        if self.arclen_parametrization is not None:
            return self.arclen_parametrization
        x_vals = np.arange(0, self.span, interval)
        y_vals = []
        for x_val in x_vals:
            y_vals.append(self.generating_func(x_val))

        y_vals = np.array(y_vals)
        dists = np.sqrt((y_vals - np.roll(y_vals, 1))**2 + interval**2)
        dists[0] = 0
        cum_dists = np.cumsum(dists)

        self.arclen_parametrization = {
            'x': x_vals,
            'dists': cum_dists
        }

        return self.arclen_parametrization

    def get_point_sample_by_dist(self, arc_dist):

        parametrization = self.get_arclen_parametrization()
        x_vals = parametrization['x']
        dists = parametrization['dists']
        if arc_dist > dists[-1]:
            raise ValueError

        ind_end = np.argmax(arc_dist <= dists)
        ind_start = ind_end - 1

        x_val = x_vals[ind_start] + (x_vals[ind_end] - x_vals[ind_start]) * (arc_dist - dists[ind_start]) / (dists[ind_end] - dists[ind_start])
        return x_val, self.generating_func(x_val)

    def generate_uniform_samples(self, interval, flush=False):

        max_dist = np.max(self.get_arclen_parametrization()['dists'])

        samples = np.arange(0, max_dist, interval)
        if flush:
            samples = np.linspace(0, max_dist, len(samples) + 1)
        xys = []
        for sample in samples:
            xys.append(self.get_point_sample_by_dist(sample))

        xys = np.array(xys)
        if self.flip:
            xys[:,0] *= -1

        return xys + self.start_pt

    def get_len(self):
        params = self.get_arclen_parametrization(0.001)
        return np.max(params['dists'])


def convert_points_to_histogram(points, ref=None, size=16):
    if ref is None:
        ref = (np.max(points, axis=0) + np.min(points, axis=0))/2

    demeaned = points - ref
    scale = np.max(np.abs(demeaned))
    bins = np.linspace(-scale, scale, num=size+1)
    hist = np.histogram2d(demeaned[:,0], demeaned[:,1], bins=bins)[0]
    hist /= np.max(hist)
    return hist, scale, ref

def construct_superpoint_histogram(pt, all_points, rad, size=16):

    valid = np.linalg.norm(all_points - pt, axis=1) < rad
    return convert_points_to_histogram(all_points[valid], ref=pt, size=size)

GLOBAL_COUNTER = None

def export_as_dataset(graph, points, rad=0.15, global_size=128, return_dict=False):

    """
    Global:
        -Histogram
    Central Superpoint
        -Image
        -Location
        -Scale
    Connecting superpoints
        N
            -Image
            -Location
            -Scale
            -Edge classification (trunk, leader, support, side branch)

    """

    to_export = {}

    # Get global info
    global_image, global_scale, global_ref_pt = convert_points_to_histogram(points, size=global_size)

    to_export['global'] = {
        'image': global_image,
    }

    to_export['superpoints'] = {}
    to_export['edges'] = []

    node_index_dict = {}
    for i, node in enumerate(graph.nodes):
        node_index_dict[node] = i

    for node in graph.nodes:
        i = node_index_dict[node]
        image, scale, _ = construct_superpoint_histogram(node, points, rad, size=32)

        to_export['superpoints'][i] = {
            'image': image,
            'scale': scale,
            'location': (np.array(node) - global_ref_pt) / global_scale
        }

    for a, b, cat in graph.edges.data('category'):
        truth_vec = np.zeros(5)
        truth_vec[cat] = 1
        i_a = node_index_dict[a]
        i_b = node_index_dict[b]
        to_export['edges'].append([(i_a, i_b), truth_vec])
        to_export['edges'].append([(i_b, i_a), truth_vec.copy()])

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experimental_training_data')
    if not return_dict:
        global GLOBAL_COUNTER
        if GLOBAL_COUNTER is None:
            GLOBAL_COUNTER = len(os.listdir(root))
        GLOBAL_COUNTER += 1

        with open(os.path.join(root, '{:06d}.info'.format(GLOBAL_COUNTER)), 'wb') as fh:
            pickle.dump(to_export, fh)
    else:
        return to_export, {i: n for n, i in node_index_dict.items()}

class LinearGenerator(TreeComponent):

    def generating_func(self, x_val):
        return x_val * self.height / self.span

class ExponentialGenerator(TreeComponent):

    def generating_func(self, x_val):
        end_val = np.exp(self.span)
        scale = self.height / (end_val - 1)
        return scale * (np.exp(x_val) - 1)

if __name__ == '__main__':

    import torch

    from test_skeletonization_network import TreeDatasetFromExportDict, SyntheticTreeClassifier
    net = SyntheticTreeClassifier().double()
    with open('synthetic_best.model', 'rb') as fh:
        state_dict = torch.load(fh)
    net.load_state_dict(state_dict)

    num = np.random.randint(3000, 20000)
    extreme_noise_p = np.random.uniform(0, 0.10)
    noise = np.random.uniform(0, 0.004)
    graph_perturbation = np.random.uniform(0, 0.0070)
    extreme_noise_level = np.random.uniform(0, 0.05)

    graph = (perturb_graph(generate_random_tree(), 0.005))
    pts = sample_points_from_graph(graph, num, noise=noise, extreme_noise=extreme_noise_level,
                                   extreme_noise_p=extreme_noise_p)
    graph_perturbed = perturb_graph(graph, graph_perturbation)
    reconnect_graph(graph_perturbed, pts, 0.15)

    # plot_graph(graph_perturbed, pts)
    to_export, index_to_nodes = export_as_dataset(graph_perturbed, pts, 0.15, return_dict=True)
    dataset = TreeDatasetFromExportDict(to_export)
    classifications, edge_ids = net.guess_from_export_dataset(dataset)

    graph_copy = graph_perturbed.copy()
    cats = {}
    # TODO: Make sure symmetric ones are consistent


    for edge_id, classification in zip(edge_ids, classifications):
        (a_i, b_i), truth = to_export['edges'][edge_id]

        if b_i < a_i:
            a_i, b_i = b_i, a_i
        cats[index_to_nodes[a_i], index_to_nodes[b_i]] = classification.argmax()

    total = 0
    right = 0

    correctness = {}
    for edge, predicted_cat in cats.items():
        if predicted_cat == graph_perturbed.edges[edge]['category']:
            right += 1
            correctness[edge] = True
        else:
            correctness[edge] = False
        total += 1
    print('Accuracy: {:.2f}%'.format(100*right/total))

    nx.set_edge_attributes(graph_copy, cats, name='category')
    nx.set_edge_attributes(graph_copy, correctness, name='correct')
    plot_graph(graph_copy, pts)


    # for _ in range(2000):
    #
    #     num = np.random.randint(3000, 20000)
    #     extreme_noise_p = np.random.uniform(0, 0.10)
    #     noise = np.random.uniform(0, 0.004)
    #     graph_perturbation = np.random.uniform(0, 0.0070)
    #     extreme_noise_level = np.random.uniform(0, 0.05)
    #
    #     graph = (perturb_graph(generate_random_tree(), 0.005))
    #     pts = sample_points_from_graph(graph, num, noise=noise, extreme_noise=extreme_noise_level,
    #                                    extreme_noise_p=extreme_noise_p)
    #     graph_perturbed = perturb_graph(graph, graph_perturbation)
    #     reconnect_graph(graph_perturbed, pts, 0.15)
    #
    #     # plot_graph(graph_perturbed, pts)
    #     export_as_dataset(graph_perturbed, pts, 0.15)