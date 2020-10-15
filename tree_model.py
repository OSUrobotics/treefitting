import mesh
import skeletonization as skel
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from enum import Enum
import scipy.signal as signal
from scipy.spatial import KDTree
from scipy.linalg import svd
from itertools import combinations, permutations, product, chain
import sys
import os
from Cylinder import Cylinder
from exp_joint_detector import CloudClassifier, NewCloudClassifier, convert_pc_to_grid, project_point_onto_plane
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from utils import rasterize_3d_points, points_to_grid_svd, PriorityQueue, edges, expand_node_subset
from ipdb import set_trace
from itertools import product
import time
import matplotlib.pyplot as plt

class TreeModel(object):

    DEFAULT_ALGO_PARAMS = {
        'angle_power': 2.0,
        'angle_coeff': 0.5,
        'angle_min_degrees': 45.0,
        'elev_power': 1.0,
        'elev_coeff': 0.3,
        'elev_min_degrees': 45.0,
        'pop_size': 500,
        'pop_proposal_size': 1000,
        'pop_max_sampling': 3,
        'force_fixed_seed': False,
        'null_confidence': 0.3,
    }

    def __init__(self):
        self.points = None
        self.base_points = None     # Used to keep the original PC around if necessary
        self.graph = None
        self.thinned_tree = None
        self.mesh = None
        self.kd_tree = None
        self.net = None
        self.raster = None
        self.raster_bounds = None
        self.raster_info = None
        self.edges_rendered = False
        self.is_classified = False
        self.template_tree = None
        self.tree_population = None

        self.params = deepcopy(TreeModel.DEFAULT_ALGO_PARAMS)

        self.superpoint_graph = None
        self.edge_settings = None

        # For color output
        self.highlighted_points = defaultdict(list)
        self.point_beliefs = None




    @classmethod
    def from_point_cloud(cls, pc, kd_tree_pts = 100):
        new_model = cls()
        new_model.base_points = pc
        new_model.points = pc
        new_model.kd_tree = KDTree(pc, kd_tree_pts)

        return new_model

    @classmethod
    def from_file_name(cls, file_name):
        import pymesh
        pc = pymesh.load_mesh(file_name).vertices
        return cls.from_point_cloud(pc)

    def set_params(self, params):
        if set(params).difference(self.DEFAULT_ALGO_PARAMS):
            raise ValueError("You're passing in keys that don't exist!")

        new_params = deepcopy(self.DEFAULT_ALGO_PARAMS)
        new_params.update(params)
        self.params = new_params

        self.initialize_template_tree()


    def initialize_template_tree(self):

        self.template_tree = GrownTree(self.superpoint_graph, params=self.params)

    def rasterize(self, grid_size = 128, override=False):
        if self.raster is not None and self.raster.shape == (grid_size, grid_size) and not override:
            return

        pts_xy = self.points[:,0:2]
        min_xy = pts_xy.min(axis=0)
        max_xy = pts_xy.max(axis=0)
        diff = max_xy - min_xy
        center = min_xy + 0.5 * diff
        scale = np.max(diff) / 2

        rescaled_pts = (pts_xy - center) / scale

        raster = np.histogram2d(rescaled_pts[:,0], rescaled_pts[:,1], bins=np.linspace(-1, 1, num=grid_size + 1))[0]
        self.raster = raster / np.max(raster)
        self.raster_info = (center, scale)

    def get_raster_dict(self, point):
        """
        Used for outputting data for neural network consumption.
        :param point:
        :return:
        """
        if self.raster is None:
            self.rasterize()

        center, scale = self.raster_info
        rasterized_point = (point[:2] - center) / scale
        return {
            'raster': self.raster,
            'raster_location': rasterized_point
        }

    def rasterize_tree(self):
        if self.raster is not None:
            return

        self.raster, self.raster_bounds = rasterize_3d_points(self.points)

    def assign_edge_renders(self):

        if self.edges_rendered:
            return

        self.rasterize_tree()

        print('TEMP: Outputting edge diagnostic information')

        for s, e in self.superpoint_graph.edges:
            s_n = self.superpoint_graph.nodes[s]
            e_n = self.superpoint_graph.nodes[e]

            point_indexes = list(set(s_n['superpoint'].neighbor_index).union(e_n['superpoint'].neighbor_index))
            points = self.points[point_indexes]


            info = points_to_grid_svd(points, s_n['point'], e_n['point'], normalize=True, output_extra=True)
            #
            # import pickle
            # with open('/home/main/diagnostics/edge_info/{}_{}.edge'.format(s, e), 'wb') as fh:
            #     pickle.dump(info, fh)

            local_render = info['grid']
            global_render = rasterize_3d_points(points, bounds=self.raster_bounds)[0]

            self.superpoint_graph.edges[s, e]['global_image'] = np.stack([self.raster, global_render], axis=2)
            self.superpoint_graph.edges[s, e]['local_image'] = local_render

        self.edges_rendered = True


    def skeletonize(self):

        if self.template_tree is None:
            self.initialize_template_tree()



        self.tree_population = TreeManager(self.template_tree, population_size=self.params['pop_size'],
                                           proposal_size=self.params['pop_proposal_size'],
                                           max_prevalence=self.params['pop_max_sampling'],
                                           show_current_best=False)

        print('Skeletonizing tree...')
        if self.params.get('force_fixed_seed', False):
            np.random.seed(0)
        else:
            np.random.seed()

        self.tree_population.iterate_to_completion()
        self.tree_population.best_tree.remove_unattended_tips(self.tree_population.orig_tips)
        self.tree_population.best_tree.run_quick_analysis()
        self.thinned_tree = self.tree_population.best_tree
        self.thinned_tree.tip_nodes = self.tree_population.orig_tips


    def assign_edge_colors(self, replay_counter = None):

        self.classify_edges()

        default_colors = [
            (0.6, 0.55, 0.4),
            (0.95, 0.4, 0.6),
            (0.4, 0.6, 0.9),
            (0.9, 0.9, 0.0),
            (0.4, 0.7, 0.7),
        ]

        process_repair_edges = False
        if replay_counter is not None:
            current_graph = self.tree_population.history[replay_counter].current_graph

        elif self.thinned_tree is None:
            current_graph = self.superpoint_graph
        elif self.thinned_tree.repair_info is None:
            current_graph = self.thinned_tree.current_graph
        else:
            current_graph = self.thinned_tree.repair_info['tree']
            process_repair_edges = True

        all_chosen_edges = current_graph.edges

        for edge in self.superpoint_graph.edges:
            if edge in all_chosen_edges or edge[::-1] in all_chosen_edges:
                if edge not in all_chosen_edges:
                    edge = edge[::-1]

                assignment = current_graph.edges[edge].get('classification', 4)
                override_color = current_graph.edges[edge].get('override_color', None)
                if override_color:
                    color = override_color
                else:
                    color = default_colors[assignment]
            else:
                color = False
            self.superpoint_graph.edges[edge]['color'] = color

        if process_repair_edges:
            for edge in edges(self.thinned_tree.repair_info['nodes']):
                self.superpoint_graph.edges[edge]['color'] = (0.9, 0.1, 0.9)


        for node in self.superpoint_graph.nodes:

            override_color = current_graph.nodes[node].get('override_color', False)
            if override_color:
                color = override_color
            elif node == self.thinned_tree.trunk_node:
                color = (0.1, 0.1, 0.9)
            elif node in self.thinned_tree.tip_nodes:
                color = (0.1, 0.9, 0.9)
            else:
                color = (0.8, 0.6, 0.6)

            self.superpoint_graph.nodes[node]['color'] = color

    def classify_edges(self):

        if self.is_classified:
            return

        from test_skeletonization_network_2 import TreeDataset, RealTreeClassifier, torch_data
        net = RealTreeClassifier().double()
        net.load()

        self.assign_edge_renders()
        dataset = TreeDataset.from_superpoint_graph(self.superpoint_graph)
        dataloader = torch_data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        predictions = net.guess_all(dataloader)

        for edge, pred in zip(dataset.ids, predictions):
            info = {
                'classification_values': pred[:5],
                'connected_values': pred[5:],
                'classification': np.argmax(pred[:5]),
                'connected': np.argmax(pred[5:]),
            }

            negative_belief, positive_belief = info['connected_values']

            self.superpoint_graph.edges[edge]['prediction'] = info
            unlikeliness = (negative_belief - positive_belief + 1) / 2
            self.superpoint_graph.edges[edge]['unlikeliness'] = unlikeliness
            self.superpoint_graph.edges[edge]['likeliness'] = 1 - unlikeliness

            p1 = self.superpoint_graph.nodes[edge[0]]['point']
            p2 = self.superpoint_graph.nodes[edge[1]]['point']


            coef = 1 / (1 - self.params['null_confidence'])

            self.superpoint_graph.edges[edge]['normalized_likeliness'] = (1 - coef * unlikeliness) * np.linalg.norm(p2 - p1)
            self.superpoint_graph.edges[edge]['normalized_unlikeliness'] = unlikeliness * np.linalg.norm(p2 - p1)

        self.is_classified = True

    def assign_branch_radii(self):

        raise NotImplementedError('Temporarily disabled.')

        # for a, b, data in self.graph.edges(data=True):
        #     edge = (a,b)
        #     try:
        #         assoc = data['associations']
        #     except KeyError:
        #         continue
        #
        #     if len(assoc) < 4:
        #         print('Dropping cylinder with {} pts'.format(len(assoc)))
        #
        #     cyl = Cylinder()
        #     cyl.set_fit_pts(assoc[0], assoc, self.points)
        #     cyl.optimize_cyl(0.005, 0.10)
        #     radii[edge] = 0.01 if cyl.radius > 0.10 else cyl.radius
        #
        # nx.set_edge_attributes(self.graph, radii, name='radius')
        #
        # return radii

    def create_mesh(self):
        raise NotImplementedError('Temporarily unavailable.')
        self.assign_branch_radii()
        self.mesh = mesh.process_skeleton(self.graph, default_radius=0.01)
        return

    def output_mesh(self, file_name):
        self.create_mesh()
        import pymesh
        my_mesh = pymesh.form_mesh(self.mesh['v'], self.mesh['f'])
        pymesh.save_mesh(file_name, my_mesh)

    def query_neighborhood(self, radius, pt_index=None, highlight=False):
        if pt_index is None:
            rand_node = random.choice(list(self.superpoint_graph.nodes))
            superpoint = self.superpoint_graph.nodes[rand_node]['superpoint']
            candidates = superpoint.neighbor_index
            pt_index = candidates[np.random.choice(len(candidates))]
        pt = self.points[pt_index]
        all_points_index = self.kd_tree.query_ball_point(pt, radius)

        if highlight:
            self.highlighted_points[(0.8, 0.2, 0.2, 1.0)] = all_points_index

        return all_points_index, pt_index

    def highlight_superpoint(self, superpoint):
        self.highlighted_points = defaultdict(list)
        if not isinstance(superpoint, Superpoint):
            superpoint = self.superpoint_graph[superpoint]['superpoint']
        self.highlighted_points[(0.8, 0.2, 0.2, 1.0)] = superpoint.neighbor_index


    def highlight_edge(self, edge):
        self.highlighted_points = defaultdict(list)
        start = self.superpoint_graph.nodes[edge[0]]['superpoint']
        end = self.superpoint_graph.nodes[edge[1]]['superpoint']

        pt_indexes = set(start.neighbor_index).union(end.neighbor_index)
        self.highlighted_points[(0.2, 0.2, 0.8, 1.0)] = pt_indexes

        return pt_indexes


    def produce_open_cover(self, radius, min_points=8, neighbor_radius=None):
        to_assign = np.arange(0, self.points.shape[0])
        np.random.shuffle(to_assign)
        output = []
        while len(to_assign):
            idx = to_assign[0]
            candidate_neighbors, _ = self.query_neighborhood(radius, pt_index=idx)
            real_center = np.median(self.points[candidate_neighbors], axis=0)
            all_pts_idx = self.kd_tree.query_ball_point(real_center, radius)
            to_assign = np.setdiff1d(to_assign, all_pts_idx, assume_unique=True)
            if neighbor_radius is not None:
                all_pts_idx, _ = self.query_neighborhood(neighbor_radius, pt_index=idx)

            if len(all_pts_idx) >= min_points:
                output.append([real_center, all_pts_idx])

        return output

    def load_superpoint_graph(self, radius=0.10, min_points=8, max_neighbors=10, neighbor_radius=None):

        cover = self.produce_open_cover(radius, min_points, neighbor_radius=neighbor_radius)
        cover_centers = np.array([p[0] for p in cover])
        nodes = list(range(len(cover)))

        superpoint_graph = skel.construct_mutual_k_neighbors_graph(cover_centers, max_neighbors, radius * 2, node_index=nodes)
        self.superpoint_graph = superpoint_graph

        all_superpoints = {}
        for node, (ref_pt, assoc_indexes) in enumerate(cover):
            superpoint = Superpoint(ref_pt, assoc_indexes, self.points, radius, flow_axis=1, reverse_axis=True,
                                    global_ref=np.median(self.points, axis=0), raster_info=self.get_raster_dict(ref_pt))
            all_superpoints[node] = superpoint

        for node, superpoint in all_superpoints.items():
            superpoint.set_neighbor_superpoints([all_superpoints[n] for n in self.superpoint_graph[node]])

        nx.set_node_attributes(self.superpoint_graph, all_superpoints, name='superpoint')

        return superpoint_graph


    def query_ball_type(self, point_indexes, reference_point_index):
        labels = {0: 'Branch', 1: 'Joint', 2: 'Other'}
        if self.net is None:

            self.net = CloudClassifier.from_model()

        pts = self.points[point_indexes]
        ref_pt = self.points[reference_point_index]

        _, s, v = svd(pts - pts.mean(axis=0))

        im_array = convert_pc_to_grid(pts, ref_pt, v=v)
        guesses = self.net.guess_from_array(im_array)
        guesses = {labels[i]: guesses[i] for i in range(3)}
        main_guess = labels[np.argmax(guesses)]

        return main_guess, guesses, im_array, (s, v)

    def classify_points(self, radius):
        print('Generating open cover...')


        n = self.points.shape[0]
        beliefs = np.zeros((n, 3))
        counts = np.zeros(n)

        for _ in range(5):
            cover = self.produce_open_cover(radius)

            print('Classifying covers...')
            for ref_index, pt_indexes in cover.items():
                if len(pt_indexes) < 10:
                    continue
                _, guesses, _, _ = self.query_ball_type(pt_indexes, ref_index)
                counts[pt_indexes] += 1
                beliefs[pt_indexes] += np.array([guesses['Branch'], guesses['Joint'], guesses['Other']])

        counts[counts == 0] = 1

        beliefs = np.divide(beliefs.T, counts).T

        jointiness = (beliefs[:,1] - beliefs[:,0]) / 2 + 0.5
        non_otherness = 1 - beliefs[:,2]

        # # Temp
        # jointiness = (jointiness > 0.5) * 1.0

        rgba = np.zeros((n,4))
        rgba[:,0] = 1 - jointiness
        rgba[:,2] = jointiness
        rgba[:,3] = non_otherness

        self.point_beliefs = rgba

        return beliefs


    def get_pt_colors(self):

        if self.point_beliefs is not None:
            for i, row in enumerate(self.point_beliefs):
                yield row, np.array([i])
        else:
            all_pts = np.arange(0, self.points.shape[0])
            to_highlight = self.highlighted_points
            accounted_for_keys = []
            for color in sorted(to_highlight):
                keys = to_highlight[color]
                accounted_for_keys.extend(keys)
                if color is None:
                    color = (0.8, 0.2, 0.2, 1.0)
                yield color, keys
            yield (0.2, 0.8, 0.8, 1.0), np.setdiff1d(all_pts, accounted_for_keys)
        raise StopIteration


    def resample(self, cover_radius=None, neighbor_radius=None):

        from autoencoder_experiment import SkeletonAutoencoder, sample_xys_from_image

        old_spg = self.superpoint_graph
        if cover_radius is not None or neighbor_radius is not None:
            self.load_superpoint_graph(cover_radius, neighbor_radius=neighbor_radius)

        net = SkeletonAutoencoder(30, [32, 64, 128]).double()
        net.load()

        all_pts = []
        for node in self.superpoint_graph:
            superpt = self.superpoint_graph.nodes[node]['superpoint']
            im = superpt.image_array
            # output = net.from_numpy_array(im)
            output = im
            xys = sample_xys_from_image(output, len(superpt.neighbor_index)) - 15.5
            # zs = np.random.uniform(-1, 1, (len(superpt.neighbor_index), 1))
            zs = np.zeros((len(xys), 1))

            xyzs = np.hstack([xys, zs]) / 16 * superpt.image_scale
            homog = np.hstack([xyzs, np.ones((len(xyzs), 1))])

            tf = np.identity(4)
            tf[:3, :3] = superpt.svd[2].T
            tf[:3, 3] = superpt.ref_point

            world_xyzs_homog = tf @ homog.T
            world_xyzs = world_xyzs_homog[:3].T

            all_pts.append(world_xyzs)

        all_pts = np.concatenate(all_pts)
        if len(all_pts) >= len(self.points):
            all_pts = all_pts[np.random.choice(len(all_pts), len(self.points), replace=False)]


        self.superpoint_graph = old_spg
        print('New graph has {} pts'.format(len(all_pts)))
        return all_pts


class GrownTree:

    """
    INITIALIZATION FUNCTIONS
    """

    def __init__(self, base_graph, params, score_key='normalized_likeliness', cost_key='normalized_unlikeliness',
                 precomputed_info=None, debug=False):

        self.base_graph = base_graph
        self.current_graph = None
        self.set_rep = frozenset([])

        self.debug = debug
        self.score_key = score_key
        self.cost_key = cost_key

        self.repair_info = None

        # Parameters
        self.params = params

        # Information which is automatically computed about the tree
        self.trunk_node = None
        self.tip_nodes = []

        # Stuff to keep track of progress/repairs
        self.base_score = 0
        self.segment_to_fix = None
        self.edges_to_queue_removal = None
        self.debug_next = False

        if precomputed_info:
            self.current_graph = precomputed_info['current_graph'].copy()
            self.trunk_node = precomputed_info['trunk_node']
            self.tip_nodes = precomputed_info['tip_nodes'].copy()
            self.dijkstra_maps = precomputed_info['dijkstra_maps'].copy()
            self.base_score = precomputed_info['base_score']
            self.set_rep = precomputed_info['set_rep']
        else:
            # Initialize information such as the trunk nodes, tip nodes, etc.
            self.current_graph = nx.DiGraph()
            self.current_graph.add_nodes_from(self.base_graph.nodes)

            self.estimate_trunk_and_tips()
            self.dijkstra_maps = {tip: self.run_dijkstras_exhaust(tip) for tip in self.tip_nodes}

            self.update_node_eligibility(self.trunk_node)

        # Some debugging for plotting
        self.bounds = None

    def copy(self):
        precomputed_info = {
            'dijkstra_maps': self.dijkstra_maps,
            'current_graph': self.current_graph,
            'tip_nodes': self.tip_nodes,
            'trunk_node': self.trunk_node,
            'base_score': self.base_score,
            'set_rep': self.set_rep
        }

        return GrownTree(self.base_graph, cost_key=self.cost_key, score_key=self.score_key,
                         params = self.params, precomputed_info=precomputed_info, debug=self.debug)




    def plot_tree(self, file_name, title=''):

        all_pts = np.array([self.base_graph.nodes[n]['point'] for n in self.base_graph.nodes])
        if self.bounds is None:
            self.bounds = list(zip(all_pts.min(axis=0), all_pts.max(axis=0)))
        colors = ['green', 'red', 'blue']
        for edge in self.current_graph.edges:
            pt_1 = self.base_graph.nodes[edge[0]]['point']
            pt_2 = self.base_graph.nodes[edge[1]]['point']
            assignment = self.current_graph.edges[edge]['classification']
            plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]], color=colors[assignment])

        plt.xlim(*self.bounds[0])
        plt.ylim(*self.bounds[1])
        plt.title(title)

        plt.savefig(file_name)
        plt.clf()

    def estimate_trunk_and_tips(self):

        TRUNK_SEARCH_THRESHOLD = 0.50
        TIP_SCAN_THRESHOLD = 0.60
        ANGLE_THRESHOLD = np.radians(45)

        # Preprocessing - Find the maximal connected component, which will serve as our base tree

        max_nodes = list(max(nx.connected_components(self.base_graph), key=len))
        pts = np.array([self.base_graph.nodes[node]['point'] for node in max_nodes])

        # PART 1 - Find the trunk node by estimating where you think the node should be, then searching
        # within a threshold to find the minimal connected component in that radius
        # We look for the point in the middle of the X/Z dimensions, with a Y coordinate closest to the ground

        est = np.median(pts, axis=0)
        est[1] = pts[:,1].max()
        est_dist = np.linalg.norm(pts - est, axis=1)
        valid = est_dist < TRUNK_SEARCH_THRESHOLD
        valid_indices = np.where(valid)[0]
        if not len(valid_indices):
            raise Exception("Couldn't find any point within {:.2f}m of the estimate!".format(TRUNK_SEARCH_THRESHOLD))

        min_close_pt_idx = np.argmax(pts[valid][:,1])
        estimated_trunk = max_nodes[valid_indices[min_close_pt_idx]]
        self.trunk_node = estimated_trunk

        # PART 2 - Estimate tips
        # First, run minimum spanning tree to maximize the normalized likeliness
        # Then remove all edges which are sufficiently horizontal and which fall below the confidence threshold
        # Iterate through the remaining subcomponents and finding the maximally located node
        mst = nx.algorithms.minimum_spanning_tree(self.base_graph.subgraph(max_nodes), weight=self.cost_key)

        y_points = pts[:,1]
        y_min = y_points.min()
        y_max = y_points.max()

        y_start = y_min
        y_end = y_min + (y_max - y_min) * TIP_SCAN_THRESHOLD
        in_scan = (y_start <= y_points) & (y_points <= y_end)
        valid_nodes = [n for n, i in zip(max_nodes, in_scan) if i]
        subgraph = mst.subgraph(valid_nodes).copy()

        # Throw out horizontal looking branches
        cond = lambda e: self.get_edge_elevation(*e) < ANGLE_THRESHOLD or self.base_graph.edges[e]['likeliness'] < self.params['null_confidence']
        subgraph.remove_edges_from([e for e in subgraph.edges if cond(e)])
        subgraph = subgraph.edge_subgraph(subgraph.edges)

        # For each connected component in the graph, get the node with the most tip-like value
        tips = []
        for comp_nodes in nx.algorithms.components.connected_components(subgraph):
            best_node = min(comp_nodes, key=lambda x: self.base_graph.nodes[x]['point'][1])
            tips.append(best_node)

        self.tip_nodes = tips


    """
    FUNCTIONS FOR DETERMINING VIOLATIONS AND ASSESSING THE TREE QUALITY
    """

    def commit_edge(self, edge, assignment, validate=True, debug=False):
        """
        Adds edge to the current branch and also incrementally updates the tree score.
        :param edge:
        :param assignment:
        :return:
        """

        if validate:
            active = self.current_graph.edge_subgraph(self.current_graph.edges).nodes
            if edge[0] in active:
                raise ValueError("Cannot add an edge which starts in an existing part of the tree!")
            if edge[1] not in active:
                raise ValueError("Edge must end in an existing part of the tree!")

        self.current_graph.add_edge(*edge, classification=assignment)
        self.set_rep = self.set_rep.union([edge[0], edge[1], assignment])

        # Assess violations

        score_change = self.base_graph.edges[edge][self.score_key]
        score_change -= self.assess_edge_violation(*edge, assignment)
        out_edge = list(self.current_graph.out_edges(edge[1]))
        if not out_edge:
            out_class = 0
        else:
            out_edge = out_edge[0]
            out_class = self.current_graph.edges[out_edge]['classification']
            score_change -= self.assess_angular_violation(out_edge[0], out_edge[1], out_class, edge[0], assignment)
        # in_classes = [self.current_graph.edges[e]['classification'] for e in self.current_graph.in_edges(edge[1])]
        # previous_topology_violation = self.current_graph.nodes[edge[1]].get('topology_violation', 0)
        # current_topology_violation = self.assess_topology_split_violation(out_class, in_classes)
        # self.current_graph.nodes[edge[1]]['topology_violation'] = current_topology_violation
        # score_change -= (current_topology_violation - previous_topology_violation)

        self.base_score += score_change
        self.update_node_eligibility(edge[0])
        self.update_node_eligibility(edge[1])

    def assess_angular_violation(self, node, next_node, next_class, prev_node, prev_class):

        angle = self.get_node_turn_angle(prev_node, node, next_node)
        if next_class != prev_class:
            return 0

        angle_min = np.radians(self.params['angle_min_degrees'])

        if angle < angle_min:
            return 0
        return self.params['angle_coeff'] * (angle - angle_min) ** self.params['angle_power']


    def assess_edge_violation(self, node, next_node, classification):

        if classification == 0:
            return 0

        xy_angle = self.get_edge_elevation(node, next_node)

        if classification == 1:
            deviation = xy_angle
        else:
            deviation = np.pi/2 - xy_angle

        elev_min = np.radians(self.params['elev_min_degrees'])

        if deviation < elev_min:
            return 0
        return self.params['elev_coeff'] * (deviation - elev_min) ** self.params['elev_power']

    def assess_topology_split_violation(self, out_class, in_classes):
        raise NotImplementedError("Do you need this?")

        if len(in_classes) < 2:
            return 0

        is_equal = map(lambda x: x == out_class, in_classes)


        return 0


    @property
    def complete(self):
        return not self.tip_nodes

    @property
    def score(self):
        return self.base_score


    def remove_unattended_tips(self, orig_tip_nodes, max_len=None):
        cond = lambda n: self.current_graph.out_degree(n) and not self.current_graph.in_degree(n)
        candidates = {n for n in self.current_graph.nodes if cond(n)}.difference(orig_tip_nodes)

        DIAG_ORIG_SCORE = self.score
        total = 0
        for active_node in candidates:
            to_remove = []
            while self.current_graph.out_degree(active_node):

                # If we've already scooped up max_len number of edges,
                # then this branch is too long, and we don't remove any edges
                if max_len is not None and len(to_remove) >= max_len:
                    to_remove = []
                    break

                successor = list(self.current_graph.successors(active_node))[0]
                to_remove.append((active_node, successor))
                terminate = self.current_graph.in_degree(successor) > 1 or successor in orig_tip_nodes
                active_node = successor

                # If we reach a tip node before the max_len cutoff, cut off all the branches to there
                if terminate:
                    break

            self.current_graph.remove_edges_from(to_remove)
            total += len(to_remove)

        if total:
            print("Trimmed {}...".format(total))
            self.rescore()
        else:
            assert abs(DIAG_ORIG_SCORE - self.score) < 1e-5

    def rescore(self):

        # TODO: This can be sped up, or rather there's no need to rebuild the whole tree from scratch if you just
        # keep track of the edge score contributions

        self.base_score = 0

        old_graph = self.current_graph
        self.current_graph = self.current_graph.copy()
        self.current_graph.remove_edges_from(list(self.current_graph.edges))
        for e_r in nx.algorithms.dfs_edges(old_graph.reverse(copy=False), source=self.trunk_node):
            edge = (e_r[1], e_r[0])
            assignment = old_graph.edges[edge]['classification']
            self.commit_edge(edge, assignment, validate=False)

        return self.score


    def get_edge_elevation(self, a, b, xy_project=False):
        pt_a = self.base_graph.nodes[a]['point']
        pt_b = self.base_graph.nodes[b]['point']

        if xy_project:
            pt_a = pt_a[:2]
            pt_b = pt_b[:2]

        diff = np.abs(pt_a - pt_b)
        return np.arctan2(diff[1], diff[0])


    def get_node_turn_angle(self, a, b, c):
        pt_a = self.base_graph.nodes[a]['point']
        pt_b = self.base_graph.nodes[b]['point']
        pt_c = self.base_graph.nodes[c]['point']

        diff_1 = pt_b - pt_a
        diff_2 = pt_c - pt_b

        diff_1 = diff_1 / np.linalg.norm(diff_1)
        diff_2 = diff_2 / np.linalg.norm(diff_2)

        dot = diff_1.dot(diff_2)
        if dot > 1:
            dot = 1.0
        if dot < -1:
            dot = -1.0

        return np.arccos(dot)


    def run_dijkstras_exhaust(self, source_node):
        target_nodes = set(self.base_graph.nodes).difference({source_node})
        return self.run_dijkstras(source_node, target_nodes, consider_target_edges=True, halt_at_targets=False)

    def run_dijkstras(self, source_node, target_nodes, consider_target_edges=False, halt_at_targets=True):
        """
        Runs Dijkstra's algorithm to find the shortest path from a start node to a set of ending nodes.

        State is an ordered edge. Cost is normalized unlikeliness cost + curvature (angle complement * curvature penalty)

        :param source_node: Single target node.
        :param target_nodes: Single target node or list of nodes.
        :param consider_target_edges: If set to True, the set of goal nodes will be all incoming edges
        :param halt_at_targets: If True, the search process does not expand nodes in the target set, which means that
            it assumes all paths to the targets cannot pass through other targets.
        :return:
        """

        if isinstance(target_nodes, int):
            target_nodes = [target_nodes]

        if source_node in target_nodes:
            raise ValueError("Please make sure your source isn't in the target set.")

        # Construct the set of incoming edges for each of the target nodes
        if halt_at_targets:
            subgraph = self.base_graph.copy()
            subgraph.remove_nodes_from(target_nodes)
            comps = nx.algorithms.connected_components(subgraph)
            for comp in comps:
                if source_node in comp:
                    target_comp = expand_node_subset(comp, self.base_graph)
                    break
            else:
                raise ValueError("Your source node wasn't in the base graph!")
        else:
            target_comp = set(self.base_graph.nodes)

        target_nodes = target_comp.intersection(target_nodes)

        if consider_target_edges:
            goals = []
            for target_node in target_nodes:
                neighbors = target_comp.intersection(self.base_graph[target_node])
                if halt_at_targets:
                    neighbors = neighbors.difference(target_nodes)
                goals.extend([(neighbor, target_node) for neighbor in neighbors])
        else:
            goals = target_nodes

        remaining_goals = set(goals)
        goals = deepcopy(goals)

        queue = PriorityQueue(minimize=True)
        queue.add((None, source_node), 0)

        default_metadata =  {
            'path': [source_node],                      # Path is expressed as towards the tip
            'total_cost': 0,                            # Unlikeliness cost + curvature penalties
            'total_reward': 0,                          # Confidence-length of all edges in path
            'accumulated_angle_penalties': [],          # All curvature penalties, listed in order
            'total_angle_penalties': 0,                 # Sum of accumulated costs
            'total_angle_penalties_worst_one_out': 0,   # Sum of accumulated costs removing highest cost
            'total_angle_penalties_worst_two_out': 0,   # Sum of accumulated costs removing second highest cost
        }

        path_dict = {(None, source_node): default_metadata}

        while queue:

            edge, running_cost = queue.pop()
            previous, current = edge

            # Goal resolution
            if consider_target_edges and edge in remaining_goals:
                remaining_goals.remove(edge)

            elif not consider_target_edges and current in goals:
                remaining_goals.remove(current)
                # A bit of a hack to identify the edge associated with the goal node
                path_dict[current] = deepcopy(path_dict[edge])

            if not remaining_goals:
                break

            if current in target_nodes and halt_at_targets:
                continue

            # Expanding to neighbors
            for neighbor in self.base_graph[current]:
                if neighbor == previous:
                    continue

                edge_cost = self.base_graph.edges[current, neighbor][self.cost_key]
                curvature_cost = 0
                if previous is not None:
                    curvature_cost = self.assess_angular_violation(current, neighbor, -1, previous, -1)

                new_cost = running_cost + edge_cost + curvature_cost

                try:
                    existing_metadata = path_dict[(current, neighbor)]
                except KeyError:
                    existing_metadata = {'total_cost': np.inf}
                existing_cost = existing_metadata['total_cost']

                if new_cost < existing_cost:
                    past_metadata = path_dict[previous, current]

                    new_path = [neighbor] + past_metadata['path']

                    past_reward = past_metadata['total_reward']
                    new_reward = past_reward + self.base_graph.edges[current, neighbor][self.score_key]

                    angle_penalties = [curvature_cost] + past_metadata['accumulated_angle_penalties']
                    angle_penalites_sorted = sorted(angle_penalties)

                    total_penalties = sum(angle_penalties)
                    total_penalties_worst_one_out = total_penalties - sum(angle_penalites_sorted[-1:])
                    total_penalties_worst_two_out = total_penalties - sum(angle_penalites_sorted[-2:])

                    new_metadata = {
                        'path': new_path,
                        'total_cost': new_cost,
                        'total_reward': new_reward,
                        'accumulated_angle_penalties': angle_penalties,
                        'total_angle_penalties': total_penalties,
                        'total_angle_penalties_worst_one_out': total_penalties_worst_one_out,
                        'total_angle_penalties_worst_two_out': total_penalties_worst_two_out,
                    }

                    path_dict[(current, neighbor)] = new_metadata
                    queue.add((current, neighbor), new_cost)

        return path_dict

    def update_node_eligibility(self, node, inplace=True):
        """
        Says what kind of classifications a node is eligible to have given its current assignments.
        E.g. if you have a node with both an in and out trunk, nothing is eligible to be expanded out.
        :return:
        """

        in_edges = list(self.current_graph.in_edges(node))
        out_edges = list(self.current_graph.out_edges(node))

        # Special case: Trunk node
        if node == self.trunk_node:
            if in_edges:
                eligible = set()
            else:
                eligible = {0}
            if inplace:
                self.current_graph.nodes[node]['eligible'] = eligible
            return eligible

        in_assignments = [self.current_graph.edges[e]['classification'] for e in in_edges]
        out_assignment = self.current_graph.edges[out_edges[0]]['classification']

        # If you have a null edge, nothing may grow out of it
        if out_assignment is None:
            eligible = set()
        else:
            eligible = set(range(out_assignment, 2 + 1))
            # Case 0: When dealing with a trunk, if you have an in edge also assigned as trunk, nothing is eligible
            if out_assignment == 0 and 0 in in_assignments:
                eligible = set()

            # Case 1: If you have a trunk and there's something that's not a trunk coming out, then you can't have a trunk
            if out_assignment == 0 and sum(map(lambda x: x!=0, in_assignments)):
                eligible.difference_update({0})

            # Case 2: If you have 2 supports branching out, you can't have any more supports
            if (sum(map(lambda x: x==1, in_assignments)) >= 2):
                eligible.difference_update({1})

            # Case 3: If you're between two supports/leaders, can't have a support/leader coming out
            for assignment in [1, 2]:
                if (out_assignment == assignment and assignment in in_assignments):
                    eligible.difference_update({assignment})

        if inplace:
            self.current_graph.nodes[node]['eligible'] = eligible

        return eligible

    def compute_edge_label_pairing_scores(self, restrict_tips=None):
        # For the active tree, computes a dictionary of (edge, assignment) keys, where the values are the change in
        # potential

        cost_key_to_use = {
            0: 'total_angle_penalties_worst_two_out',
            1: 'total_angle_penalties_worst_one_out',
            2: 'total_angle_penalties'
        }

        existing_edges = list(self.current_graph.edges)
        if not existing_edges:
            active_nodes = {self.trunk_node}
        else:
            active_nodes = set(self.current_graph.edge_subgraph(self.current_graph.edges).nodes)

        rez = {}

        target_tips = restrict_tips if restrict_tips is not None else self.tip_nodes


        for active_node in active_nodes:

            eligible = self.current_graph.nodes[active_node].get('eligible', set())
            neighbors = set(self.base_graph[active_node]).difference(active_nodes)
            to_check = [(neighbor, active_node) for neighbor in neighbors]

            if active_node == self.trunk_node:
                out_node = None
                out_class = 0
            else:
                out_node = list(self.current_graph.out_edges(active_node))[0][1]
                out_class = self.current_graph.edges[active_node, out_node]['classification']

            for edge, proposed_assignment in product(to_check, eligible):
                cost_key = cost_key_to_use[proposed_assignment]
                total_score = 0

                for tip_node in target_tips:
                    dijkstra_dict = self.dijkstra_maps[tip_node]

                    # Check to see if the dijkstra map indicates the target tip node is reachable
                    try:
                        metadata = dijkstra_dict[edge]
                    except KeyError:
                        continue

                    # Check to see if the precomputed path intersects with the currently active tree
                    # Before, this is where we would recompute the Dijkstra map
                    # However to save time, we will assume that the tip is no longer reachable
                    path = metadata['path']
                    if active_nodes.intersection(path[1:]):
                        continue

                    # For the given edge-pair combo, get the path reward
                    # Subtract out the cost for the given cost key
                    # Then subtract out any penalties associated with adding the edge in, including both angular and
                    # edge wrongness
                    score = metadata['total_reward'] - metadata[cost_key]
                    score -= self.assess_edge_violation(*edge, proposed_assignment)
                    if out_node is not None:
                        curvature_penalty = self.assess_angular_violation(active_node, out_node, out_class, edge[0], proposed_assignment)
                        score -= curvature_penalty

                    # If a potential best path to a node is bad enough, we simply don't consider it
                    score = max(score, 0)

                    # If we're we assigning a leader, then we cannot extend to multiple tips - may only choose the best
                    if proposed_assignment == 2 and score > total_score:
                        total_score = score
                    else:
                        total_score += max(score, 0)

                if total_score > 0:
                    rez[edge, proposed_assignment] = total_score
        if not rez:
            self.tip_nodes = []
        return rez

    def handle_repair(self, node, assignment):

        if self.repair_info is None:
            self.initialize_repair(node)
            return

        last_node = self.repair_info['nodes'][-1]
        first_node = self.repair_info['nodes'][0]

        is_reverse_edge = (node, last_node) in self.current_graph.edges

        if node == first_node or (len(self.repair_info['nodes']) == 1 and is_reverse_edge):
            # If you reverse-click on an edge, it'll remove that particular edge and anything above it
            if is_reverse_edge:
                self.current_graph.remove_edge(node, last_node)
            else:
                # Double-clicking on a node means to toss out anything above the current node
                for edge in list(self.current_graph.in_edges(node)):
                    self.current_graph.remove_edge(*edge)
            main_comp = {self.trunk_node}
            for comp in nx.weakly_connected_components(self.current_graph):
                if self.trunk_node in comp:
                    main_comp = comp
                    break

            subgraph = self.current_graph.subgraph(main_comp)
            for edge in list(self.current_graph.edges):
                if edge not in subgraph.edges:
                    self.current_graph.remove_edge(*edge)

            self.repair_info = None
        elif node in self.repair_info['nodes']:
            print("Attempting to self-loop a repair!")
        elif node in self.repair_info['all_components'] and not node in self.repair_info['main_component']:
            print("You cannot select this node, as this would cause a loop in the tree!")
        else:
            # Clicked on a node that isn't in part of any component
            # Add the node and assignment and move along
            self.repair_info['nodes'].append(node)
            self.repair_info['assignments'].append(assignment)

            last_edge = self.repair_info['nodes'][-2:]
            if last_edge not in self.base_graph.edges:
                print('Added new edge ({}, {}) to graph'.format(*last_edge))
                self.base_graph.add_edge(*last_edge)

            if len(self.repair_info['nodes']) >= 3:
                last_last_edge = self.repair_info['nodes'][-3:-1]
                last_last_assignment = self.repair_info['assignments'][-2]
            else:
                last_last_edge = None
                last_last_assignment = None

            self.print_edge_info(last_edge, assignment, last_last_edge, last_last_assignment)

            if not node in self.repair_info['main_component']:
                return

            # At this point, the node is in the main component
            # Commit all edges and assignments to the temporary tree, then replace the current tree and recompute
            for edge, assignment in zip(edges(self.repair_info['nodes']), self.repair_info['assignments']):
                self.repair_info['tree'].add_edge(*edge, classification=assignment)

            self.current_graph = self.repair_info['tree']
            self.repair_info = None


    def initialize_repair(self, node):
        self.repair_info = {}
        self.repair_info['nodes'] = [node]
        self.repair_info['assignments'] = []

        # Create a temporary copy of the tree with the downhill segment removed
        tree_to_assign = self.current_graph.copy()
        edges_to_remove = []
        current_node = node

        while True:
            try:
                next_node = list(self.current_graph.out_edges(current_node))[0][1]
            except IndexError:
                break
            edges_to_remove.append((current_node, next_node))
            if self.current_graph.in_degree(next_node) > 1 or next_node == self.trunk_node:
                break
            current_node = next_node
        tree_to_assign.remove_edges_from(edges_to_remove)

        # Figure out which segment is the main section, and record which nodes are part of some connected component
        main_comp = {self.trunk_node}
        all_comp_nodes = set()
        for comp in nx.weakly_connected_components(tree_to_assign):
            if self.trunk_node in comp:
                main_comp = comp
            if len(comp) > 1:
                all_comp_nodes.update(comp)

        self.repair_info['main_component'] = main_comp
        self.repair_info['all_components'] = all_comp_nodes
        self.repair_info['tree'] = tree_to_assign

    def print_edge_info(self, edge, assignment, prev_edge=None, prev_assignment=None):
        print('Edge ({}, {}):'.format(*edge))
        p1 = self.base_graph.nodes[edge[0]]['point']
        p2 = self.base_graph.nodes[edge[1]]['point']
        length = np.linalg.norm(p1 - p2)
        conf = self.base_graph.edges[edge].get('likeliness', -99.99)
        score = self.base_graph.edges[edge].get(self.score_key, -99.99)
        edge_violation = self.assess_edge_violation(*edge, assignment)

        print('\tLength is: {:.4f}'.format(length))
        print('\tConfidence is: {:.4f}'.format(conf))
        print('\tScore contrib is: {:.4f}'.format(score))

        curvature_penalty = 0
        if prev_edge is not None:
            angle = self.get_node_turn_angle(prev_edge[0], edge[0], edge[1])
            print('\tAngle with previous node: {:.4f}'.format(np.degrees(angle)))
            curvature_penalty = self.assess_angular_violation(edge[0], edge[1], assignment, prev_edge[0], prev_assignment)
            print('\tIncurred curvature penalty of: {:.4f}'.format(curvature_penalty))

        print('\tIncurred elevation penalty of: {:.4f}'.format(edge_violation))

        total = score - edge_violation - curvature_penalty
        print('\tTOTAL SCORE CONTRIB: {:.4f}'.format(total))
        return total




    def run_quick_analysis(self):
        all_tips = sorted([n for n in self.current_graph.nodes if self.current_graph.out_degree(n) == 1 and not self.current_graph.in_degree(n)],
                          key=lambda x: self.base_graph.nodes[x]['point'][0])
        for tip in all_tips:
            print('For Tip {} located at ({:.2f}, {:.2f}, {:.2f}):'.format(tip, *self.base_graph.nodes[tip]['point']))
            path = [tip]
            # Build up the list of nodes associated with the tip
            while True:
                next_node = list(self.current_graph.successors(path[-1]))[0]
                if self.current_graph.edges[path[-1], next_node]['classification'] != 2:
                    break
                path.append(next_node)

            total_len = 0
            total_reward = 0

            for edge in edges(path):
                start = self.base_graph.nodes[edge[0]]['point']
                end = self.base_graph.nodes[edge[1]]['point']
                length = np.linalg.norm(start-end)
                reward = self.base_graph.edges[edge][self.score_key]

                total_len += length
                total_reward += reward

            print('\tTotal length was: {:.2f}m'.format(total_len))
            print('\tTotal reward was: {:.2f}'.format(total_reward))


    @staticmethod
    def extract_path_from_dijkstras_dict(results_dict, start, start_is_edge_pair=False, target_to_source=False):

        if not start_is_edge_pair:
            start = results_dict[start]['previous_edge']

        path = [start[1], start[0]]
        try:
            metadata = results_dict[start]
        except KeyError:
            return {
                'path': None,
                'reward': -np.inf,
                'cost': np.inf,
            }

        current_edge = metadata['previous_edge']
        while current_edge[0] is not None:
            path.append(current_edge[0])
            current_edge = results_dict[current_edge]['previous_edge']

        # At this point, the path travels from target to source in a backwards fashion. Flip if necessary
        if not target_to_source:
            path = path[::-1]

        return {
            'path': path,
            'reward': metadata['reward'],
            'cost': metadata['running_cost']
        }

    def find_side_branches(self, angle_threshold=np.pi/4, len_threshold=0.00):

        graph_copy = self.base_graph.copy()
        graph_copy.remove_edges_from([e for e in graph_copy.edges if graph_copy.edges[e]['likeliness'] < self.params['null_confidence']])

        leader_edges = [e for e in self.current_graph.edges if self.current_graph.edges[e]['classification'] == 2]
        leader_nodes = set().union(*leader_edges)
        all_edges = [e for e in self.current_graph.edges if self.current_graph.edges[e]['classification'] != 3]

        # Remove the existing side branches from the tree
        existing_side_branches = set(self.current_graph.edges).difference(all_edges)
        self.current_graph.remove_edges_from(existing_side_branches)

        base_tree_nodes = set(self.current_graph.edge_subgraph(all_edges).nodes)
        non_leader_nodes = base_tree_nodes.difference(leader_nodes)

        to_analyze = []

        for comp in nx.algorithms.connected_components(graph_copy.subgraph(set(graph_copy.nodes).difference(base_tree_nodes))):
            # Expand connected component by one element each
            expanded = set().union(*[graph_copy[n] for n in comp])
            diff = expanded.difference(comp)

            if diff.intersection(leader_nodes) and not diff.intersection(non_leader_nodes):
                to_analyze.append(expanded)



        paths_to_add = []
        for comp in to_analyze:

            # Run Dijkstras where the goal termination condition is no nodes available to expand to (also cannot turn more than 90 degrees)

            sub_leader_nodes = comp.intersection(leader_nodes)
            non_assigned_nodes = comp.difference(sub_leader_nodes)

            queue = PriorityQueue()

            # Initialize starting states by making sure they are almost perpendicular to the connecting leader
            for leader_node in sub_leader_nodes:

                targets = set(graph_copy[leader_node]).intersection(non_assigned_nodes)
                leader_node_preds = list(self.current_graph.predecessors(leader_node))
                leader_node_successor = list(self.current_graph.successors(leader_node))[0]

                if self.current_graph.edges[leader_node, leader_node_successor]['classification'] != 2:
                    continue

                if len(leader_node_preds) != 1:
                    start = leader_node
                else:
                    start = leader_node_preds[0]

                p1 = self.base_graph.nodes[start]['point']
                p2 = self.base_graph.nodes[leader_node_successor]['point']
                leader_vector = p1 - p2
                leader_vector = leader_vector / np.linalg.norm(leader_vector)
                for target in targets:
                    target_pt = self.base_graph.nodes[target]['point']
                    target_vector = target_pt - self.base_graph.nodes[leader_node]['point']
                    target_vector = target_vector / np.linalg.norm(target_vector)

                    dp = leader_vector.dot(target_vector)
                    if dp > 1:
                        dp = 1
                    if dp < -1:
                        dp = -1
                    angle = np.arccos(dp)
                    if np.abs(angle - np.pi/2) > angle_threshold:
                        continue
                    queue.add([leader_node, target], self.base_graph.edges[leader_node, target][self.cost_key])

            while queue:
                current_path, cost = queue.pop()
                current_node = current_path[-1]
                new_targets = (set(graph_copy[current_node]).difference(current_path)).intersection(non_assigned_nodes)
                if not new_targets:
                    if self.path_len(current_path) < len_threshold:
                        continue
                    paths_to_add.append(current_path[::-1])
                    break

                for target in new_targets:
                    angle = self.get_node_turn_angle(target, current_node, current_path[-2])
                    if angle > np.pi/2:
                        continue
                    penalty = self.assess_angular_violation(current_node, current_path[-2], None, target, None)
                    new_cost = cost + self.base_graph.edges[target, current_node][self.score_key] + penalty
                    new_path = current_path + [target]
                    queue.add(new_path, new_cost)


        for path in paths_to_add:
            for e in edges(path):
                self.current_graph.add_edge(*e, classification=3)

























            queue = PriorityQueue()







    def path_len(self, path):
        d = 0
        for e in edges(path):
            p1 = self.base_graph.nodes[e[0]]['point']
            p2 = self.base_graph.nodes[e[1]]['point']
            d = d + np.linalg.norm(p1-p2)
        return d


class TreeManager:

    def __init__(self, template_tree, population_size=10, proposal_size=None,
                 selection_decay=1.0, best_repopulate=0,
                 max_prevalence=3, aggregate_tips=False, show_current_best=False):


        # The half lengths says: A tree with total reward (i.e. normalized length) this much less than another tree
        # should be half as likely to be picked during the resampling phase
        self.selection_decay = selection_decay
        self.show_current_best = show_current_best
        self.best_repopulate = best_repopulate
        self.proposal_size = proposal_size if proposal_size is None else population_size
        self.max_prevalence = max_prevalence
        self.aggregate_tips = aggregate_tips

        self.population = []
        for _ in range(population_size):
            self.population.append(template_tree.copy())

        self.orig_tips = template_tree.tip_nodes

        self.last_scores = np.zeros(population_size)
        self.is_first = True
        self.iteration = 0

        self.global_best_score = 0
        self.global_best_tree = self.population[0]
        self.global_best_tree = self.population[0]

        self.history = []

    def iterate_to_completion(self):
        start = time.time()

        while not self.complete:
            self.iterate_once()

        end = time.time()
        return end - start

    def iterate_once(self):
        self.iteration += 1
        start = time.time()

        self.grow_all()
        self.score_all()

        # print('\tPlotting diagnostics...')
        #
        # ranked = np.argsort(self.last_scores)[::-1]
        #
        # for rank, pop_index in enumerate(ranked):
        #     rank = rank + 1
        #     tree = self.population[pop_index]
        #     resampled_count = (self.next_resample == pop_index).sum()
        #     file_name = 'i_{:03d}_rank_{:02d}.png'.format(self.iteration, rank)
        #     title = 'Iteration {}, rank {} (Score {:.2f}, resampled {} times)'.format(self.iteration, rank, self.last_scores[pop_index], resampled_count)
        #     tree.plot_tree(os.path.join('/home/main/diagnostics', file_name), title)


        end = time.time()

        current_max_idx = np.argmax(self.last_scores)
        current_max = self.last_scores[current_max_idx]
        if current_max > self.global_best_score:
            print('Best tree changed from having a score of {:.2f} to {:.2f}'.format(self.global_best_score, current_max))
            self.global_best_score = current_max
            self.global_best_tree = self.population[current_max_idx]
        else:
            print('Score of best tree did not change. (Best was {:.2f} but current max is {:.2f})'.format(self.global_best_score, current_max))

        if not self.iteration % 10:
            print('Removing stray tips...')
            for tree in self.population:
                tree.remove_unattended_tips(self.orig_tips, max_len=1)

        self.history.append(self.best_tree)

        print('-'*40)
        print('Iteration {} took {:.1f}s'.format(self.iteration, end-start))
        print('Best-scoring tree is now {:.2f}'.format(self.last_scores.max()))

    @property
    def best_tree(self):
        if self.show_current_best:
            return self.population[np.argmax(self.last_scores)]
        else:
            return self.global_best_tree

    @property
    def complete(self):
        return all((tree.complete for tree in self.population))


    def grow_all(self):

        PROPOSALS = defaultdict(lambda: 0)
        BASES = {}

        tree_ranks = pd.Series(self.last_scores).rank(pct=True).values

        for (i, tree), tree_rank in zip(enumerate(self.population), tree_ranks):

            if tree.complete:
                continue

            target = None
            if not self.aggregate_tips:
                target = [random.choice(tree.tip_nodes)]

            tree_scores = tree.compute_edge_label_pairing_scores(target)
            keys = list(tree_scores.keys())
            vals = np.array([tree_scores[k] for k in keys])
            proposal_ranks = pd.Series(vals).rank(pct=True).values

            for (edge, assignment), proposal_rank in zip(keys, proposal_ranks):
                set_rep = tree.set_rep.union([(edge[0], edge[1], assignment)])
                if set_rep not in BASES:
                    BASES[set_rep] = (i, edge, assignment)
                PROPOSALS[set_rep] += proposal_rank * tree_rank

        # Temp diagnostics
        print('{} unique proposals from which to pick {}'.format(len(PROPOSALS), self.proposal_size))

        if not PROPOSALS:
            print('Growth has terminated!')
            return

        # Max prevalence
        choose_len = len(PROPOSALS)
        all_trees = list(PROPOSALS.keys())
        weights = np.array([PROPOSALS[key] for key in all_trees])
        weights = weights / weights.sum()

        # If the desired population size is more than prevalence * the number of possible entries, pad the choices
        chosen = []
        remaining = self.proposal_size
        to_be_randomly_chosen = self.max_prevalence * choose_len

        if to_be_randomly_chosen < self.proposal_size:
            padding = np.arange(choose_len)
            required_padding = np.ceil((self.proposal_size - to_be_randomly_chosen) / choose_len).astype(np.int)
            for _ in range(required_padding):
                chosen.extend(padding)
            remaining -= required_padding * choose_len

        sample_count = remaining // self.max_prevalence
        to_sample = [sample_count] * self.max_prevalence
        remainder = remaining % sample_count
        for i in range(remainder):
            to_sample[i] += 1

        for sample_count in to_sample:
            chosen.extend(np.random.choice(choose_len, sample_count, replace=False, p=weights))

        new_population = []

        for picked_index in chosen:
            tree_set = all_trees[picked_index]
            base_tree_index, edge, assignment = BASES[tree_set]
            new_tree = self.population[base_tree_index].copy()
            new_tree.commit_edge(edge, assignment, validate=False)
            new_population.append(new_tree)

        if self.proposal_size > len(self.population):
            scores = np.array([tree.score for tree in new_population])

            # Selection based on ranking
            weights = pd.Series(scores).rank().values
            weights = weights / weights.sum()

            to_select = np.random.choice(len(new_population), len(self.population), replace=False, p=weights)
            new_population = [new_population[i] for i in to_select]

        
        self.population = new_population

        NEW_TREES_2 = defaultdict(lambda: 0)
        for tree in self.population:
            NEW_TREES_2[tree.set_rep] += 1
        counts = sorted(list(NEW_TREES_2.values()))
        print('Final population had {} unique out of {} total pop'.format(len(NEW_TREES_2), len(self.population)))
        print('Final tree counts: {}'.format(counts))



    def score_all(self):
        self.last_scores = np.array([tree.score for tree in self.population])



class Superpoint:

    CLASSIFIER_NET = NewCloudClassifier.from_data_file('training_data/000000.pt', load_model='best_new_model.model')

    CLASSIFICATIONS = {
        0: 'Leader',
        1: 'Side Branch',
        2: 'Endpoint',
        3: 'Joint (Leader to Side)',
        4: 'Joint (Support to Leader)',
        5: 'Support',
        6: 'Trunk',
        7: 'Non-Attached Component',
        8: 'Noise',
    }

    ATTRIBS_TO_EXPORT = sorted([
        'primary_axis',
        'ref_point',
        'neighbors',
        'primary_axis_variance',
        'global_ref'
    ])

    IMAGE_FEATURE = 'neighbor_image_array'

    def __init__(self, ref_pt, neighbor_index, pc, radius, flow_axis=0, reverse_axis=False, global_ref=None, raster_info=None):
        self.ref_point = ref_pt
        self.neighbor_index = neighbor_index

        self.points = pc[neighbor_index]
        self.pc = pc
        self.flow_axis = flow_axis      # What is the positive direction? Used for defining leading/terminal points
        self.reverse_axis = reverse_axis
        self.radius = radius
        self.neighbor_superpoints = None
        self.global_ref = global_ref
        self.raster_info = raster_info

        # Computed stats
        self._svd = None
        self._image_array = None
        self._image_scale = None
        self._main_classification = None
        self._classification = None
        self._center = None
        self._leading_point = None
        self._terminal_point = None
        self._tf = None

        # Computed stats for neighbor superpoints
        self._primary_axis_variance = None
        self._neighbor_image_array = None


    def __repr__(self):
        template = '<Superpoint at ({:.2f}, {:.2f}, {:.2f}): {} points, axis [{:.2f}, {:.2f}, {:.2f}] ({:.2f} rad)>'
        cent = self.center
        ax = self.primary_axis
        return template.format(cent[0], cent[1], cent[2], len(self.neighbor_index), ax[0], ax[1], ax[2], self.flow_angle)

    def __lt__(self, other):
        if isinstance(other, Superpoint):
            if other.flow_axis != self.flow_axis or other.reverse_axis != self.reverse_axis:
                raise ValueError('Cannot order two superpoints with inconsistent axes!')
            other = other.center[other.flow_axis]
        if not self.reverse_axis:
            return self.center[self.flow_axis] < other
        else:
            return self.center[self.flow_axis] > other

    def __gt__(self, other):
        return not self.__lt__(other)

    def set_neighbor_superpoints(self, superpoints):
        self.neighbor_superpoints = superpoints

    def export(self, categorization=None):
        output = {}
        if categorization is not None:
            output['classification'] = np.zeros(len(self.CLASSIFICATIONS))
            output['classification'][categorization] = 1

        all_linear = [np.reshape(self.__getattribute__(attrib), -1) for attrib in self.ATTRIBS_TO_EXPORT]
        output['linear_features'] = np.concatenate(all_linear)
        output['image_feature'] = self.__getattribute__(self.IMAGE_FEATURE)

        if self.raster_info is None or 'raster' not in self.raster_info or 'raster_location' not in self.raster_info:
            raise ValueError("raster_info data must contain keys 'raster' and 'raster_location'")

        output['raster_info'] = self.raster_info

        return output





    @property
    def svd(self):
        if self._svd is None:
            self._svd = svd(self.points - self.points.mean(axis=0))

        return self._svd

    @property
    def primary_axis(self):

        primary = self.svd[2][0]

        if (self.reverse_axis and primary[self.flow_axis] > 0) or (not self.reverse_axis and primary[self.flow_axis] < 0):
            return -primary

        return primary

    @property
    def tf(self):
        """
        Transforms points into the frame of the superpoint
        :return:
        """
        if self._tf is None:
            tf = np.identity(4)
            tf[:3,:3] = self.svd[2]
            tf[:3,3] = -self.svd[2].dot(self.center)
            self._tf = tf

        return self._tf

    @property
    def flow_angle(self):
        axis_vector = np.zeros(3)
        axis_vector[self.flow_axis] = -1 if self.reverse_axis else 1
        dp = np.dot(self.primary_axis, axis_vector)
        if dp > 1:
            dp = 1
        if dp < -1:
            dp = -1

        return np.arccos(dp)

    @property
    def classification(self):
        if self._classification is None:
            self.classify()
        return self._classification

    @property
    def image_array(self):
        if self._image_array is None:
            self.compute_image_array()

        return self._image_array

    @property
    def image_scale(self):
        if self._image_scale is None:
            self.compute_image_array()
        return self._image_scale

    @property
    def center(self):
        if self._center is None:
            self.compute_flow()
        return self._center

    @property
    def leading_point(self):
        if self._leading_point is None:
            self.compute_flow()
        return self._leading_point

    @property
    def terminal_point(self):
        if self._terminal_point is None:
            self.compute_flow()
        return self._terminal_point

    @property
    def neighbors(self):
        if self.neighbor_superpoints is None:
            raise ValueError("Haven't defined superpoints")

        return len(self.neighbor_superpoints)

    @property
    def primary_axis_variance(self):
        if self._primary_axis_variance is None:
            axes = []
            for superpoint in chain([self], self.neighbor_superpoints):
                axes.append(superpoint.primary_axis * (superpoint.svd[1][0] / superpoint.svd[1].sum() - 1/3))
            self._primary_axis_variance = np.array(axes).var(axis=0)

        return self._primary_axis_variance

    @property
    def neighbor_image_array(self):
        if self._neighbor_image_array is None:
            all_pts_indexes = self.neighbor_index
            for neighbor_superpoint in self.neighbor_superpoints:
                all_pts_indexes = np.union1d(all_pts_indexes, neighbor_superpoint.neighbor_index)
            all_pts = self.pc[all_pts_indexes]

            v = self.svd[2]
            im_array = convert_pc_to_grid(all_pts, self.ref_point, grid_size=32, v=v)
            self._neighbor_image_array = im_array

        return self._neighbor_image_array

    def compute_image_array(self):
        if self._image_array is None:
            v = self.svd[2]
            im_array, scale = convert_pc_to_grid(self.points, self.ref_point, grid_size=32, v=v, return_scale=True)
            self._image_array = im_array
            self._image_scale = scale

        return self._image_array

    def classify(self):

        guesses = self.CLASSIFIER_NET.guess_from_superpoint_export(self.export(None))

        guesses_dict = {i: guesses[i] for i in range(len(self.CLASSIFICATIONS))}
        main_guess = self.CLASSIFICATIONS[np.argmax(guesses)]

        self._main_classification = main_guess
        self._classification = guesses_dict

        return guesses_dict, main_guess

    def compute_flow(self):

        # TODO: May need more sophisticated way of doing this in the presence of noise, e.g. k-means clustering
        self._center = np.median(self.points, axis=0)

        self._leading_point = self._center - self.primary_axis * self.radius
        self._terminal_point = self._center + self.primary_axis * self.radius

    def compute_cylindrical_coords(self, other_pt):
        if isinstance(other_pt, Superpoint):
            other_pt = other_pt.center

        homog = np.ones(4)
        homog[:3] = other_pt
        x, y, z, _ = self.tf.dot(homog)
        axis_dist = abs(x)
        planar_dist = np.sqrt(y**2 + z**2)
        return axis_dist, planar_dist, np.arctan(planar_dist / axis_dist)

    # def compute_conic_distance_metric(self, other_pt, angle_cutoff=np.pi/2, angular_penalty=2):
    #     primary, angle = self.compute_conic_distance(other_pt)
    #     if angle > angle_cutoff:
    #         return np.inf
    #     true_dist = primary / np.cos(angle)
    #     multiplier = 1 + (angular_penalty - 1) * (angle / angle_cutoff)
    #     return true_dist * multiplier




class Segment:

    MASTER_LIST = {}

    @classmethod
    def get_by_id(cls, id):
        return cls.MASTER_LIST[id]

    @classmethod
    def get_by_type(cls, segment_type):
        return [cls.MASTER_LIST[key] for key in cls.MASTER_LIST if cls.MASTER_LIST[key].classification == segment_type]


    @classmethod
    def register(cls, obj):
        cls.MASTER_LIST[obj.id] = obj

    def __init__(self, segment, segment_id=None):

        if segment_id is None:
            if not len(self.MASTER_LIST):
                segment_id = 0
            else:
                segment_id = max(self.MASTER_LIST) + 1

        self.segment = segment
        self.id = segment_id
        self.register(self)

        self.endpoints = [np.array(segment[0]), np.array(segment[-1])]
        self.vector = self.endpoints[1] - self.endpoints[0]
        self.vector /= np.linalg.norm(self.vector)
        self.angle = np.abs(np.arcsin(self.vector[1]))

        cum_l = 0
        len_index = {}
        for i, (s, e) in enumerate(zip(segment[:-1], segment[1:])):
            len_index[i] = cum_l
            cum_l += np.linalg.norm(np.array(s) - np.array(e))

        self.length = cum_l
        # Find midpoint
        rel_index = max([i for i in len_index if len_index[i] <= self.length / 2])
        s = np.array(segment[rel_index])
        e = np.array(segment[rel_index + 1])
        remaining_len = self.length / 2 - len_index[rel_index]
        self.midpoint = s + (e-s) * remaining_len / np.linalg.norm(e-s)
        self.endpoint_vectors = [self.midpoint - self.endpoints[0], self.midpoint - self.endpoints[1]]


    def edges(self):
        return zip(self.segment[:-1], self.segment[1:])


def preprocess_point_cloud(pc, downsample=10000):
    # Pre-process the cloud
    # First, filter out Y-coordinate
    hist_y, hist_edges = np.histogram(pc[:, 1], bins=20, density=True)
    hist_y = -hist_y
    req_height = (hist_y.max() + hist_y.min()) / 2
    peak = max(signal.find_peaks(hist_y, height=req_height, distance=5)[0])
    corresponding_edge = hist_edges[peak + 1]

    pc = pc[pc[:, 1] < corresponding_edge]

    # Next, filter out the back row of branches
    hist_z, hist_edges = np.histogram(pc[:, 2], bins=20, density=True)
    hist_z = -hist_z
    req_height = (hist_z.max() + hist_z.min()) / 2
    peak = max(signal.find_peaks(hist_z, height=req_height, distance=5)[0])
    pc = pc[pc[:, 2] < hist_edges[peak + 1]]

    if downsample and pc.shape[0] > downsample:
        print('DOWNSAMPLING POINT CLOUD FROM {} TO {} POINTS'.format(pc.shape[0], downsample))
        pc = pc[np.random.choice(pc.shape[0], downsample, replace=False)]

    return pc


if __name__ == '__main__':

    file_path = sys.argv[1]
    try:
        output_path = sys.argv[2]
    except IndexError:
        components = list(os.path.split(file_path))
        if '.' in components[-1]:
            components[-1] = '.'.join(components[-1].split('.')[:-1]) + '.obj'
        else:
            components[-1] = components[-1] + '.obj'

        output_path = os.path.join(*components)

    print('Reading file from: {}'.format(file_path))
    print('Outputting file:   {}'.format(output_path))

    model = TreeModel.from_file_name(file_path, process=True)
    model.output_mesh(output_path)
