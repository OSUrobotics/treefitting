import mesh
import skeletonization as skel
import networkx as nx
from collections import defaultdict
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
from utils import rasterize_3d_points, points_to_grid_svd
from ipdb import set_trace

color_wheel = (np.array([
    (204, 153, 153),
    (163, 204, 0),
    (0, 109, 204),
    (255, 0, 0),
    (204, 54, 0),
    (187, 204, 153),
    (153, 173, 204),
    (204, 0, 82),
    (204, 136, 0),
    (0, 204, 190),
    (0, 27, 204),
    (255, 20, 20)
]) / 255).tolist()

class SegmentType(Enum):
    UNASSIGNED = -1
    TRUNK = 0
    SUPPORT = 1
    LEADER = 2
    TARGET = 3
    AUXILIARY = 4
    NOISE = 5
    FALSE_CONNECTION = 6

SUPPORT_LEN_THRESHOLD = 0.4

class TreeModel(object):

    def __init__(self):
        self.points = None
        self.base_points = None     # Used to keep the original PC around if necessary
        self.graph = None
        self.segment_graph = None
        self.segments = []
        self.mesh = None
        self.kd_tree = None
        self.net = None
        self.trunk_guess = None
        self.raster = None
        self.raster_bounds = None
        self.raster_info = None
        self.edges_rendered = False
        self.is_classified = False

        self.superpoint_graph = None

        # For color output
        self.highlighted_points = defaultdict(list)
        self.point_beliefs = None

        # Temporary
        self.cover = None
        self.toggle = 0

    @classmethod
    def from_point_cloud(cls, pc, process=False, kd_tree_pts = 100, bin_width=0.01):
        new_model = cls()
        new_model.base_points = pc
        if process:
            pc = preprocess_point_cloud(pc)

        new_model.points = pc
        new_model.kd_tree = KDTree(pc, kd_tree_pts)

        new_model.trunk_guess = np.array([pc[:, 0].mean(), np.max(pc[:, 1]), np.max(pc[:, 2])])
        print('Trunk guess: {}'.format(new_model.trunk_guess))

        return new_model

    @classmethod
    def from_file_name(cls, file_name, process=False):
        import pymesh
        pc = pymesh.load_mesh(file_name).vertices
        return cls.from_point_cloud(pc, process=process)

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

        for s, e in self.superpoint_graph.edges:
            s_n = self.superpoint_graph.nodes[s]
            e_n = self.superpoint_graph.nodes[e]

            point_indexes = list(set(s_n['superpoint'].neighbor_index).union(e_n['superpoint'].neighbor_index))
            points = self.points[point_indexes]
            local_render = points_to_grid_svd(points, s_n['point'], e_n['point'])
            global_render = rasterize_3d_points(points, bounds=self.raster_bounds)[0]

            self.superpoint_graph.edges[s, e]['global_image'] = np.stack([self.raster, global_render], axis=2)
            self.superpoint_graph.edges[s, e]['local_image'] = local_render

        self.edges_rendered = True


    def thin_skeleton(self, lower_threshold, upper_threshold):

        self.classify_edges()
        graph = self.superpoint_graph.copy()
        all_edges = [(e, graph.edges[e]['prediction']['connected_values'][1]) for e in graph.edges]
        all_edges.sort(key=lambda v: v[1])
        to_return = []

        # Assign bridges
        bridges = set(nx.algorithms.bridges(graph))
        graph.remove_edges_from(bridges)

        for edge, val in all_edges:
            if val < lower_threshold:
                graph.remove_edge(*edge)
                continue
            if val > upper_threshold or edge in bridges:
                to_return.append(edge)
                continue
            # Final case: We're in the middle threshold and dealing with a non-bridge edge
            # Remove it and recompute the bridges from one of the edge nodes
            graph.remove_edge(*edge)
            bridges.update(nx.algorithms.bridges(graph, root=edge[0]))

        print('Reduced edges from {} to {}'.format(len(all_edges), len(to_return)))

        return to_return


    def assign_edge_colors(self, settings_dict):

        self.classify_edges()

        to_show = self.superpoint_graph.edges
        if settings_dict['thinning'] is not None:
            to_show = self.thin_skeleton(*settings_dict['thinning'])

        for edge in self.superpoint_graph.edges:
            if edge not in to_show:
                self.superpoint_graph.edges[edge]['color'] = False
                continue

            pred = self.superpoint_graph.edges[edge]['prediction']

            connection_col = 1 if settings_dict['show_connected'] else 0
            is_visible = pred['connected_values'][connection_col] > settings_dict['connection_threshold']

            if not is_visible:
                self.superpoint_graph.edges[edge]['color'] = False
                continue

            if settings_dict['multi_classify']:
                raise NotImplementedError
            else:
                # Single - Even if a branch doesn't have a category as its primary
                # classification, show it
                val = pred['classification_values'][settings_dict['data']['category']]
                if val < settings_dict['data']['threshold']:
                    color = False
                else:
                    color = (1.0, 1.0, 1.0)

            self.superpoint_graph.edges[edge]['color'] = color

        #
        # COLORS = {
        #     0: (0.6, 0.4, 0.05),    # Brown
        #     1: (0.15, 0.6, 0.3),    # Green
        #     2: (0.5, 0.6, 0.9),     # Light Blue
        #     3: (0.9, 0.6, 0.6),     # Light Red
        #     4: (0.85, 0.2, 0.85),   # Hot Pink
        # }
        #
        # for edge in self.superpoint_graph.edges:
        #     pred = self.superpoint_graph.edges[edge]['prediction']
        #     if pred['connected']:
        #         color = COLORS[pred['classification']]
        #     else:
        #         color = False
        #     self.superpoint_graph.edges[edge]['color'] = color

    def classify_edges(self):

        if self.is_classified:
            return

        from test_skeletonization_network_2 import TreeDataset, RealTreeClassifier, torch_data
        net = RealTreeClassifier(point_dim=0).double()
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

            self.superpoint_graph.edges[edge]['prediction'] = info
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

            # real_pts = self.points[superpt.neighbor_index]
            # discrep = world_xyzs.mean(axis=0) - real_pts.mean(axis=0)
            # if np.any(np.abs(discrep) > 0.07):
            #     print('Big discrepancy!')
            #     print(discrep)
            #     all_pts = [world_xyzs, real_pts]
            #     break

            all_pts.append(world_xyzs)

        all_pts = np.concatenate(all_pts)
        if len(all_pts) >= len(self.points):
            all_pts = all_pts[np.random.choice(len(all_pts), len(self.points), replace=False)]


        self.superpoint_graph = old_spg
        print('New graph has {} pts'.format(len(all_pts)))
        return all_pts


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
        self.classification = SegmentType.UNASSIGNED

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


    # def guess_initial_classification(self):
    #     if self.angle > np.pi / 4:
    #         self.classification = SegmentType.LEADER
    #     else:
    #         self.classification = SegmentType.SUPPORT

    def edges(self):
        return zip(self.segment[:-1], self.segment[1:])


    def get_segment_color(self):
        return color_wheel[self.classification.value]


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
