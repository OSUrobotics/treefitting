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
from utils import rasterize_3d_points, points_to_grid_svd, PriorityQueue, edges
from ipdb import set_trace

class TreeModel(object):

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


        self.superpoint_graph = None
        self.edge_settings = None

        # For color output
        self.highlighted_points = defaultdict(list)
        self.point_beliefs = None


    @classmethod
    def from_point_cloud(cls, pc, process=False, kd_tree_pts = 100, bin_width=0.01):
        new_model = cls()
        new_model.base_points = pc
        if process:
            pc = preprocess_point_cloud(pc)

        new_model.points = pc
        new_model.kd_tree = KDTree(pc, kd_tree_pts)

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
            local_render = points_to_grid_svd(points, s_n['point'], e_n['point'], normalize=True)
            global_render = rasterize_3d_points(points, bounds=self.raster_bounds)[0]

            self.superpoint_graph.edges[s, e]['global_image'] = np.stack([self.raster, global_render], axis=2)
            self.superpoint_graph.edges[s, e]['local_image'] = local_render

        self.edges_rendered = True





    def assign_edge_colors(self, settings_dict=None, iterate=True):

        if settings_dict is None:
            settings_dict = self.edge_settings
        else:
            self.edge_settings = settings_dict

        self.classify_edges()

        if settings_dict['correction']:

            if self.thinned_tree is None:
                self.initialize_final_tree()
            elif iterate:
                self.thinned_tree.iterate()

            if self.thinned_tree.repair_info is None:
                current_graph = self.thinned_tree.current_graph
            else:
                current_graph = self.thinned_tree.repair_info['tree']

            all_chosen_edges = current_graph.edges
            for edge in self.superpoint_graph.edges:

                if edge in all_chosen_edges or edge[::-1] in all_chosen_edges:
                    if edge not in all_chosen_edges:
                        edge = edge[::-1]

                    if settings_dict['multi_classify']:
                        default_color = settings_dict['data'][current_graph.edges[edge]['classification']]['color']
                        color = current_graph.edges[edge].get('override_color', default_color)
                    else:
                        color = (1.0, 1.0, 1.0)

                elif settings_dict['show_foundation'] and edge in self.thinned_tree.foundation_graph.edges:
                    color = (0.2, 0.2, 0.2)
                else:
                    color = False
                self.superpoint_graph.edges[edge]['color'] = color

            if self.thinned_tree.repair_info is not None:

                for edge in edges(self.thinned_tree.repair_info['nodes']):
                    if edge not in self.superpoint_graph.edges:
                        self.superpoint_graph.add_edge(*edge)
                    self.superpoint_graph.edges[edge]['color'] = (0.9, 0.1, 0.9)

            for node in self.superpoint_graph.nodes:

                in_main = node in current_graph and current_graph.out_degree(node) > 0
                in_foundation = node in self.thinned_tree.foundation_graph and self.thinned_tree.foundation_graph.degree(node) > 0

                if in_main or in_foundation:

                    override_color = current_graph.nodes[node].get('override_color', False)
                    if override_color:
                        color = override_color
                    elif node == self.thinned_tree.trunk_node:
                        color = (0.1, 0.1, 0.9)
                    elif node in self.thinned_tree.tip_nodes:
                        color = (0.1, 0.9, 0.9)
                    elif in_main:
                        if current_graph.nodes[node].get('violation', False):
                            color = (0.9, 0.1, 0.1)
                        else:
                            color = (0.1, 0.9, 0.1)
                    else:
                        color = (0.4, 0.4, 0.4)
                else:
                    color = False
                self.superpoint_graph.nodes[node]['color'] = color

            return

        to_show = set(self.superpoint_graph.subgraph(max(nx.connected_components(self.superpoint_graph))).edges)

        #
        #
        # if settings_dict['thinning'] is not None:
        #     to_show.intersection_update(self.thin_skeleton(*settings_dict['thinning']))

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
                predicted_cat = np.argmax(pred['classification_values'])
                predicted_val = pred['classification_values'][predicted_cat]
                cat_info = settings_dict['data'][predicted_cat]
                if predicted_val >= cat_info['threshold']:
                    color = cat_info['color']
                else:
                    color = False

            else:
                # Single - Even if a branch doesn't have a category as its primary
                # classification, show it
                val = pred['classification_values'][settings_dict['data']['category']]
                if val < settings_dict['data']['threshold']:
                    color = False
                else:
                    color = (1.0, 1.0, 1.0)

            self.superpoint_graph.edges[edge]['color'] = color

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
            self.superpoint_graph.edges[edge]['unlikeliness'] = (negative_belief - positive_belief + 1) / 2
        self.is_classified = True

    def thin_skeleton(self):

        """
        Produces a minimum spanning tree where the weights are the unlikeliness of the edge being valid, normalized from 0 to 1.
        :return:
        """

        self.classify_edges()
        return nx.algorithms.minimum_spanning_edges(self.superpoint_graph, weight='unlikeliness', data=False)


    def initialize_final_tree(self):
        """
        :return:
        """
        self.classify_edges()
        edges_to_keep = self.thin_skeleton()
        subgraph = self.superpoint_graph.edge_subgraph(edges_to_keep)
        self.thinned_tree = ThinnedTree(self.superpoint_graph, subgraph)

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


class ThinnedTree:

    VIOLATION_COSTS = {
        'angle': 5,
        'direction': 2,
        'trunk_wrong_direction': 5,
        'topology':
            {0: 1000,
             1: 10,
             2: 5
             }
    }

    """
    INITIALIZATION FUNCTIONS
    """

    def __init__(self, base_graph, foundation_graph):

        self.base_graph = base_graph
        self.foundation_graph = foundation_graph
        self.current_graph = nx.DiGraph()
        self.current_graph.add_nodes_from(self.base_graph.nodes)
        self.trunk_node = None
        self.tip_nodes = None
        self.disconnected_tips = []

        self.estimate_trunk_node()
        self.estimate_tips()
        try:
            self.estimate_tree_from_tips()
        except Exception as e:
            print(e)

        self.determine_all_violations(commit=True)
        # self.determine_node_violations()

        self.active_iterator = None

        # For repairing
        self.repair_info = None

        # For debugging
        score = self.score_tree()
        print('Starting score is: {}'.format(score))
        print('Base graph has {} nodes'.format(len(self.base_graph.nodes)))
        print('Foundation graph has {} nodes'.format(len(self.foundation_graph.nodes)))

        self.segment_to_fix = None
        self.edges_to_queue_removal = None
        self.debug_next = False

    def estimate_trunk_node(self):
        nodes = list(max(nx.connected_components(self.foundation_graph), key=len))
        pts = np.array([self.foundation_graph.nodes[node]['point'] for node in nodes])
        estimated_trunk = nodes[np.argmax(pts[:,1])]

        #
        # est = np.median(pts, axis=0)
        # est[1] = pts[:,1].max()
        # estimated_trunk = nodes[np.argmin(np.linalg.norm(pts - est, axis=1))]

        self.trunk_node = estimated_trunk

    def estimate_tips(self):
        # Scans the tree for anything which looks "tippy".
        # This can be refined, just want something for now
        SCAN_ZONE = 0.5


        all_node_points = np.array([self.base_graph.nodes[n]['point'] for n in self.foundation_graph.nodes])
        y_points = all_node_points[:,1]
        y_min = y_points.min()
        y_max = y_points.max()

        y_start = y_min
        y_end = y_min + (y_max - y_min) * SCAN_ZONE

        valid_nodes = [n for n in self.foundation_graph.nodes
                       if y_start <= self.foundation_graph.nodes[n]['point'][1] <= y_end]
        subgraph = self.foundation_graph.subgraph(valid_nodes)
        # For each connected component in the graph, get the node with the most tip-like value
        tips = []
        for comp_nodes in nx.algorithms.components.connected_components(subgraph):
            best_node = min(comp_nodes, key=lambda x: self.base_graph.nodes[x]['point'][1])
            tips.append(best_node)

        self.tip_nodes = tips

        # FOR FUTURE REFERENCE
        # For each node, check if there's an edge that crosses the scan border.
        # If so, take that node and see if there's any chains it can connect to above.
        # If not, start a tip chain.

    def estimate_tree_from_tips(self):

        for tip_node in self.tip_nodes:
            try:
                to_trunk = nx.algorithms.shortest_paths.dijkstra_path(self.foundation_graph, tip_node, self.trunk_node)
                for s, e in zip(to_trunk[:-1], to_trunk[1:]):
                    try:
                        edge_dict = self.current_graph.edges[s, e]
                        edge_dict['count'] += 1
                    except KeyError:
                        self.current_graph.add_edge(s, e, count=1)

            except nx.NetworkXNoPath:
                self.disconnected_tips.append(tip_node)

        # Initialize assignments
        # Anything with just 1 path is a leader, everything else
        # Also assign trunk by looking at split
        for edge in self.current_graph.edges:
            edge_dict = self.current_graph.edges[edge]
            if edge_dict['count'] > 1:
                edge_dict['classification'] = 1
            else:
                edge_dict['classification'] = 2

        for edge in nx.algorithms.bfs_edges(self.current_graph, source=self.trunk_node, reverse=True):
            end, start = edge
            if self.current_graph.in_degree[end] > 1:
                break
            else:
                self.current_graph.edges[start, end]['classification'] = 0

        else:
            raise Exception("Couldn't find trunk split? Highly unlikely")





    """
    FUNCTIONS FOR DETERMINING VIOLATIONS AND ASSESSING THE TREE QUALITY
    
    To do list:
    - Maybe change search strategy to be top-down? Start at a tip. As you assign segments, check if previous encountered segments should be changed as well.
    """

    def determine_all_violations(self, graph=None, commit=True):
        if graph is None:
            graph = self.current_graph

        total = 0

        # Get all the angular violations, node by node
        for node in graph.nodes:

            in_nodes = [e[0] for e in list(graph.in_edges(node))]
            try:
                out_node = list(graph.out_edges(node))[0][1]

            except IndexError:
                # Trunk node, the only type of violation that can occur there is a topology split violation
                in_classes = [graph.edges[e]['classification'] for e in graph.in_edges(node)]
                new_violation = self.assess_topology_split_violation(0, in_classes)
                graph.nodes[node]['violation'] = new_violation
                total += new_violation
                continue

            out_class = graph.edges[node, out_node]['classification']


            all_in_classes = []
            node_violations = 0
            for in_node in in_nodes:
                in_class = graph.edges[in_node, node]['classification']
                all_in_classes.append(in_class)
                node_violations += self.assess_angular_violation(node, out_node, out_class, in_node, in_class)

            node_violations += self.assess_topology_split_violation(out_class, all_in_classes)

            if commit:
                graph.nodes[node]['violation'] = node_violations

            total += node_violations

        # Then get all the edge violations, edge by edge
        for edge in graph.edges:
            edge_class = graph.edges[edge]['classification']
            edge_violations = self.assess_edge_violation(edge[0], edge[1], edge_class)
            if commit:
                graph.edges[edge]['violation'] = edge_violations

            total += edge_violations

        return total


    def assess_angular_violation(self, node, next_node, next_class, prev_node, prev_class):
        INTER_VIOLATION = np.radians(135)
        INTRA_VIOLATION = np.radians(45)

        angle = self.get_node_turn_angle(prev_node, node, next_node)
        if next_class != prev_class and angle > INTER_VIOLATION:
            return self.VIOLATION_COSTS['angle']

        elif next_class == prev_class and angle > INTRA_VIOLATION:
            return self.VIOLATION_COSTS['angle']

        return 0

    def assess_edge_violation(self, node, next_node, classification):
        TRUNK_VIOLATION = np.radians(90)
        SUPPORT_VIOLATION = np.radians(60)
        LEADER_VIOLATION = np.radians(30)

        violation = 0

        pt = self.foundation_graph.nodes[node]['point']
        next_pt = self.foundation_graph.nodes[next_node]['point']
        diff = np.abs(pt - next_pt)
        xy_angle = np.arctan2(diff[1], diff[0])

        if classification == 1 and xy_angle > SUPPORT_VIOLATION:
            violation += self.VIOLATION_COSTS['direction']
        elif classification == 2 and xy_angle < LEADER_VIOLATION:
            violation += self.VIOLATION_COSTS['direction']

        trunk_angle = self.get_node_turn_angle(node, next_node, self.trunk_node)
        if trunk_angle > TRUNK_VIOLATION:
            violation += self.VIOLATION_COSTS['trunk_wrong_direction']

        return violation

    def assess_topology_split_violation(self, out_class, in_classes):

        if len(in_classes) < 2:
            return 0

        is_equal = map(lambda x: x == out_class, in_classes)

        if out_class == 0:
            # A trunk cannot have a split resulting in another trunk
            if any(is_equal):
                return self.VIOLATION_COSTS['topology'][out_class]
            # Can't have more than 2 splits of the support from the trunk
            if sum(map(lambda x: x == 1, in_classes)) >= 3:
                return self.VIOLATION_COSTS['topology'][out_class]
        elif sum(is_equal) > 1:
            return self.VIOLATION_COSTS['topology'][out_class]

        return 0



    def score_tree(self, tree=None, recompute_violations=True):

        if tree is None:
            tree = self.current_graph

        if not recompute_violations:
            raise NotImplementedError

        total_violations = self.determine_all_violations(tree, commit=False)
        pos_contrib = len(list(tree.edges))
        return pos_contrib - total_violations

    def reassign_tree(self, old_path, new_path=None, new_assignments=None, tree=None):
        if tree is None:
            tree = self.current_graph

        for edge in edges(old_path):
            tree.remove_edge(*edge)

        if new_path is None or new_assignments is None:
            new_path = []
            new_assignments = []

        for edge, assignment in zip(edges(new_path), new_assignments):
            tree.add_edge(*edge, classification=assignment)

        # If removing the edge induces a disconnect, throw out the part that doesn't have the trunk
        final_node_subset = set()
        for comp in nx.weakly_connected_components(tree):
            if self.trunk_node in comp:
                final_node_subset = comp
                break

        subcomp = tree.subgraph(final_node_subset)
        tree.remove_edges_from([e for e in tree.edges if e not in subcomp.edges])

        self.determine_all_violations()

    def reassign_multiple(self, reassignments, tree=None, inplace=False, disconnect_at=None):
        """
        Reassignments come in the form of a list of (segment, reassignment) pairs, where segment is a list of
        adjacent nodes to be reassigned and reassignment can either be a class integer, a corresponding list
        of classifications, or None, indicating that the component should be disconnected.
        :param reassignments:
        :return:
        """
        if tree is None:
            tree = self.current_graph
        if not inplace:
            tree = tree.copy()

        for segment, reassignment in reassignments:
            if reassignment is None or isinstance(reassignment, int):
                reassignment = [reassignment] * (len(segment) - 1)

            for edge, new_assignment in zip(edges(segment), reassignment):
                if new_assignment is None:
                    tree.remove_edge(*edge)
                else:
                    tree.edges[edge]['classification'] = new_assignment

        has_warned = False
        for comp in nx.weakly_connected_components(tree):
            if len(comp) <= 1 or self.trunk_node in comp:
                continue

            if not has_warned:
                print('Warning! Your reassignment was incomplete and led to disconnected components. Cutting out those...')
                has_warned = True

            edges_to_remove = list(tree.subgraph(comp).edges)
            for edge in edges_to_remove:
                tree.remove_edge(*edge)

        if disconnect_at is not None:
            predecessors = set(nx.algorithms.dfs_preorder_nodes(tree.reverse(), disconnect_at))
            tree = tree.subgraph(predecessors).copy()

        self.determine_all_violations(tree)

        return tree






    def score_tree_with_reassignments(self, old_path, new_path, new_assignments, tree=None):

        if tree is None:
            tree = self.current_graph.copy()

        self.reassign_tree(old_path, new_path, new_assignments, tree)
        return self.score_tree(tree)

    def score_segment(self, segment, assignments):
        # Total violations
        if isinstance(assignments, int):
            assignments = [assignments] * (len(segment) - 1)

        score = len(segment) - 1
        edges_and_assignments = list(zip(edges(segment), assignments))

        for edge, assignment in edges_and_assignments:
            score -= self.assess_edge_violation(*edge, assignment)

        for (in_edge, in_assignment), (out_edge, out_assignment) in edges(edges_and_assignments):
            score -= self.assess_angular_violation(out_edge[0], out_edge[1], out_assignment, in_edge[0], in_assignment)

        return score


    """
    SEARCH ALGORITHM TOOLS FOR SPLITTING UP THE TREE INTO LOCAL SEGMENTS AND CORRECTING THEM
    """


    def iterator(self):
        processed_actions = set()

        # This needs to be regenerated every time (or updated in a smart manner) because of updates to the structure. Could be done better
        segment_graph = self.split_graph_into_segments()
        remaining_actions = self.get_segment_graph_action_order(segment_graph, include_tips=True)
        update_actions = False

        while remaining_actions:
            segment_graph = self.split_graph_into_segments()
            if update_actions:
                remaining_actions = self.get_segment_graph_action_order(segment_graph, include_tips=True)
                update_actions = False

            action = remaining_actions[0]
            remaining_actions = remaining_actions[1:]

            if action in processed_actions:
                continue

            for edge in self.current_graph.edges:
                try:
                    del self.current_graph.edges[edge]['override_color']
                except KeyError:
                    pass
            for node in self.current_graph.nodes:
                try:
                    del self.current_graph.nodes[node]['override_color']
                except KeyError:
                    pass
            print(action)
            if isinstance(action, int):
                # Node fix

                if self.current_graph.in_degree(action) <= 1:
                    print('Node {} is no longer a joint, skipping...'.format(action))
                    continue

                print('Applying recursive fix to node {}'.format(action))
                self.current_graph.nodes[action]['override_color'] = (1.0, 0.0, 1.0)
                yield

                _, reassignments = self.reassess_segment_classifications(action, segment_graph)
                self.reassign_multiple(reassignments, inplace=True)
                yield
            else:
                if len(action) == 1:
                    print('Attempting to reconnect tip')
                    segment = action        # FOr discon node tips
                else:
                    print('Attempting to fix segment')
                    segment = segment_graph.edges[action]['segment']
                for node in segment:
                    self.current_graph.nodes[node]['override_color'] = (1.0, 0.0, 1.0)
                for edge in edges(segment):
                    self.current_graph.edges[edge]['override_color'] = (1.0, 0.0, 0.0)
                yield

                solution, _, new_score = self.segment_fix_search(segment)
                if solution is None:
                    print('No better solution was found than simply excluding the given segment')
                    self.reassign_tree(segment)

                else:
                    (new_path, new_assignments) = solution
                    print('Found tree with new score of {}'.format(new_score))
                    self.reassign_tree(segment, new_path, new_assignments)

                    if new_path[-1] != segment[-1]:
                        update_actions = True
                yield

            processed_actions.add(action)

        raise StopIteration

    def iterate(self):
        if self.active_iterator is None:
            self.active_iterator = self.iterator()

        try:
            next(self.active_iterator)
        except StopIteration:
            print('Tree iteration is all done')
            self.active_iterator = None


    def split_graph_into_segments(self, graph=None):
        """
        Takes a graph and splits it into "linear" segments
        These are based both on junctions as well as whether there are node violations
        """
        if graph is None:
            graph = self.current_graph
        final_graph = nx.DiGraph()

        queue = [n for n in graph.nodes if (graph.in_degree(n) == 0 and graph.out_degree(n) > 0)]
        # queue = [n for n in graph.nodes if n in self.disconnected_tips]
        covered_nodes = set()

        while queue:
            node = queue.pop()
            start_node = node
            if node in covered_nodes:
                continue
            covered_nodes.add(node)

            segment = [node]
            total_violation = 0

            while True:
                try:
                    next_node = list(graph.out_edges(node))[0][1]
                    next_node_violation = graph.nodes[next_node].get('violation', 0)
                    next_edge_violation = graph.edges[node, next_node]['violation']
                    # next_violation = bool(next_node_violation or next_edge_violation)

                    total_violation += next_node_violation + next_edge_violation

                    next_node_degree = graph.in_degree(next_node)

                    # if is_first:
                    #     is_violation = is_violation or next_violation
                    #     is_first = False

                    # if not is_violation and next_violation:
                    #     queue.append(node)
                    #     all_segments.append((segment, total_violation))
                    #     break
                    #
                    # elif (is_violation and not next_violation) or next_node_degree > 1:

                    if next_node_degree > 1:
                        segment.append(next_node)
                        queue.append(next_node)

                        break
                    else:
                        segment.append(next_node)
                        node = next_node


                except IndexError:
                    # Reached trunk
                    break

            last_node = next_node

            if start_node == last_node:
                continue

            final_graph.add_nodes_from([start_node, last_node])
            contained_classes = [graph.edges[e]['classification'] for e in edges(segment)]
            final_graph.add_edge(start_node, last_node, segment=segment, violations=total_violation, assignments=contained_classes)


        #
        #
        # Count upstream assignments - Done recursively
        def assign_upstream_counts(node):
            start_count = 0
            for edge in final_graph.in_edges(node):
                start_count += len(final_graph.edges[edge]['segment']) - 1 + assign_upstream_counts(edge[0])

            final_graph.nodes[node]['upstream'] = start_count
            return start_count

        try:
            assign_upstream_counts(self.trunk_node)
        except RecursionError as e:
            print(e)
            set_trace()

        return final_graph

    def get_segment_graph_action_order(self, graph, node=None, shuffle=True, debug=False, include_tips=False):

        if debug:
            set_trace()

        if node is None:
            node = self.trunk_node

        actions = []

        try:
            preds = list(graph.predecessors(node))
        except Exception as e:
            print(e)
            set_trace()
        if not preds:
            return []


        if shuffle:
            random.shuffle(preds)

        for pred in preds:
            actions += self.get_segment_graph_action_order(graph, node=pred, include_tips=False) + [(pred, node)]

        if include_tips:
            # Attempt to process disconnected tips
            discon_tip_segments = [(t,) for t in self.tip_nodes if not self.current_graph.out_degree(t)]
            actions = discon_tip_segments + actions

        return actions + [node]


    def segment_fix_search(self, segment, max_allowable_violation=10):
        """
        Pick a segment that you want to fix.
        You want to find a rerouting of that segment that increases the overall score of the tree.
        Perform the search in a way which minimizes the number of violations, but that also allows skips.

        Also allow for the possibility of just not connecting the segment at all.

        Scoring state:

        (Violations, Skips)
        Termination condition is when you reach a node outside of the segment (though that node may include the endpoint of the segment,
        if the best thing to do is simply to reroute the segment)

        Actions:
        - Should you use a skip? (You can't if you've skipped once)
        - Should you switch class from the initial? # TODO: Figure out a smart way to implement this later

        :return:
        """
        # When rerouting the graph, the tree should be split into up to two sections, one which has the trunk
        remainder_nodes = set(self.current_graph.nodes).difference(segment[1:-1])
        current_graph_subset = self.current_graph.subgraph(remainder_nodes).copy()     # Removes all edges associated with the current path
        if len(segment) == 2:
            current_graph_subset.remove_edge(segment[0], segment[1])

        connected_components = [comp for comp in nx.weakly_connected_components(current_graph_subset) if len(comp) > 1]
        assert len(connected_components) <= 2
        # In the case that you're dealing with a tip, you will still only have the main tree with no disconnected part
        # If you're dealing with the trunk, need to flip the attached/detached components, add the trunk node as the main component

        if segment[-1] == self.trunk_node:
            assert len(connected_components) == 1
            main_component = {self.trunk_node}
            detached_component = connected_components[0]

        elif len(connected_components) == 1:
            main_component = connected_components[0]
            detached_component = set()
        # Otherwise, your graph will contain two segments. You want to make sure not to loop back to your detached segment
        else:
            main_component, detached_component = connected_components
            if self.trunk_node in detached_component:
                main_component, detached_component = detached_component, main_component

        # Evaluate the tree condition when you only evaluate the main component

        # Evaluate the current state of the tree

        # Evaluate the local evaluation of the existing segment

        # Set the violation threshold to be the maximum [?]


        # Get the min and max class assignment
        in_edges = list(self.current_graph.in_edges(segment[0]))
        if not in_edges:
            max_class = 2
        else:
            max_class = min([self.current_graph.edges[e]['classification'] for e in in_edges])

        out_edges = list(self.current_graph.out_edges(segment[-1]))
        if not out_edges:
            min_class = 0
        else:
            min_class = self.current_graph.edges[out_edges[0]]['classification']




        # Set up queue, where the state is a tuple of (path, assignments, has_skipped)
        # The val should be the number of incurred violations

        queue = PriorityQueue()
        current_node = segment[0]
        start_path = (current_node, )
        start_state = (start_path, (), False)
        queue.add(start_state, 0)

        # TODO: Evaluate initial solution and make sure the returned solution is better than it

        best_solution = None
        best_tree_score = self.score_tree(self.current_graph.subgraph(main_component))
        best_violation_count = None


        # TODO: Need to count violations from start due to branching? Will do later.
        count = 0
        while queue:
            count += 1
            # if count > 2000:
            #     print('Infinite loop detected!')
            #     self.debug_next = True
            #     set_trace()



            (path, assignments, has_skipped), violations = queue.pop()
            node = path[-1]

            if violations > max_allowable_violation:
                print('Exhausted all possible paths with a sufficiently low violation count')
                break

            if self.debug_next:
                print('Current path: {}\nWith assignments: {}'.format(path, assignments))
                print('Currently {} violations'.format(violations))
                set_trace()


            # If you've skipped once, you have to follow the foundation graph
            if has_skipped:
                neighbors = set(self.foundation_graph[node])
            else:
                neighbors = set(self.base_graph[node])

            if assignments:
                last_assignment = assignments[-1]
            else:
                last_assignment = max_class

            possible_assignments = list(range(min_class, last_assignment + 1))


            if self.debug_next:
                print('Processing neighbors...')

            for neighbor, next_assignment in product(neighbors, possible_assignments):

                if self.debug_next:
                    print('\tProcessing {} as {}'.format(neighbor, next_assignment))

                # Case 1: The neighbor is in the path already, ignore it.
                if neighbor in path:
                    continue

                # Case 2: The neighbor opposes an existing edge, ignore it.
                # (Should only happen on first node since path nodes should be on
                if (neighbor, node) in current_graph_subset.edges:
                    continue

                # Case 3: The path is already worse than a found solution, ignore it.
                if best_violation_count is not None and violations > best_violation_count:
                    continue

                # Case 4: If the neighbor is in the disconnected component, it means you're self-looping. Avoid this.
                if neighbor in detached_component:
                    continue

                # Case 5 (HACK): If the neighbor doesn't exist in the foundation graph, ignore it.
                # (Happens due to the edge subgraph func throwing out some nodes)
                if neighbor not in self.foundation_graph:
                    continue

                # Check to see if this counts as a skip connection
                is_skip_connection = (node, neighbor) not in self.foundation_graph.edges
                if has_skipped:
                    assert not is_skip_connection

                # Add the neighbor to the path and compute any new angular/edge violations
                new_path = path + (neighbor, )
                new_assignments = assignments + (next_assignment, )


                try:
                    new_violation_cost = self.assess_edge_violation(node, neighbor, next_assignment) + violations
                except KeyError:
                    print('WTF')
                    set_trace()
                    new_violation_cost = 0

                if len(new_path) >= 3:
                    new_violation_cost += self.assess_angular_violation(node, neighbor, next_assignment, new_path[-3], last_assignment)

                # If you land on part of the original graph, "terminate" the search and assess the final angular violation
                if neighbor in main_component:
                    if self.debug_next:
                        print('\tNode {} is in main component'.format(neighbor))

                    out_edges = list(current_graph_subset.out_edges(neighbor))
                    if out_edges or neighbor == self.trunk_node:

                        # set_trace()
                        # Search termination - Evaluate final score and update best
                        if out_edges:
                            next_next_node = out_edges[0][1]
                            next_next_class = current_graph_subset.edges[neighbor, next_next_node]['classification']

                            # Topological violation - Cannot progress upwards in class, must go 2->1->0
                            if next_next_class > next_assignment:
                                if self.debug_next:
                                    print('\t\tThrown out since next class is {} but assignment is {}'.format(next_next_class, next_assignment))
                                continue

                            new_violation_cost += self.assess_angular_violation(neighbor, next_next_node, next_next_class, node, next_assignment)

                        # TODO: Think about a better condition for when to stop considering a path
                        if best_violation_count is None or new_violation_cost <= best_violation_count:
                            # Evaluate the tree and see if it's better than the thing
                            tree_score = self.score_tree_with_reassignments(segment, new_path, new_assignments)

                            if self.debug_next:
                                print('\t\tViolation of {} beats current best of {}. Checking if reassignment leads to better score...'.format(new_violation_cost, best_violation_count))

                            if best_tree_score is None or tree_score > best_tree_score:
                                if self.debug_next:
                                    print(
                                        '\t\tNew tree score of {} is better than previous {}! Setting new solution'.format(
                                            tree_score, best_tree_score))

                                best_tree_score = tree_score
                                best_violation_count = new_violation_cost
                                best_solution = (new_path, new_assignments)

                            else:

                                if self.debug_next:
                                    print(
                                        '\t\tNew tree score of {} is WORSE than previous {}! DIscarding solution and continuing...'.format(
                                            tree_score, best_tree_score))
                                continue
                        else:
                            if self.debug_next:
                                print('\t\tViolation cost of {} was worse than the current best violation count {}'.format(new_violation_cost, best_violation_count))
                            continue

                # If you don't land on part of the original graph, add it to the queue, but only if it's not worse than the existing solution
                new_state = (new_path, new_assignments, is_skip_connection)
                queue.add(new_state, new_violation_cost)

        return best_solution, best_violation_count, best_tree_score

    def reassess_segment_classifications(self, joint_node, segment_graph=None, start_score=0, downstream_class=None):
        """ Algorithm description:
                Select a joint with a downstream classification.
                Look at each of the in edges. For each in edge, do 1 of the 3 things to compute a violation score:
                    - Uniformly assign it a classification based off of the downstream
                    - Leave the existing classification, if it isn't already uniform.
                    - Cut out the branch entirely. (This also cuts out all the upstreams)

                For the first two situations, compute the violations induced by reassigning the branch to the given category.
                    - If more violations are created than there exist upstream nodes from the start of the segment, including,
                      the existing downstream violation, throw out the option
                    - Otherwise, apply a recursive form of the algorithm to figure out the rest of the upstream assignments,
                      passing in the current score to make sure that current violations have an impact on upstream assignments

                At this point, for each edge, you will have some number of feasible classifications for each edge.
                Find the combination of them which maximizes the total score when topological constraints are taken into account.
                Then commit them

        """




        if segment_graph is None:
            segment_graph = self.split_graph_into_segments()
        if joint_node not in segment_graph:
            raise ValueError('Input node is not a junction for a segment')

        if downstream_class is None:
            try:
                out_edge = list(self.current_graph.out_edges(joint_node))[0]
                downstream_class = self.current_graph.edges[out_edge]['classification']
            except IndexError:
                downstream_class = 0

        edges = []
        edge_options = []
        for input_segment_edge in segment_graph.in_edges(joint_node):

            options = []
            edges.append(input_segment_edge)

            upstream = segment_graph.nodes[input_segment_edge[0]]['upstream']
            existing_segment = segment_graph.edges[input_segment_edge]['segment']
            existing_assignments = segment_graph.edges[input_segment_edge]['assignments']

            # Check what possible valid assignments are possible and
            valid_assignments = list(range(downstream_class, 2+1))
            if len(set(existing_assignments)) > 1:
                valid_assignments.append(existing_assignments)

            for assignments in valid_assignments:
                if not isinstance(assignments, int):
                    new_downstream_class = assignments[0]
                else:
                    new_downstream_class = assignments

                partial_score = self.score_segment(existing_segment, assignments)
                if partial_score + upstream <= 0:
                    continue

                score, upstream_assignments = self.reassess_segment_classifications(input_segment_edge[0], segment_graph, start_score,
                                                                                    downstream_class=new_downstream_class)
                score += partial_score
                upstream_assignments.append((existing_segment, assignments))

                options.append((score, upstream_assignments))


            # Also allow the possibility of just not assigning anything
            segment_and_reassignment = [(existing_segment, None)]
            options.append([0, segment_and_reassignment])
            edge_options.append(options)

        # Get all the edge combinations and sort them based off of their total score

        if not edge_options:
            return (0, [])

        all_combos = list(product(*edge_options))
        all_combos.sort(key=lambda l: sum([x[0] for x in l]), reverse=True)

        best_score = 0
        best_reassignments = []
        for combo in all_combos:
            naive_score = sum([x[0] for x in combo])        # Naive because does not take into account topology

            # If all remaining scores have no possibility of beating the existing best, just terminate the search, since the options are sorted
            if start_score + naive_score < best_score:
                break

            all_reassignments = []
            for _, reassignments in combo:
                all_reassignments.extend(reassignments)

            # Apply reassignments, get score including topology constraints
            modified_tree = self.reassign_multiple(all_reassignments, disconnect_at=joint_node)
            final_score = self.score_tree(modified_tree)

            # Temporary hack: Since the downstream edge isn't in the graph, but we need to consider the topology violation
            # induced by this assignment, do it manually
            all_classes = [modified_tree.edges[e]['classification'] for e in modified_tree.in_edges(joint_node)]
            final_score -= self.assess_topology_split_violation(downstream_class, all_classes)

            if final_score > best_score:
                best_score = final_score
                best_reassignments = all_reassignments

        return best_score, best_reassignments

    def get_node_turn_angle(self, a, b, c):
        pt_a = self.foundation_graph.nodes[a]['point']
        pt_b = self.foundation_graph.nodes[b]['point']
        pt_c = self.foundation_graph.nodes[c]['point']

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


    def handle_repair(self, node, assignment):

        if self.repair_info is None:
            self.initialize_repair(node)
            return

        if node == self.repair_info['nodes'][0]:
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

            self.determine_all_violations(commit=True)
            self.repair_info = None

        elif node in self.repair_info['all_components'] and not node in self.repair_info['main_component']:
            print("You cannot select this node, as this would cause a loop in the tree!")
        else:
            # Clicked on a node that isn't in part of any component
            # Add the node and assignment and move along
            self.repair_info['nodes'].append(node)
            self.repair_info['assignments'].append(assignment)

            if not node in self.repair_info['main_component']:
                return

            # At this point, the node is in the main component
            # Commit all edges and assignments to the temporary tree, then replace the current tree and recompute
            for edge, assignment in zip(edges(self.repair_info['nodes']), self.repair_info['assignments']):
                self.repair_info['tree'].add_edge(*edge, classification=assignment)

            self.current_graph = self.repair_info['tree']
            self.determine_all_violations(commit=True)
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
