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

class Flags(Enum):
    UNPROCESSED = 0
    SKELETONIZED = 1

    RADII_ASSIGNED = 2
    MESH_CREATED = 3


SUPPORT_LEN_THRESHOLD = 0.4

class TreeModel(object):

    def __init__(self):
        self.points = None
        self.graph = None
        self.segment_graph = None
        self.segments = []
        self.flags = {}
        self.mesh = None
        self.kd_tree = None
        self.net = None
        self.bin_data = {}
        self.trunk_guess = None
        self.raster = None
        self.raster_info = None

        self.superpoint_graph = None

        # For color output
        self.highlighted_points = defaultdict(list)
        self.point_beliefs = None

        # Temporary
        self.cover = None
        self.toggle = 0

        self.status = Flags.UNPROCESSED

    @classmethod
    def from_point_cloud(cls, pc, process=False, kd_tree_pts = 100, bin_width=0.01):
        new_model = cls()
        if process:
            pc = preprocess_point_cloud(pc)

        new_model.points = pc
        new_model.kd_tree = KDTree(pc, kd_tree_pts)
        new_model.process_bins(bin_width)

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


    def skeletonize(self):

        if self.status.value >= Flags.SKELETONIZED.value:
            return

        print('Computing skeletonization...')

        graph = skel.construct_mutual_k_neighbors_graph(self.points, 15, 0.05)
        skel.clean(graph, threshold=0.10)

        spanning_tree = nx.algorithms.tree.minimum_spanning_tree(graph)
        while True:
            rez = skel.clean(spanning_tree, threshold=0.05)
            if not rez:
                break

        segments = skel.split_graph_into_segments(spanning_tree)

        smoothed_segments = []

        for segment in segments:
            smoothed_segments.append(skel.redistribute_branch_nodes(segment, 0.10))

        graph = nx.Graph()

        vertex_associations = defaultdict(lambda: {'segments': [], 'is_endpoint': False})
        edge_id_counter = 0

        for i, segment in enumerate(smoothed_segments):

            self.segments.append(Segment(segment, segment_id=i))

            vertex_associations[segment[0]]['is_endpoint'] = True
            vertex_associations[segment[-1]]['is_endpoint'] = True

            for pt in segment:
                vertex_associations[pt]['segments'].append(i)
                graph.add_node(pt, segments=vertex_associations[pt]['segments'], is_endpoint=vertex_associations[pt]['is_endpoint'])

            for start, end in zip(segment[:-1], segment[1:]):
                graph.add_edge(start, end, segment=i, edge_id=edge_id_counter)
                edge_id_counter += 1

        self.graph = graph

        # Make the segment-based graph
        edges = []
        for vertex, metadata in vertex_associations.items():
            if len(metadata['segments']) <= 1:
                continue

            connections = combinations(metadata['segments'], 2)
            edges.extend(connections)

        self.segment_graph = nx.Graph()
        self.segment_graph.add_nodes_from(range(len(self.segments)))
        self.segment_graph.add_edges_from(edges)

        print('Running branch classification...')
        self.run_classification()

        print('Skeletonization complete!')
        self.status = Flags.SKELETONIZED

    def run_classification(self):
        self.classify_support_branches()
        self.classify_vertical_leaders()
        self.classify_targets()


    def classify_support_branches(self):
        candidates = [seg for seg in self.segments if abs(seg.angle) < np.pi / 4]
        y_loc = []
        weights = []
        corresponding_segment = []
        for segment in candidates:
            for s, e in segment.edges():
                y_loc.append((s[1] + e[1]) / 2)
                weights.append(np.linalg.norm(np.array(s) - np.array(e)))
                corresponding_segment.append(segment.id)

        y_loc = np.array(y_loc)
        corresponding_segment = np.array(corresponding_segment)



        bin_size = 0.10
        bins = np.arange(min(y_loc), max(y_loc) + bin_size, bin_size)
        hist, hist_edges = np.histogram(y_loc, bins=bins, weights=weights, density=True)

        req_height = (hist.max() + hist.min()) / 2

        peaks = signal.find_peaks(hist, height=req_height, distance=2)[0]
        peak = max(peaks)
        eligible = (y_loc > hist_edges[peak] - 0.05) & (y_loc < hist_edges[peak + 1] + 0.05)
        cand_segments = set(corresponding_segment[eligible])
        subgraph = self.segment_graph.subgraph([seg.id for seg in candidates])
        connected_components = nx.connected_components(subgraph)

        for subgraph_nodes in connected_components:

            if not cand_segments.intersection(subgraph_nodes):
                continue

            sum_len = 0
            for segment_id in subgraph_nodes:
                sum_len += Segment.get_by_id(segment_id).length
            if sum_len < SUPPORT_LEN_THRESHOLD:
                continue

            for segment_id in subgraph_nodes:
                Segment.get_by_id(segment_id).classification = SegmentType.SUPPORT


    def classify_vertical_leaders(self):


        endpoint_graph = nx.Graph()
        for seg_id in self.segment_graph.nodes:
            seg = Segment.get_by_id(seg_id)
            endpts = [tuple(pt) for pt in seg.endpoints]
            is_support = seg.classification == SegmentType.SUPPORT
            endpoint_graph.add_nodes_from(endpts, is_support_node=is_support)
            if not is_support:
                endpoint_graph.add_edge(endpts[0], endpts[1], length=seg.length, id=seg_id)

        for sg_nodes in nx.connected_components(endpoint_graph):
            sg = endpoint_graph.subgraph(sg_nodes)
            roots = [n for n,data in sg.nodes(data=True) if data['is_support_node']]
            if not roots:
                # Heuristically assign the top-most terminal vertex
                candidate_roots = [n for n, deg in sg.degree if deg == 1]
                roots = [min(candidate_roots, key=lambda r: r[1])]

            for root in roots:
                root_neighbors = set(sg[root])
                for n in root_neighbors:

                    sg_mod = sg.copy()
                    sg_mod.remove_nodes_from(root_neighbors.difference({n}))

                    dists, paths = nx.algorithms.shortest_paths.single_source_dijkstra(sg_mod, root, weight='length')
                    winner = paths[max(dists, key=dists.get)]

                    for s, e in zip(winner[:-1], winner[1:]):
                        seg_id = sg_mod.edges[(s, e)]['id']
                        Segment.get_by_id(seg_id).classification = SegmentType.LEADER

    def classify_targets(self):
        graph = self.segment_graph.copy()
        for seg_id in graph.nodes:
            if Segment.get_by_id(seg_id).classification == SegmentType.LEADER:
                neighbors = set(graph[seg_id])
                for neighbor in neighbors:
                    seg = Segment.get_by_id(neighbor)
                    if seg.classification == SegmentType.UNASSIGNED:
                        seg.classification = SegmentType.TARGET

    def iterate_skeleton_segments(self):

        if self.points is None:
            raise StopIteration()

        self.skeletonize()
        for segment in self.segments:
            color = segment.get_segment_color()
            for s, e in segment.edges():
                yield (s, e), color
        raise StopIteration()

    def assign_branch_radii(self):

        if self.status.value >= Flags.RADII_ASSIGNED.value:
            return

        self.skeletonize()
        skel.create_edge_point_associations(self.graph, self.points, in_place=True)
        radii = {}
        for a, b, data in self.graph.edges(data=True):
            edge = (a,b)
            try:
                assoc = data['associations']
            except KeyError:
                continue

            if len(assoc) < 4:
                print('Dropping cylinder with {} pts'.format(len(assoc)))

            cyl = Cylinder()
            cyl.set_fit_pts(assoc[0], assoc, self.points)
            cyl.optimize_cyl(0.005, 0.10)
            radii[edge] = 0.01 if cyl.radius > 0.10 else cyl.radius

        nx.set_edge_attributes(self.graph, radii, name='radius')
        self.status = Flags.RADII_ASSIGNED

        return radii

    def create_mesh(self):
        if self.status.value >= Flags.MESH_CREATED.value:
            return

        self.assign_branch_radii()
        self.mesh = mesh.process_skeleton(self.graph, default_radius=0.01)
        self.status = Flags.MESH_CREATED
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


    def produce_open_cover(self, radius, min_points=8, neighbor_radius=None):
        to_assign = np.arange(0, self.points.shape[0])
        np.random.shuffle(to_assign)
        output = {}
        while len(to_assign):
            idx = to_assign[0]
            all_pts_idx, _ = self.query_neighborhood(radius, pt_index=idx)
            to_assign = np.setdiff1d(to_assign, all_pts_idx, assume_unique=True)
            if neighbor_radius is not None:
                all_pts_idx, _ = self.query_neighborhood(neighbor_radius, pt_index=idx)

            output[idx] = all_pts_idx

        output = {k: output[k] for k in output if len(output[k]) >= min_points}

        return output

    def load_superpoint_graph(self, radius=0.10, min_points=8, max_neighbors=10, neighbor_radius=None):

        cover = self.produce_open_cover(radius, min_points, neighbor_radius=neighbor_radius)
        reference_nodes = np.array(list(cover.keys()))
        pts = np.array([self.points[k] for k in reference_nodes])

        superpoint_graph = skel.construct_mutual_k_neighbors_graph(pts, max_neighbors, radius * 1.5, node_index=reference_nodes)
        self.superpoint_graph = superpoint_graph

        all_superpoints = {}
        for node, assoc_indexes in cover.items():
            superpoint = Superpoint(node, assoc_indexes, self.points, radius, flow_axis=1, reverse_axis=True,
                                    global_ref=np.median(self.points, axis=0), raster_info=self.get_raster_dict(self.points[node]))
            all_superpoints[node] = superpoint

        for node, superpoint in all_superpoints.items():
            superpoint.set_neighbor_superpoints([all_superpoints[n] for n in self.superpoint_graph[node]])


        nx.set_node_attributes(self.superpoint_graph, all_superpoints, name='superpoint')

    #
    # def add_superpoint_graph_attributes(self):
    #     if self.superpoint_graph is None:
    #         self.load_superpoint_graph()
    #
    #     edge_densities = {}
    #     for edge in self.superpoint_graph.edges:
    #         edge_densities[edge] = self.query_segment_density(self.points[edge[0]], self.points[edge[1]])
    #
    #     nx.set_edge_attributes(self.superpoint_graph, edge_densities, name='density')
    #
    #     # to_update = defaultdict(dict)
    #     # for node in self.superpoint_graph:
    #     #
    #     #     degree = self.superpoint_graph.degree[node]
    #     #     for neighbor in self.superpoint_graph[node]:
    #     #         superpoint =3




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

    # def connect_open_cover(self, cover=None, neighbors=4, max_dist=0.20):
    #
    #     if cover is None:
    #         if self.cover is None:
    #             cover = self.produce_open_cover(0.10)
    #             self.cover = cover
    #         else:
    #             cover = self.cover
    #
    #     self.toggle = (self.toggle + 1) % 2
    #     if self.toggle:
    #         print('RUNNING WITH WEIGHTS')
    #     else:
    #         print('RUNNING WITHOUT WEIGHTS')
    #
    #     reference_nodes = np.array(list(cover.keys()))
    #     reference_pts = self.points[reference_nodes]
    #
    #
    #     superpoint_graph = skel.construct_mutual_k_neighbors_graph(reference_pts, neighbors, max_dist, leafsize=20, node_index=True)
    #
    #     all_frames = {}
    #     # For reference
    #     all_scales = {}
    #     all_normal_frames = {}
    #     for node in reference_nodes:
    #         pt_indexes = cover[node]
    #         pts = self.points[pt_indexes]
    #         node_pt = self.points[node]
    #
    #
    #         # Compute the TF matrix which brings a point into the frame of the
    #         u, s, v = svd(pts - pts.mean(axis=0))
    #         tf = np.identity(4)
    #         tf[:3, :3] = v
    #         tf[:3, 3] = -v.dot(node_pt)
    #
    #         all_normal_frames[node] = tf.copy()
    #
    #         # Penalize distances in non-primary directions
    #         scale_vals = np.ones(4)
    #         if self.toggle:
    #             scale_vals[:3] = s[0] / s
    #         tf = np.diag(scale_vals).dot(tf)
    #
    #         all_frames[node] = tf
    #         all_scales[node] = scale_vals
    #
    #
    #     superpoint_graph = nx.Graph(nx.to_directed(superpoint_graph))
    #     dists = {}
    #     segment_densities = []
    #     for s, e in superpoint_graph.edges:
    #
    #
    #
    #         frame_start = all_frames[reference_nodes[s]]
    #         pt_end = reference_pts[e]
    #         pt_start = reference_pts[s]
    #         pt_homog = np.ones(4)
    #         pt_homog[:3] = pt_end
    #
    #         density = self.query_segment_density(pt_start, pt_end)
    #         segment_densities.append(density)
    #         if density < 10:
    #             superpoint_graph.remove_edge(s, e)
    #             continue
    #
    #         diff = pt_end - pt_start
    #         # Drop the z-component so that only upward movement in the xy plane matters
    #         diff = diff[:2]
    #         diff = diff / np.linalg.norm(diff)
    #
    #
    #
    #
    #         dot_prod = np.array([0, -1]).dot(diff)
    #         if dot_prod > 1:
    #             dot_prod = 1
    #         elif dot_prod < -1:
    #             dot_prod = -1
    #
    #         angle = np.arccos(dot_prod)
    #         scaled_multiplier = (angle / np.pi) * (2 if self.toggle else 0) + 1   # Ranges from 1 to 2 - Could be modified
    #         # print(scaled_multiplier)
    #
    #         # normal_frame_pt = all_normal_frames[reference_nodes[s]].dot(pt_homog)[:3]
    #         scaled_frame_pt = frame_start.dot(pt_homog)[:3]
    #
    #         dists[(s, e)] = scaled_multiplier * np.linalg.norm(scaled_frame_pt)
    #
    #     nx.set_edge_attributes(superpoint_graph, dists, name='dist')
    #     # Pick random node to start at and run shortest paths
    #
    #
    #     node_dists = np.linalg.norm(reference_pts - self.trunk_guess, axis=1)
    #     first_node = np.argmin(node_dists)
    #
    #     all_paths = nx.algorithms.single_source_dijkstra(superpoint_graph, first_node, weight='dist')[1]
    #
    #     # TODO: There's def a better way to do this
    #
    #     final_graph = nx.Graph()
    #     final_graph.add_nodes_from(superpoint_graph.nodes)
    #     for _, path in all_paths.items():
    #         for s, e in zip(path[:-1], path[1:]):
    #             final_graph.add_edge(s, e)
    #
    #     # mst = nx.algorithms.tree.minimum_spanning_tree(superpoint_graph)
    #     segs = skel.split_graph_into_segments(final_graph)
    #     self.segments = []
    #     for seg in segs:
    #         self.segments.append(Segment(reference_pts[seg]))
    #
    #     self.status = Flags.SKELETONIZED    # Hack for testing


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

    # Code for binning
    def process_bins(self, bin_size, override=False):
        print('Processing bins...')
        if not override and self.bin_data:
            if self.bin_data['bin_size'] != bin_size:
                print('Warning: Bins have already been processed, and you\'ve asked for a different bin size.')
                print('Please pass in override=True if you want to reprocess the bins.')
            return

        self.bin_data = {}
        self.bin_data['bin_size'] = bin_size
        self.bin_data['bin_counts'] = defaultdict(lambda: 0)
        all_bins = self.get_bins(self.points)
        for row in all_bins:
            self.bin_data['bin_counts'][tuple(row)] += 1

    def get_bins(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        bins = np.floor(points / self.bin_data['bin_size'] - 0.5).astype(np.int)
        return bins

    def query_segment_density(self, start, end):

        dist = np.linalg.norm(start - end)
        cells_to_traverse = np.floor(dist / self.bin_data['bin_size']).astype(np.int)
        to_query = np.linspace(0, 1, endpoint=True, num=cells_to_traverse)
        interpolated_pts = to_query.reshape((-1, 1)).dot((end - start).reshape((1, -1))) + start
        interpolated_indexes = self.get_bins(interpolated_pts)
        count = 0
        for index in interpolated_indexes:
            count += self.bin_data['bin_counts'].get(tuple(index), 0)

        return count / dist

    #
    # def do_pca(self):
    #     from CylinderCover import CylinderCover, MyPointCloud
    #     my_pcd = MyPointCloud()
    #     my_pcd.load_point_cloud(self.points)
    #     cover = CylinderCover(pcd=my_pcd)
    #     cover.find_good_pca(0.5, 0.20, 0.01, 0.10)
    #
    #     set_trace()

    def get_bitmap_histogram(self, axis=1, callback=lambda: None):
        counts = defaultdict(lambda: 0)
        for key, val in self.bin_data['bin_counts'].items():
            counts[key[axis]] += val
        counts = pd.Series(counts).reindex(np.arange(min(counts), max(counts) + 1)).fillna(0)
        smoothed = counts.rolling(7).mean().shift(-3)
        smoothed = smoothed / smoothed.max()

        # plt.plot(smoothed)

        peaks_idx, props = signal.find_peaks(smoothed.values, prominence=0.1)
        peaks = smoothed.index[peaks_idx]
        threshold = max(peaks) * self.bin_data['bin_size'] - 0.20       # Hard-coded



        if self.superpoint_graph is None:
            self.load_superpoint_graph()


        superpoint_graph = self.superpoint_graph.copy()
        all_superpoints = {node: superpoint_graph.nodes[node]['superpoint'] for node in superpoint_graph}
        reference_nodes = np.array(list(superpoint_graph.nodes))

        superpoints_to_nix = [index for index, superpoint in all_superpoints.items() if superpoint < threshold]

        # cand_superpoints = {k: sp for k, sp in all_superpoints.items() if sp > threshold and sp.flow_angle < np.pi / 4 and sp.classification['Branch'] > 0.95}
        cand_superpoints = [k for k, sp in all_superpoints.items() if
                            sp > threshold and sp.flow_angle < np.pi / 4 and sp.classification['Branch'] > 0.95]
        # Algo:
        # Find all connected segments. Order by top-most point
        # Look for candidate branches in cones. Check which ones are connected

        connected_components = nx.algorithms.connected_components(superpoint_graph.subgraph(cand_superpoints))
        all_segments = []
        resolved_nodes = set()
        for superpoint_refs in connected_components:
            superpoints = sorted([all_superpoints[idx] for idx in superpoint_refs])
            subgraph = superpoint_graph.subgraph(superpoint_refs)

            for index, superpt in enumerate(superpoints):
                expected_degree = 1 if (index == 0 or index == len(superpoints) - 1) else 2
                actual_degree = subgraph.degree[superpt.ref_index]
                if actual_degree == expected_degree:
                    continue
                # TODO: This is where the tie-breaking logic comes in
                print('Unexpected degree {} found at index {} of {}'.format(actual_degree, index, len(superpoints) - 1))
                pass

            all_segments.append(superpoints)
            resolved_nodes.update(superpoint_refs)

        # Sort by the position of the uppermost segment
        all_segments = sorted(all_segments, key=lambda seg: seg[-1])
        start_to_cand_ends_proposals = defaultdict(dict)
        end_to_cand_starts_proposals = defaultdict(list)

        self.highlighted_points = defaultdict(list)
        for seg in all_segments:
            for superpoint in seg:
                self.highlighted_points[(0.2, 0.2, 0.8, 1.0)].extend(superpoint.neighbor_index)



        for segment_id, current_segment in enumerate(all_segments):

            for superpoint in current_segment:
                self.highlighted_points[(0.2, 0.8, 0.2, 1.0)].extend(superpoint.neighbor_index)

            callback()

            # For the current segment, check what other segments have been proposed to connect to it
            possible_connections = end_to_cand_starts_proposals[segment_id]
            possible_start_desiredness = {}     # How much does each starting branch want to connect to this segment?
            start_to_target_desiredness = {}    # For each segment ID, maps segment ID to ranking
            for possible_connection in possible_connections:
                proposals = start_to_cand_ends_proposals[possible_connection]
                all_ranks = pd.Series({cand: -proposals[cand]['directness'] for cand in proposals}).rank(method='min')
                rank = all_ranks[segment_id]


                possible_start_desiredness[possible_connection] = rank
                start_to_target_desiredness[possible_connection] = all_ranks

            possible_start_desiredness = pd.Series(possible_start_desiredness)

            determined_rank = 0
            contenders = []
            for rank in range(1, len(possible_start_desiredness) + 1):
                contenders = possible_start_desiredness[possible_start_desiredness <= rank].index
                if len(contenders) <= rank:
                    determined_rank = rank
                    break
            else:
                if len(contenders):
                    raise Exception("You shouldn't be here!")

            if len(contenders):

                filter_rank = lambda ser: ser[ser <= determined_rank]
                best_assignments = None
                best_directness = 0

                for assignment_set in product_with_throwaways(*[filter_rank(start_to_target_desiredness[seg_id]).index for seg_id in contenders]):

                    for assignments in assignment_set:
                        if len(set(assignments)) < determined_rank:        # Multiple segments are trying to connect to the same endpoint
                            continue

                        success = True
                        directnesses = []
                        nodes_crossed = set()
                        for cand, assignment in zip(contenders, assignments):
                            if assignment is None:
                                continue
                            info = start_to_cand_ends_proposals[cand][assignment]
                            path = info['path']
                            exp_nodes = len(nodes_crossed) + len(path)
                            nodes_crossed.update(path)
                            if len(nodes_crossed) != exp_nodes:
                                # This set of assignments is invalid because of collisions
                                success = False
                                break

                            directnesses.append(info['directness'])

                        if success:
                            directness = np.mean(directnesses)
                            if directness > best_directness:
                                best_directness = directness
                                best_assignments = assignments

                    # At this point, we check if there was at least one set of assignments which was able to resolve
                    # If not, we start considering alternatives in which we don't match up starts to ends
                    if best_assignments is not None:
                        break
                else:
                    raise Exception("You shouldn't be here, at least one candidate should have been able to resolve!")

                if best_assignments is not None:
                    for cand, assignment in zip(contenders, best_assignments):


                        if assignment is not None:
                            start_segment = all_segments[cand]
                            end_segment = all_segments[assignment]
                            path = start_to_cand_ends_proposals[cand][assignment]['path']
                            assert start_segment[-1].ref_index == path[0] and path[-1] == end_segment[0].ref_index

                            intermediate = [all_superpoints[p] for p in path[1:-1]]
                            for superpoint in intermediate:
                                self.highlighted_points[(0.2, 0.2, 0.8, 1.0)].extend(superpoint.neighbor_index)

                            all_segments[assignment] = start_segment + intermediate + end_segment
                            all_segments[cand] = None
                            resolved_nodes.update(path)


                        all_proposals = start_to_cand_ends_proposals[cand].keys()
                        for obsolete_proposal in all_proposals:
                            end_to_cand_starts_proposals[obsolete_proposal].remove(cand)
                        del start_to_cand_ends_proposals[cand]

                elif best_assignments is None and len(contenders) > 0:
                    print('!!!!!!!FAILED RESOLUTION!!!!!!!!')
                    set_trace()

            # =========================================
            # Find candidates by shooting out in an arc
            # =========================================

            current_end = current_segment[-1]

            for candidate_id, candidate_segment in enumerate(all_segments[segment_id+1:], start=segment_id+1):
                candidate_start = candidate_segment[0]
                if current_end > candidate_start:
                    continue

                main_dist, planar_dist, angle = current_end.compute_cylindrical_coords(candidate_start)
                if angle > np.pi / 4:
                    continue

                # Check if there's a connection which doesn't route through other existing branches
                valid_nodes = set(reference_nodes).difference(resolved_nodes).difference(superpoints_to_nix).union([candidate_start.ref_index, current_end.ref_index])
                subgraph = superpoint_graph.subgraph(valid_nodes)
                try:
                    dist, path = nx.algorithms.single_source_dijkstra(subgraph, source=current_end.ref_index,
                                                                      target=candidate_start.ref_index, weight='weight')
                except nx.NetworkXNoPath:
                    continue

                directness = np.linalg.norm(candidate_start.ref_point - current_end.ref_point) / dist
                start_to_cand_ends_proposals[segment_id][candidate_id] = {
                    'path': path,
                    'dist': dist,
                    'directness': directness
                }
                end_to_cand_starts_proposals[candidate_id].append(segment_id)

                for superpoint in candidate_segment:
                    self.highlighted_points[(0.8, 0.2, 0.8, 1.0)].extend(superpoint.neighbor_index)

            callback()
            yield

            for superpoint in current_segment:
                self.highlighted_points[(0.8, 0.2, 0.2, 1.0)].extend(superpoint.neighbor_index)


            try:
                del self.highlighted_points[(0.8, 0.2, 0.8, 1.0)]
            except:
                pass

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


def product_with_throwaways(*choices):
    """
    Gives all permutations, but allows for the the selection of None values.
    Prioritizes permutations with no None selections.

    E.g. for [1, 2], [a, b] will produce the following blocks:
    ---
    [1, a]
    [1, b]
    [2, a]
    [2, b]
    ---
    [1, None]
    [2, None]
    [None, a]
    [None, b]
    :param choices:
    :return:
    """

    choices = [list(choice) for choice in choices]
    indexes = list(range(len(choices)))
    for n_exclude in range(len(choices)):
        all_results = []
        for inds_to_exclude in combinations(indexes, n_exclude):
            modified_choices = []
            for index in indexes:
                if index in inds_to_exclude:
                    modified_choices.append([None])
                else:
                    modified_choices.append(choices[index])
            all_results.extend(product(*modified_choices))

        yield all_results




    pass

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

    def __init__(self, ref_index, neighbor_index, pc, radius, flow_axis=0, reverse_axis=False, global_ref=None, raster_info=None):
        self.ref_index = ref_index
        self.neighbor_index = neighbor_index
        self.ref_point = pc[ref_index]
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
