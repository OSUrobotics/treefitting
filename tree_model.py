import mesh
import skeletonization as skel
import networkx as nx
from collections import defaultdict
import numpy as np
from enum import Enum
import scipy.signal as signal
from itertools import combinations
import sys
import os
from Cylinder import Cylinder

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

        self.status = Flags.UNPROCESSED

    @classmethod
    def from_point_cloud(cls, pc):
        new_model = cls()
        new_model.points = pc
        return new_model


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
            smoothed_segments.append(skel.smooth_graph_nodes(segment, 0.015))

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

        self.mesh = mesh.process_skeleton(self.graph, default_radius=0.01)
        self.status = Flags.MESH_CREATED
        return

    def output_mesh(self, file_name):
        self.create_mesh()
        import pymesh
        my_mesh = pymesh.form_mesh(self.mesh['v'], self.mesh['f'])
        pymesh.save_mesh(file_name, my_mesh)


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



if __name__ == '__main__':
    file_path = sys.argv[1]