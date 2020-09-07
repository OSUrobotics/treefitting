#!/usr/bin/env python

# THIS IS COPIED FROM MY OTHER FILE

import numpy as np
numpy_ver = [int(x) for x in np.version.version.split('.')]
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial import KDTree
from collections import defaultdict
import pandas as pd


def view_edges_in_matplotlib(edge_list):

    if isinstance(edge_list, nx.Graph):
        edge_list = edge_list.edges.items()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    edges_processed = [np.array(x) for x in [edge[0] for edge in edge_list]]
    collection = Line3DCollection(edges_processed)
    ax.add_collection3d(collection)

    fig.show()




def get_node_dist(n1, n2):
    return np.linalg.norm(np.array(n1) - np.array(n2))

def construct_mutual_k_neighbors_graph(pc, k, max_dist, eps=0.0, leafsize=10, node_index=False):
    """
    Constructs a graph from the points in point cloud in which each point is connected to its k-nearest neighbors,
    but only if the connection is mutual.

    :param pc: A numpy array
    :param k: The number of neighbors to search for each point
    :param max_dist: Maximum distance a connection is allowed to have for the graph
    :param eps: Approximate search epsilon
        (see https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query)
    :return: An undirected NetworkX graph
    """

    tree = KDTree(pc, leafsize=leafsize)
    _, indexes = tree.query(pc, k+1, eps, distance_upper_bound=max_dist)
    indexes = indexes[:, 1:]    # The nearest neighbor is always itself

    graph = nx.Graph()
    edge_counts = defaultdict(lambda: 0)
    idx_to_node_dict = {}
    idx_to_point_dict = {}

    for current_idx in range(indexes.shape[0]):

        if isinstance(node_index, np.ndarray):
            node = node_index[current_idx]
            pt = pc[current_idx]
        elif node_index:
            node = current_idx
            pt = pc[current_idx]
        else:
            pt = pc[current_idx]
            node = tuple(pt)
        graph.add_node(node)
        idx_to_node_dict[current_idx] = node
        idx_to_point_dict[current_idx] = pt

        for col in range(k):
            neighbor_idx = indexes[current_idx, col]
            if neighbor_idx >= pc.shape[0]:
                break

            if neighbor_idx < current_idx:
                edge_counts[(neighbor_idx, current_idx)] += 1
            else:
                edge_counts[(current_idx, neighbor_idx)] += 1

    nx.set_node_attributes(graph, {idx_to_node_dict[i]: idx_to_point_dict[i] for i in idx_to_node_dict}, name='point')

    for (a, b), v in edge_counts.items():
        if v < 2:
            continue
        node_a = idx_to_node_dict[a]
        node_b = idx_to_node_dict[b]
        pt_a = idx_to_point_dict[a]
        pt_b = idx_to_point_dict[b]
        graph.add_edge(node_a, node_b, weight=get_node_dist(pt_a, pt_b))

    return graph


def clean(graph, threshold=0):

    """
    Cleans a tree by outer branches whose lengths are less than the threshold. Runs recursively.
    :param graph: A NetworkX graph
    :param threshold: A non-negative float
    :return: Returns the number of removed nodes. Modifications are made in-place to the graph.
    """


    endpoints = [n for n, v in graph.degree if v == 1]
    to_remove = set()

    for endpoint in endpoints:

        if endpoint in to_remove:
            continue

        accumulated_dist = 0
        visited_nodes = set()
        active_node = endpoint
        start_node = endpoint

        while True:
            neighbors = set(graph.neighbors(active_node)).difference(visited_nodes)
            if len(neighbors) == 0:
                # Completely 1-D structure - not useful! Toss out
                visited_nodes.add(active_node)
                to_remove = to_remove.union(visited_nodes)
                break

            elif len(neighbors) == 1:
                (neighbor, ) = neighbors
                try:
                    accumulated_dist += graph[active_node][neighbor]['weight']
                except KeyError:
                    accumulated_dist += get_node_dist(active_node, neighbor)

                visited_nodes.add(active_node)  # Does not add the neighbor, in case of a multi-D neighbor
                active_node = neighbor

            else:   # Junction node

                endpoint_d = get_node_dist(active_node, start_node)

                if accumulated_dist < threshold or endpoint_d < threshold:
                    to_remove = to_remove.union(visited_nodes)      # Cuts out all nodes NOT including junction node
                break

    graph.remove_nodes_from(to_remove)
    return len(to_remove)

def get_point_line_distance(point_or_points, start, end):

    diff = end - start
    return np.linalg.norm(np.cross(diff, start-point_or_points) / np.linalg.norm(diff), axis=1)

def extract_segment(graph, starting_edge):

    prev, current = starting_edge
    segment = [prev, current]

    while True:
        if graph.degree(current) != 2:
            break

        neighbors = graph[current]
        for neighbor in neighbors:
            if neighbor != prev:
                segment.append(neighbor)
                prev, current = current, neighbor
                break

    return segment


def get_edge(a, b):
    if a < b:
        return a,b
    else:
        return b,a

def get_edge_str(a, b):
    a, b = get_edge(a,b)
    return '({:.3f}, {:.3f}, {:.3f}),({:.3f},{:.3f},{:.3f})'.format(a[0], a[1], a[2], b[0], b[1], b[2])

def split_graph_into_segments(graph):
    # Converts a graph into segments
    sources = set([n for n, deg in graph.degree if deg != 2])
    visited_edges = set()
    segments = []

    for source in sources:
        neighbors = graph[source]
        for neighbor in neighbors:
            start_edge = (source, neighbor)
            if get_edge(*start_edge) in visited_edges:
                continue

            segment = extract_segment(graph, start_edge)
            segments.append(segment)
            new_edges = [get_edge(*x) for x in zip(segment[:-1], segment[1:])]
            visited_edges = visited_edges.union(new_edges)

    return segments



def smooth_graph_nodes(node_list, deviation, min_branch_len = 0.0):
    """
    Attempts to smooth redundant nodes from a graph by drawing lines between non-adjacent nodes on the same branch
    and checking the amount of deviation between the nodes along the path and the drawn line

    :param node_list: A list of nodes to be smoothed sequentially, or a corresponding numpy array
    :param deviation: Once the average node-line distance exceeds this amount,
    :param min_branch_len: Requires that each cleaned-up segment exceed a certain length for consideration
    :return: A list of node tuples
    """

    if not len(node_list):
        return []

    if not isinstance(node_list, np.ndarray):
        node_list = np.array(node_list)

    return_list = [tuple(node_list[0])]
    last_node = node_list[0]
    active_root_node = node_list[0]

    accumulated_dist = 0
    accumulated_intermediate_nodes = []

    for node in node_list[1:]:
        accumulated_dist += get_node_dist(node, last_node)

        if accumulated_dist > min_branch_len and accumulated_intermediate_nodes:
            dists = get_point_line_distance(accumulated_intermediate_nodes, active_root_node, node)
            mean_dist = dists.mean()
            if mean_dist > deviation:

                return_list.append(tuple(last_node))
                active_root_node = last_node
                accumulated_dist = get_node_dist(node, last_node)
                accumulated_intermediate_nodes = []

        last_node = node
        accumulated_intermediate_nodes.append(node)

    else:
        # Reached last node in branch - Append it to node list
        return_list.append(tuple(last_node))

    return return_list

def redistribute_branch_nodes(node_list, min_branch_len):
    """
    Alternative to smooth_graph_nodes (will decide on one or the other). Simply converts a series of nodes to another
    series of evenly-spaced nodes.
    :param node_list:
    :param min_branch_len:
    :return:
    """

    if not len(node_list):
        return []

    if not isinstance(node_list, np.ndarray):
        node_list = np.array(node_list)

    return_list = []
    return_list.append(tuple(node_list[0]))

    cumul_dists = np.concatenate([[0], np.cumsum(np.linalg.norm(node_list[1:] - node_list[:-1], axis=1))])
    n_new = np.max([np.int(np.floor(cumul_dists[-1] / min_branch_len)), 1])

    distance_distrib = np.linspace(0, cumul_dists[-1], num=n_new + 1)[1:]
    for dist_marker in distance_distrib:
        end_index = np.argmin(dist_marker > cumul_dists)
        progress = (dist_marker - cumul_dists[end_index - 1]) / (cumul_dists[end_index] - cumul_dists[end_index - 1])

        start_pt = node_list[end_index - 1]
        end_pt = node_list[end_index]

        intermediate_node = start_pt + progress * (end_pt - start_pt)

        return_list.append(tuple(intermediate_node))

    return return_list






def create_edge_point_associations(graph, point_cloud_array, node_attribute=None, in_place=False):
    """
    Given a graph and a point cloud, assigns each edge a set of points in the point cloud
    :param graph: A NetworkX graph object
    :param point_cloud_array: a Nx3 Numpy array
    :param in_place: if True, sets the association attribute for each edge
    :return: A dictionary indexed by edge, with each value being a list of indexes from the numpy array.
        All point assignments are unique, though some points may not be assigned.
    """
    tree = KDTree(point_cloud_array)
    assignments = np.zeros(point_cloud_array.shape[0], dtype=np.int) - 1
    winning_dist = np.zeros(point_cloud_array.shape[0]) + np.inf

    edges = []

    for idx, edge in enumerate(graph.edges):
        edges.append(edge)

        start, end = edge
        if node_attribute is not None:
            start = graph.nodes[start][node_attribute]
            end = graph.nodes[end][node_attribute]


        start = np.array(start)
        end = np.array(end)
        midpoint = (start + end) / 2
        r = np.linalg.norm(start - end) / 2

        pt_indexes = np.array(tree.query_ball_point(midpoint, r))
        midpoint_dists = np.linalg.norm(point_cloud_array[pt_indexes] - midpoint, axis=1) / r

        subidx = midpoint_dists < winning_dist[pt_indexes]

        to_assign = pt_indexes[midpoint_dists < winning_dist[pt_indexes]]
        assignments[to_assign] = idx
        winning_dist[to_assign] = midpoint_dists[subidx]

    reindexer = assignments != -1
    indexes = np.arange(0, point_cloud_array.shape[0])[reindexer]
    assignments = assignments[reindexer]

    rez = {}
    for id, grouping in pd.Series(indexes).groupby(assignments):
        rez[edges[id]] = grouping.values

    if in_place:
        nx.set_edge_attributes(graph, rez, name='associations')

    return rez


def skeletonize(pc):
    graph = construct_mutual_k_neighbors_graph(pc, 15, 0.05)
    clean(graph, threshold=0.10)

    spanning_tree = nx.algorithms.tree.minimum_spanning_tree(graph)
    while True:
        rez = clean(spanning_tree, threshold=0.05)
        if not rez:
            break

    segments = split_graph_into_segments(spanning_tree)

    smoothed_segments = []

    for segment in segments:
        smoothed_segments.append(smooth_graph_nodes(segment, 0.015))

    vertex_pairs = []
    for segment in smoothed_segments:
        vertex_pairs.extend(zip(segment[:-1], segment[1:]))

    return vertex_pairs


def convert_skeleton_to_graph(segments):
    new_graph = nx.Graph()
    for segment in segments:
        new_graph.add_nodes_from(segment)
        new_graph.add_edges_from(zip(segment[:-1], segment[1:]))

    return new_graph

if __name__ == '__main__':

    import os
    from pypcd import pypcd as pypcd
    import sys
    file_name = os.path.join(os.path.expanduser('~'), 'data', 'point_clouds', 'bag_5', 'cloud_final.pcd')
    data = pypcd.point_cloud_from_path(file_name).pc_data
    pc = data.view((data.dtype[0], 3))
    pc = pc[(pc[:,1] < 1.25) & (pc[:,2] < 1.7)]

    keep_idx = np.random.choice(pc.shape[0], 15000, replace=False)
    pc_sub = pc[keep_idx]

    graph = construct_mutual_k_neighbors_graph(pc_sub, 15, 0.05)
    clean(graph, threshold=0.10)

    spanning_tree = nx.algorithms.tree.minimum_spanning_tree(graph)
    while True:
        rez = clean(spanning_tree, threshold=0.05)
        if not rez:
            break

    segments = split_graph_into_segments(spanning_tree)
    # Smooth out spanning tree
    smoothed_segments = []

    for segment in segments:
        smoothed_segments.append(smooth_graph_nodes(segment, 0.015))

    new_graph = nx.Graph()
    for segment in smoothed_segments:
        new_graph.add_nodes_from(segment)
        new_graph.add_edges_from(zip(segment[:-1], segment[1:]))

    if len(sys.argv) > 1:
        print('Processing radii')
        edge_radii = {}

        rez = create_edge_point_associations(new_graph, pc)
        # Use Cindy's cylinder-fitting code

        from CylinderCover import CylinderCover, Cylinder
        print('Fitting cylinders...')
        cyls = []
        for edge, point_indexes in rez.items():

            if len(point_indexes) < 4:
                print('\tDropped cylinder with {} pts'.format(len(cyl.pts)))
                continue

            cyl = Cylinder()
            cyl.set_fit_pts(point_indexes[0], point_indexes, pc)
            cyl.optimize_cyl(0.005, 0.10)

            edge_radii[edge] = {'radius': cyl.radius}
            cyls.append(cyl)

        nx.set_edge_attributes(new_graph, edge_radii)

    # This stuff is purely test, you should move it somewhere else later
    import mesh

    joints = {}
    for node in new_graph.nodes:
        neighbors = [n for n in new_graph[node]]
        radius = np.array([new_graph.edges[(node, neighbor)].get('radius', 0.01) for neighbor in neighbors]).mean()
        joints[node] = mesh.SphereJoint(np.array(node), 0.01, np.array(neighbors),
                                        [get_edge_str(node, n) for n in neighbors])

    for node, deg in new_graph.degree:
        if deg < 3:
            continue
        joints[node].divide_sphere()

    all_data = []
    for segment in split_graph_into_segments(new_graph):

        if len(new_graph[segment[0]]) < 3:
            segment = segment[::-1]
        if len(new_graph[segment[0]]) < 3:
            print('Temporarily skipping this segment...')
            continue

        sleeves = [mesh.Sleeve(joints[a], joints[b], get_edge_str(a, b)) for a, b in zip(segment[:-1], segment[1:])]
        for s in sleeves:
            s.generate_faces()
            all_data.append(s.convert_to_mesh())

    v, f = zip(*all_data)
    counter = 0
    for i, faces in enumerate(f):
        faces += counter
        counter += v[i].shape[0]

    v = np.concatenate(v)
    f = np.concatenate(f)

    output = {'v': v, 'f': f}
    import pickle

    with open('test_v_f.pickle', 'wb') as fh:
        pickle.dump(output, fh, protocol=2)

    print('Saving cylinders...')
    cover = CylinderCover()
    cover.cyls_fitted = cyls
    cover.my_pcd.pc_data = pc
    cover.my_pcd.min_pt = np.min(pc, axis=0)
    cover.my_pcd.max_pt = np.max(pc, axis=0)

    cover_file = 'data/test_cyl_cover.txt'
    with open(cover_file, 'w') as fh:
        cover.write(fh)

    print('Saving point cloud data...')
    pcd_file = 'data/TestPointCloud.pickle'
    cover.my_pcd.write(pcd_file)

    print('All done!')








