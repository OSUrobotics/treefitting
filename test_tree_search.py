import networkx as nx
import pandas as pd
import numpy as np
from functools import reduce
from collections import defaultdict
from scipy.spatial import KDTree

"""
Step 1: Take your graph and process the classification probabilities for each edge 
        (trunk, support, vertical, false/other)
Step 2: Initialize an empty state to consist of the trunk node. Add all outward edges as potential actions.
Step 3: At any given state, select a random edge as a potential action.
        The selection probability weight is 1 - (chance of fake)/(chance of all viable options, incl fake)
Step 4: For the connected node, examine all the potential edges. Filter out any actions which are unviable given
        the current assignment (see constraints below).
        Also remove any edges which end in that node from the action pool.
Step 5: Repeat until some termination condition is met. E.g. all leaders have have terminated,
        or there are no more remaining actions.

Constraints:
- Grammar based - Cannot grow trunk from support/leader, cannot grow support from leader
- Total curvature constraints on any nodes classified as leaders (no more than, say, 90 degrees)

Idea: Maybe use Bayesian updating to update probabilities based on coverage? Mainly of falsity
Idea: Simple flow prediction scheme that can make long-distance connections
Idea: Maybe use evolutionary where the optimal is coverage and the "traits" are the probabilities in the probability table
      Randomly crossover table entries to each other
"""

class DirectedTree:

    COLS = ['p_trunk', 'p_support', 'p_leader', 'p_other']

    def __init__(self, graph, starting_node, weights=None):
        self.base_graph = graph
        self.assigned_graph = nx.DiGraph(nx.create_empty_copy(graph))
        self.action_queue = set()
        self.all_assignments = set()
        self.do_not_consider = set()
        self.probability_table = pd.DataFrame()
        self.probability_table_original = None
        self.assignments = None
        self.covered_points = set()

        self.initialize_edge_weights(weights)
        nx.set_node_attributes(self.assigned_graph, {starting_node: 0}, 'classification')
        self.add_actions(starting_node)

    def grow(self, deterministic=False, plot_each_step=False, pts=None):

        i=0
        null_assignments = []
        while self.action_queue:
            try:
                edge, classification = self.pick_action(commit=True, deterministic=deterministic)
                if classification == 3:
                    null_assignments.append(edge)
            except ZeroDivisionError:
                print('All remaining options have zero probability!')
                break
            if plot_each_step:
                self.plot(pts, file='outputs/{:03d}'.format(i), null_assignments=null_assignments)

            i += 1

    def get_coverage(self):

        return len(set().union(*[self.base_graph.edges[e].get('points', set()) for e in self.assigned_graph.edges]))


        raise NotImplementedError("FIX THIS TOMORROW!")
        return len(self.all_assignments)
        # return len(self.covered_points)


    def plot(self, pts=None, file=None, null_assignments=None):
        from test_skeletonization_data import plot_graph
        plot_graph(self.assigned_graph, 'classification', pts=pts, title='Assigned Tree', save_to=file, null_assignments=null_assignments)

    def initialize_edge_weights(self, weights=None):
        if weights is None:
            df_dict = {}
            for a, b in self.base_graph.edges:
                pt_a = self.base_graph.nodes[a]['point']
                pt_b = self.base_graph.nodes[b]['point']
                probs = self.base_graph.edges[a, b].get('class_probs', None)
                if probs is None:
                    weights = [0.5, 0.5, 0.5, 0.1]
                else:
                    weights = np.zeros(4)
                    weights[:3] = probs[:3] * probs[4]          # Is connected prob * respective class prob
                    weights[3] = probs[3]*probs[4] + probs[5]  # Is connected prob * side branch prob + not connected prob

                # Enforce hard cutoffs on leaders, supports
                diff = np.abs(pt_a - pt_b)
                angle = np.arctan2(diff[1], diff[0])
                if angle < np.pi / 4:   # Less than 45 degrees, cannot be a vertical leader
                    weights[2] = 0
                if angle > np.pi / 3:   # Greater than 60 degrees, cannot be a support
                    weights[1] = 0
                df_dict[(a, b)] = weights
                df_dict[(b, a)] = weights

            cols = ['p_trunk', 'p_support', 'p_leader', 'p_other']
            df = pd.DataFrame.from_dict(df_dict, orient='index')
            df.rename(columns=dict(zip(df.columns, cols)), inplace=True)
            df.index = pd.MultiIndex.from_tuples(df.index)
        else:
            df = weights.copy()
        self.probability_table = df
        self.probability_table_original = df.copy()
        self.assignments = pd.DataFrame(False, index=df.index, columns=df.columns)

    def add_edge(self, edge, edge_classification):

        start, end = edge
        self.action_queue.remove(edge)
        self.assignments.loc[edge, self.probability_table.columns[edge_classification]] = True

        # Case 1: We assign this as being False - simply remove this one from consideration
        if edge_classification == 3:
            self.do_not_consider.add(edge)

        # Case 2: We have an assignment to class C. Do the following:
        # Add the edge to the assigned graph
        # Update actions
        # Zero out any probabilities corresponding to now invalid assignments

        else:
            start_class = self.assigned_graph.nodes[start].get('classification', 0)
            assert edge_classification >= start_class
            self.assigned_graph.add_edge(start, end, classification=edge_classification)
            nx.set_node_attributes(self.assigned_graph, {end: edge_classification}, 'classification')
            new_actions = self.add_actions(end)

            # Update action set based on constraints
            # Include curvature constraint, grammar constraint
            # TODO: Also constraint to encourage not to move down?

            # Trunk constraint - Once we transition from a trunk to a non-trunk, no more trunks may be assigned.
            # Also, the trunk may only have at most two neighbors classified as a support (cat 1)
            if start_class == 0:

                if edge_classification != start_class:
                    self.probability_table['p_trunk'] = 0

                    support_successors = [n for n in self.assigned_graph.successors(start)
                                          if self.assigned_graph.nodes[n]['classification'] == 1]
                    if len(support_successors) == 2:
                        self.probability_table.loc[self.get_edges(start), 'p_support'] = 0
                else:
                    self.remove_actions(start)
            else:
                # Non-branching constraint - For supports/leaders, cannot have previous nodes "branch off"
                # with same category/lesser category
                if start_class == edge_classification:
                    edges = self.get_edges(start)
                    to_null = ['p_support'] if edge_classification == 1 else ['p_support', 'p_leader']
                    self.probability_table.loc[edges, to_null] = 0.0

            # Topology constraint: All future assignments must be of a "higher" category
            # Here only applies to leaders, in which case support probability is nulled out
            if edge_classification == 2:
                self.probability_table.loc[list(new_actions), 'p_support'] = 0.0

            # Curvature constraint on leaders
            if start_class == 2:
                self.compute_curvature(start, end, include_base=True, commit=True)



    def pick_action(self, commit=True, deterministic=False):
        subset = self.probability_table.reindex(list(self.action_queue))
        selection_chance = subset.sum(axis=1)
        if deterministic:
            to_pick = np.argmax(selection_chance.values)
        else:
            total_weight = selection_chance.sum()
            if not total_weight:
                raise ZeroDivisionError
            selection_chance = selection_chance / total_weight
            to_pick = np.random.choice(len(selection_chance), p=selection_chance.values)
        edge = selection_chance.index[to_pick]
        if deterministic:
            chosen_class = np.argmax(subset.loc[edge].values)
        else:
            p_s = subset.loc[edge] / subset.loc[edge].sum()
            chosen_class = np.random.choice(len(self.COLS), p=p_s.values)


        if commit:
            self.add_edge(edge, chosen_class)
        return edge, chosen_class


    def add_actions(self, node):
        self.all_assignments.add(node)

        neighbors = set(self.base_graph[node])
        to_consider = neighbors.difference(self.all_assignments)

        # Remove all possible actions leading to this action, but add all actions leading FROM this action
        # (So long as they are not part of the do-not-consider list)
        self.action_queue.difference_update([(neighbor, node) for neighbor in neighbors])
        new_actions = set([(node, neighbor) for neighbor in to_consider]).difference(self.do_not_consider)

        # If dealing with a leader, impose curvature constraint
        curvature_violations = set()
        if self.assigned_graph.nodes[node]['classification'] == 2:
            for _, neighbor in new_actions:
                if self.compute_curvature(node, neighbor, include_base=True) >= np.pi/2:
                    curvature_violations.add((node, neighbor))

        self.action_queue.update(new_actions.difference(curvature_violations))

        return new_actions

    def get_edges(self, node, as_set=False):
        neighbors = [(node, neighbor) for neighbor in self.base_graph[node]]
        if as_set:
            neighbors = set(neighbors)
        return neighbors

    def remove_actions(self, node):
        self.action_queue.difference_update(self.get_edges(node))


    def compute_curvature(self, node, neighbor, include_base=True, commit=False):

        base = self.assigned_graph.nodes[node].get('curvature', 0) if include_base else 0
        prev = next(self.assigned_graph.predecessors(node))

        angle = base + three_point_angle(*[self.base_graph.nodes[n]['point'] for n in [prev, node, neighbor]])
        if commit:
            nx.set_node_attributes(self.assigned_graph, {neighbor: angle}, 'curvature')

        return angle

    def get_probability_update(self, weight):
        # Updates the weights of the chosen assignments
        # Weight can be from -1 to 1
        # -1 reflects that the chosen assignments were bad, with -1 setting all weights to 0
        # +1 reflects that the chosen assignments were good, with +1 setting all weights to 1
        p = self.probability_table_original.copy()
        subsetted = p[self.assignments]
        if weight > 0:
            new = subsetted + (1-subsetted) * weight
        else:
            new = subsetted * (1-abs(weight))
        p[self.assignments] = new
        return p


class EvolutionaryManager:
    def __init__(self, graph, min_node, population_size=5, mutation_prob=0.00, crossover_prob=0.3,
                 mutation_proportion=0.1, weight_magnitude=0.1):
        """
        Manages the trees for the evolutionary algorithm.
        :param graph: The base graph
        :param population_size: How many samples should be kept around at each generation.
        :param mutation_prob: The probability of a sample being derived from mutation.
        :param crossover_prob: The probability of a sample being derived from a random crossover.
        :param mutation_proportion: The proportion of table entries to be randomly assigned when mutating.
        :param weight_magnitude: The strength at which an assignment probability is shifted towards 0 or 1
        """

        self.orig_graph = graph
        self.orig_node = min_node

        self.pool = [DirectedTree(graph, min_node) for _ in range(population_size)]
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.mutation_proportion = mutation_proportion
        self.weight_magnitude = weight_magnitude

    def mutate_weights(self, df):
        zeroed = df == 0
        to_mutate = pd.DataFrame(np.random.uniform(size=df.shape) < self.mutation_proportion, columns=df.columns, index=df.index)
        df[to_mutate] = np.random.uniform(size=to_mutate.shape)
        df[zeroed] = 0
        return df

    def crossover_weights(self, df1, df2):
        weights = np.random.uniform(size=df1.shape)
        return df1 * weights + df2 * (1-weights)

    def iterate(self):
        objectives = []
        for tree in self.pool:
            tree.grow()
            objectives.append(tree.get_coverage())

        ranks = pd.Series(objectives).rank()
        selection_prob = ranks / ranks.sum()
        update_weights = ranks - ranks.mean()
        update_weights = update_weights / update_weights.abs().max() * self.weight_magnitude

        # Prepare the next generation
        new_pool = []
        probs = np.random.uniform(size=len(self.pool))

        for p in probs:
            if p < self.mutation_prob:
                # Pick one at random to mutate
                chosen = np.random.choice(len(self.pool))
                probs = self.pool[chosen].get_probability_update(update_weights[chosen])
                probs = self.mutate_weights(probs)
            elif p < self.mutation_prob + self.crossover_prob:
                chosen_a, chosen_b = np.random.choice(len(self.pool), 2, replace=False, p=selection_prob)
                probs_a = self.pool[chosen_a].get_probability_update(update_weights[chosen_a])
                probs_b = self.pool[chosen_b].get_probability_update(update_weights[chosen_b])
                probs = self.crossover_weights(probs_a, probs_b)
            else:
                chosen = np.random.choice(len(self.pool), p=selection_prob)
                probs = self.pool[chosen].get_probability_update(update_weights[chosen])

            new_pool.append(DirectedTree(self.orig_graph, self.orig_node, weights=probs))

        best_i = np.argmax(objectives)
        best_objective = objectives[best_i]
        best_orig_tree = self.pool[best_i]

        self.pool = new_pool

        return best_objective, best_orig_tree


class ActionAttributionManager:
    def __init__(self, graph, min_node, max_shift=0.5, tries_per_iter=5, keep_best=True):
        # Idea: Keep the best tree copy around and have that influence the selection each round?
        self.tries_per_iter = tries_per_iter
        self.keep_best = keep_best

        self.graph = graph
        self.min_node = min_node
        self.max_shift = max_shift

        self.current_weights = None
        self.current_best_assignments = None
        self.current_best_objective = -1

    def iterate(self):

        trees = [DirectedTree(graph, min_node, weights = self.current_weights) for _ in range(self.tries_per_iter)]
        objectives = []
        assignments = []
        for tree in trees:
            tree.grow()
            objectives.append(tree.get_coverage())
            assignments.append(tree.assignments)



        best_obj_i = np.argmax(objectives)
        best_obj = objectives[best_obj_i]

        if best_obj > self.current_best_objective:
            self.current_best_objective = best_obj
            self.current_best_assignments = assignments[best_obj_i]
        elif self.keep_best and self.current_best_assignments is not None:
            objectives.append(self.current_best_objective)
            assignments.append(self.current_best_assignments)

        n = len(assignments)
        assignment_weights = (pd.Series(objectives).rank() - (n+1) / 2) * get_ranking_weight(n) * self.max_shift
        # Sum the weighted assignment values together
        update_weight = pd.DataFrame(0, index=assignments[0].index, columns=assignments[0].columns)
        for wgt, assignment in zip(assignment_weights, assignments):
            update_weight = update_weight + wgt * assignment

        if self.current_weights is None:
            self.current_weights = trees[0].probability_table_original
        new_weights = self.current_weights.copy()
        new_weights[update_weight > 0] = self.current_weights + (1 - self.current_weights) * update_weight
        new_weights[update_weight < 0] = self.current_weights * (1 - update_weight.abs())

        self.current_weights = new_weights
        return best_obj, trees[best_obj_i]


        # p = self.probability_table_original.copy()
        # subsetted = p[self.assignments]
        # if weight > 0:
        #     new = subsetted + (1-subsetted) * weight
        # else:
        #     new = subsetted * (1-abs(weight))
        # p[self.assignments] = new
        # return p


def get_point_segment_distance(point_or_points, start, end):
    is_1d = False
    if len(point_or_points.shape) == 1:
        is_1d = True
        point_or_points = point_or_points.reshape((1,-1))

    diff = end - start
    segment_magnitude = np.linalg.norm(diff)
    diff_n = diff / segment_magnitude

    frame_pts = point_or_points - start
    linear_comp = (frame_pts).dot(diff_n)
    dist_comp = np.linalg.norm(frame_pts - linear_comp.reshape(-1,1) * diff_n.reshape(1,-1), axis=1)

    # Adjust the dist components so that anything "outside" of the segment range is normalized by the extent to which
    # it's outside

    before_start = linear_comp < 0
    past_end = linear_comp > segment_magnitude
    linear_comp[past_end] -= segment_magnitude
    outside = before_start | past_end
    dist_comp[outside] = np.sqrt(linear_comp[outside]**2 + dist_comp[outside]**2)

    if is_1d:
        dist_comp = dist_comp.reshape(-1)

    return dist_comp

def assign_points_to_edges(graph, pts, max_threshold=0.2, visualize=False):
    """
    Takes a graph and a set of points, and tries to assign each point to at most one edge to which it is closest.
    :param graph:
    :param pts:
    :param max_threshold:
    :return:
    """

    nodes = list(graph.nodes)
    all_edges = list(graph.edges)
    pt_assignments = np.zeros(pts.shape[0], dtype=np.int64)
    pt_distances = np.ones(pts.shape[0]) * np.inf

    point_index_cache = {n: graph.nodes[n].get('points', None) for n in nodes}
    kdtree = None



    for i, edge in enumerate(graph.edges):
        candidate_idx = set()
        for n in edge:
            pt_indexes = point_index_cache[n]
            if pt_indexes is None:
                if kdtree is None:
                    kdtree = KDTree(pts, leafsize=20)
                pt_indexes = kdtree.query_ball_point(graph.nodes[n]['point'], max_threshold)
                point_index_cache[n] = pt_indexes
            candidate_idx.update(pt_indexes)

        candidate_idx = np.array(list(candidate_idx))
        candidate_pts = pts[candidate_idx]
        dists = get_point_segment_distance(candidate_pts, graph.nodes[edge[0]]['point'], graph.nodes[edge[1]]['point'])
        beaten = (dists < pt_distances[candidate_idx]) & (dists < max_threshold)

        to_update = candidate_idx[beaten]
        pt_assignments[to_update] = i
        pt_distances[to_update] = dists[beaten]

    update_dict = defaultdict(set)
    for pt_idx, edge_idx in enumerate(pt_assignments):
        if pt_distances[pt_idx] < np.inf:
            update_dict[all_edges[edge_idx]].add(pt_idx)

    nx.set_edge_attributes(graph, update_dict, 'points')

    if visualize:
        import matplotlib.pyplot as plt
        for edge, pt_indexes in update_dict.items():
            a = graph.nodes[edge[0]]['point']
            b = graph.nodes[edge[1]]['point']
            plt.plot([a[0], b[0]], [a[1], b[1]])
            subset = pts[list(pt_indexes)]
            plt.scatter(subset[:,0], subset[:,1])
        plt.show()

def three_point_angle(x, y, z):
    # Represents the amount of "turning" you have to do.
    # (This is why a and b are flipped.)
    a = y-x
    b = z-y
    return np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def memoize(f):
    cache = {}
    def func(*args):
        if args in cache:
            return cache[args]
        rez = f(*args)
        cache[args] = rez
        return rez
    return func

@memoize
def get_ranking_weight(n):
    # Gets the scaling factor so that when ranking from 1, 2, ..., n,
    # and then demeaning, scaling by this factor will cause the positive elements to sum to 1.
    ranks = np.arange(1, n+1)
    demeaned = ranks - (n+1)/2
    return 1 / demeaned[demeaned > 0].sum()



if __name__ == '__main__':
    from test_skeletonization_data import generate_points_and_graph

    graph, pts = generate_points_and_graph(classify=False, map_to_points=True)
    print('Assigning points...')
    assign_points_to_edges(graph, pts, 0.15, visualize=False)
    print('Done assigning points to edges!')
    min_node = min(graph.nodes, key=lambda k: graph.nodes[k]['point'][1])

    # manager = EvolutionaryManager(graph, min_node)
    manager = ActionAttributionManager(graph, min_node, max_shift=0.75, tries_per_iter=10)
    # base_tree = DirectedTree(graph, min_node)
    # base_tree.grow(deterministic=True, plot_each_step=True, pts=pts)
    # base_tree.plot(pts)

    print('Starting evolution')
    best_obj = 0
    best_obj_ct = 0
    best_tree = None
    for i in range(50):
        print(i)
        obj, tree = manager.iterate()
        if obj > best_obj:
            best_obj = obj
            tree.plot(pts)
            best_obj_ct += 1
            best_tree = tree
    print('Ending evolution')
    new_tree = DirectedTree(graph, min_node, weights=best_tree.probability_table_original)
    new_tree.grow(deterministic=True)
    new_tree.plot(pts)



