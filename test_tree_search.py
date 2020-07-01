import networkx as nx
import pandas as pd
import numpy as np

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

def three_point_angle(x, y, z):
    # Represents the amount of "turning" you have to do.
    # (This is why a and b are flipped.)
    a = y-x
    b = z-y
    return np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == '__main__':
    from test_skeletonization_data import generate_points_and_graph

    graph, pts = generate_points_and_graph(classify=True, map_to_points=True)
    min_node = min(graph.nodes, key=lambda k: graph.nodes[k]['point'][1])

    manager = EvolutionaryManager(graph, min_node)
    # base_tree = DirectedTree(graph, min_node)
    # base_tree.grow(deterministic=True, plot_each_step=True, pts=pts)
    # base_tree.plot(pts)

    print('Starting evolution')
    best_obj = 0
    best_obj_ct = 0
    for i in range(101):
        print(i)
        obj, tree = manager.iterate()
        if obj > best_obj:
            best_obj = obj
            tree.plot(pts, file='outputs/best_{:03d}.png'.format(best_obj_ct))
            best_obj_ct += 1
    print('Ending evolution')



