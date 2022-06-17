import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from matplotlib.collections import LineCollection
from collections import defaultdict

def plot_skeleton(tree_graph, superpoint_dict, ax=None):
    """
    Operates on the converted skeletons produced by convert_results.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    pts = np.array(list(superpoint_dict.values()))[:,[0,1]]
    pts[:,1] = -pts[:,1]
    ax.scatter(pts[:,0], pts[:,1])

    lines = defaultdict(list)
    for edge in tree_graph.edges:
        classification = tree_graph.edges[edge]['classification']
        lines[classification].append([pts[edge[0]], pts[edge[1]]])

    colors = {
        0: 'brown',
        1: 'salmon',
        2: 'royalblue',
        3: 'limegreen'
    }
    for classification, edges in lines.items():
        lc = LineCollection(edges, colors=colors.get(classification, 'grey'), linewidths=2)
        ax.add_collection(lc)

if __name__ == '__main__':
    src = '/home/main/data/skeleton_data/skeletonization_results_converted'
    files = [f for f in os.listdir(src) if f.endswith('.pickle')]

    for file in files[:10]:
        with open(os.path.join(src, file), 'rb') as fh:
            data = pickle.load(fh)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plot_skeleton(data['orig'], data['superpoints'], axes[0])
        axes[0].set_title('Original')
        plot_skeleton(data['fixed'], data['superpoints'], axes[1])
        axes[1].set_title('Fixed')

        plt.show()