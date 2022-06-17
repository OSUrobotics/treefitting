import pickle
import os


src = '/home/main/data/skeleton_data/skeletonization_results_final'
dst = '/home/main/data/skeleton_data/skeletonization_results_converted'

files = [f for f in os.listdir(src) if f.endswith('.pickle')]
for file in files:
    orig_file = os.path.join(src, file)
    new_file = os.path.join(dst, file)

    with open(orig_file, 'rb') as fh:
        data = pickle.load(fh)

    new_data = {}
    new_data['fixed'] = data['fixed_assignment']
    new_data['orig'] = data['original_assignment']
    new_data['points'] = data['tree'].points
    new_data['superpoints'] = {}

    superpoints = data['tree'].superpoint_graph
    for node in superpoints.nodes:
        new_data['superpoints'][node] = superpoints.nodes[node]['point']

    with open(new_file, 'wb') as fh:
        pickle.dump(new_data, fh)