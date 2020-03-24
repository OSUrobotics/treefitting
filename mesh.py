import numpy as np
from collections import defaultdict
from itertools import product
from functools import partial

try:
    from math import isclose
except ImportError:
    def isclose(a, b, abs_tol=0.000001):
        return abs(a-b) < abs_tol



class SphereJoint(object):

    def __init__(self, center, radius, neighbor_points, neighbor_ids=None):
        self.center = center
        self.radius = radius
        self.neighbors = neighbor_points
        self.subdivisions = {}
        self.id_neighbor_map = {}
        if neighbor_ids is not None:
            self.id_neighbor_map = dict(zip(neighbor_ids, range(self.neighbors.shape[0])))
        self.propagation_frame = {}

        self.antipodes = None

        # Some bookkeeping for computing geodesics
        self.tf = None
        self.tf_inv = None

    def divide_sphere(self):

        self.compute_triangular_normalized_antipodes()
        self.compute_sphere_coordinates()
        self.subdivisions = {}


        angle_dividers = np.array([self.get_sphere_coordinate(self.neighbors[n])[0] for n in range(3)])
        ordered = np.argsort(angle_dividers)    # Maps [P0, P1, P2] to position in geodesic, angle list
        ordering_to_index_map = dict(zip(np.arange(3), ordered))

        # Compute the initial geodesics

        init_geodesic = GeodesicSplit()
        for i, pt in enumerate(self.neighbors[:3]):
            angle, azimuth = self.get_sphere_coordinate(pt)
            init_geodesic.add_point(angle, i)

        unassigned = list(range(3, self.neighbors.shape[0]))

        # This is a bit hardcoded but for the purpose of moving on, zzz
        # Assign all angular splits
        # Then for each angular split, add the horizontal splits

        while True:
            geodesics = init_geodesic.geodesic_intervals()
            for assoc_idx in geodesics:
                interval = geodesics[assoc_idx]
                assoc_pt = self.project_point(self.neighbors[assoc_idx])
                assoc_sphere = self.get_sphere_coordinate(self.neighbors[assoc_idx])

                to_remove = None
                for pt_idx in unassigned:
                    pt = self.project_point(self.neighbors[pt_idx])
                    angle, azimuth = self.get_sphere_coordinate(pt)
                    if not is_wrapped_between(angle, interval):
                        continue

                    diff_vec = assoc_pt - pt
                    diag_vec_1 = self.antipodes[1] - self.antipodes[0]
                    azi_mid = assoc_sphere[1]
                    diag_vec_2 = self.get_world_coordinate((interval[1], azi_mid)) - \
                                 self.get_world_coordinate((interval[0], azi_mid))
                    angle_1 = np.arccos(
                        np.dot(diff_vec, diag_vec_1) / (np.linalg.norm(diff_vec) * np.linalg.norm(diag_vec_1)))
                    angle_2 = np.arccos(
                        np.dot(diff_vec, diag_vec_2) / (np.linalg.norm(diff_vec) * np.linalg.norm(diag_vec_2)))

                    ortho_diff_1 = abs(angle_1 - np.pi / 2)
                    ortho_diff_2 = abs(angle_2 - np.pi / 2)

                    split_on = (ortho_diff_1 > ortho_diff_2) * 1
                    # If we find something we want to split on angle, add it to the geodesics list and start from the
                    # beginning
                    if split_on == 0:
                        init_geodesic.add_point(angle, pt_idx)
                        to_remove = pt_idx
                        break

                if to_remove is not None:
                    unassigned.remove(to_remove)
                    break
            else:
                # We went through all the geodesics and couldn't find anything to split on the angle
                break

        # All other points should be split on the azimuth
        # For all unassigned points, figure out which geodesic interval they belong to, and add them to the corresponding
        azimuth_splits = defaultdict(partial(GeodesicSplit, start=-np.pi/2, end=-np.pi/2))
        angle_geodesics = init_geodesic.geodesic_intervals()
        for assoc_idx in angle_geodesics:
            interval = angle_geodesics[assoc_idx]
            assoc_pt = self.neighbors[assoc_idx]
            assoc_angle, assoc_azimuth = self.get_sphere_coordinate(assoc_pt)
            azimuth_splits[interval].add_point(assoc_azimuth, assoc_idx)


        for pt_idx in unassigned:
            pt = self.neighbors[pt_idx]
            angle, azimuth = self.get_sphere_coordinate(pt)
            for assoc_idx in angle_geodesics:
                interval = angle_geodesics[assoc_idx]
                if not is_wrapped_between(angle, interval):
                    continue
                azimuth_splits[interval].add_point(azimuth, pt_idx)
                break

        # Produce the subdivisions
        for angle_interval in azimuth_splits:
            angle_mid = (angle_interval[0] + angle_interval[1]) / 2
            azimuth_geodesics = azimuth_splits[angle_interval].geodesic_intervals()
            for assoc_pt in azimuth_geodesics:
                azimuth_interval = azimuth_geodesics[assoc_pt]
                sphere_coords = [(angle_mid, azimuth_interval[0]),
                                 (angle_interval[1], 0),
                                 (angle_mid, azimuth_interval[1]),
                                 (angle_interval[0], 0)]

                self.subdivisions[assoc_pt] = [self.get_world_coordinate(x) for x in sphere_coords]



    def retrieve_edge_quad(self, n_id):
        coords = self.subdivisions[self.id_neighbor_map.get(n_id, n_id)]
        if self.neighbors.shape[0] > 2:
            coords = np.roll(coords, 2, axis=0)

        return coords

    def compute_sphere_coordinates(self):

        antipode_vec = normalize(self.antipodes[1] - self.center)
        neighbor_vec = normalize(self.neighbors[0] - self.center)

        self.tf = np.identity(4)
        x = normalize(project_line_onto_plane(self.center, neighbor_vec, self.center, antipode_vec)[1])
        z = antipode_vec
        y = np.cross(z, x)
        self.tf[0:3, 0] = x
        self.tf[0:3, 1] = y
        self.tf[0:3, 2] = z
        self.tf[0:3, 3] = self.center
        self.tf_inv = np.linalg.inv(self.tf)

    def get_sphere_coordinate(self, neighbor):
        nx_s, ny_s, nz_s = self.tf_inv[:3,:3].dot(neighbor) + self.tf_inv[:3,3]
        angle = np.arctan2(ny_s, nx_s)
        azimuth = np.arctan2(nz_s, np.linalg.norm([nx_s, ny_s]))
        return np.array([angle, azimuth])

    def get_world_coordinate(self, sphere_coordinate):
        angle, azimuth = sphere_coordinate
        x = np.cos(angle)
        y = np.sin(angle)
        z = np.tan(azimuth)
        norm = np.linalg.norm([x, y, z])
        point_sphere = np.array([x, y, z]) * (self.radius / norm)
        point = self.tf[:3,:3].dot(point_sphere) + self.tf[:3, 3]
        return point

    def project_point(self, pt):
        diff_vec = pt - self.center
        return self.center + diff_vec * self.radius / np.linalg.norm(diff_vec)


    def get_antipodal_angle(self, neighbor):
        antipode_vec = normalize(self.antipodes[1] - self.antipodes[0])
        neighbor_vec = normalize(neighbor - self.antipodes[0])

        zero = np.zeros(3)



        neighbor_vec_proj = normalize(project_line_onto_plane(zero, neighbor_vec, zero, antipode_vec)[1])

        angle = arccos(self.reference.dot(neighbor_vec_proj))
        # Determine the sign of the angle
        z_angle = arccos(self.z_reference.dot(neighbor_vec_proj))
        if z_angle < np.pi/2:
            angle *= -1

        return angle % (2*np.pi)


    def compute_triangular_normalized_antipodes(self):

        if self.neighbors.shape[0] < 3:
            raise NotImplementedError('Antipode computation not defined for less than 3 neighbors')

        neighbors = self.neighbors[:3]

        # Compute where neighbor points intersect sphere
        norm_neighbors = neighbors - self.center
        norm_neighbors = self.radius * (norm_neighbors.T / np.linalg.norm(norm_neighbors, axis=1)).T

        circumcenter = get_circumcenter(norm_neighbors[0], norm_neighbors[1], norm_neighbors[2])

        # Compute plane formed by neighbors
        vec = np.cross(norm_neighbors[2]-norm_neighbors[0], norm_neighbors[1]-norm_neighbors[0])
        antipodes = get_sphere_line_intersection(self.radius, circumcenter, vec)

        self.antipodes = (antipodes[0] + self.center, antipodes[1] + self.center)

    def compute_propagation_frame(self, edge_id):
        frame = self.propagation_frame.get(edge_id)
        if frame is None:
            array_idx = self.id_neighbor_map.get(edge_id, edge_id)
            quad = np.roll(self.retrieve_edge_quad(edge_id), -2, axis=0)
            z_ax = normalize(self.neighbors[array_idx] - self.center)
            quad_edge = quad[0] - quad[1]
            x_ax = normalize(np.cross(z_ax, quad_edge))
            y_ax = np.cross(z_ax, x_ax)
            frame = np.array([x_ax, y_ax, z_ax]).T
            self.propagation_frame[edge_id] = frame

        return frame


class GeodesicSplit(object):

    def __init__(self, start=None, end=None):

        self.start = start
        self.end = end
        self.points = {}

    def add_point(self, pt, pt_id):
        self.points[pt_id] = pt

    def geodesic_intervals(self):
        # Return the geodesic intervals, with an index indicating the passed in point
        ids = sorted(self.points)
        pts = [self.points[pt_id] for pt_id in ids]
        pts = np.array(pts)
        if self.start is None:
            ordering = np.argsort(pts)
        else:
            ordering = np.argsort(pts + (pts >= self.start) * (-2 * np.pi))
        pts_ordered = pts[ordering]
        ids_ordered = np.array(ids)[ordering]

        geodesics = []
        intervals = {}

        if self.start is None:
            if len(ids) < 2:
                raise ValueError("Can't do a looped geodesic split with 0 or 1 points added!")

            for start, end in zip(pts_ordered, np.roll(pts_ordered, -1)):
                geodesics.append(midpoint_wrapped(start, end))

            for i, g_start, g_end in zip(ids_ordered, np.roll(geodesics, 1), geodesics):
                intervals[i] = (g_start, g_end)

        else:

            geodesics.append(self.start)

            for start, end in zip(pts_ordered[:-1], pts_ordered[1:]):
                geodesics.append(midpoint_wrapped(start, end))

            geodesics.append(self.end)
            for i, g_start, g_end in zip(ids_ordered, geodesics[:-1], geodesics[1:]):
                intervals[i] = (g_start, g_end)

        return intervals




class Sleeve(object):
    def __init__(self, joint_s, joint_e, edge_id):
        self.js = joint_s
        self.je = joint_e
        self.id = edge_id
        self.end_quad = None

    def generate_faces(self):

        end_degree = self.je.neighbors.shape[0]

        avg_vec = self.je.center - self.js.center
        if end_degree == 2:
            # Super hacky; get point corresponding to next neighbor
            idx = np.argmax(np.abs(self.je.neighbors - self.js.center).sum(axis=1))
            avg_vec = avg_vec + self.je.neighbors[idx] - self.je.center





        z_ax = normalize(avg_vec)

        prev_frame = self.js.compute_propagation_frame(self.id)
        x_prev = prev_frame[0].reshape((-1, 1))
        z_prev = prev_frame[2].reshape((-1, 1))

        # Compute the requirements for reflection
        n1 = self.je.center - self.js.center
        n1_2d = n1.reshape((-1, 1))
        R1 = np.identity(3) - 2*(n1_2d.dot(n1_2d.T)) / n1.dot(n1).item()
        n2 = self.je.center + z_ax - (self.je.center + R1.dot(z_prev.reshape(-1)))
        n2_2d = n2.reshape((-1, 1))
        R2 = np.identity(3) - 2*(n2_2d.dot(n2_2d.T)) / n2.dot(n2).item()

        x_ax = R2.dot(R1.dot(x_prev)).reshape(-1)
        y_ax = np.cross(z_ax, x_ax)

        r = self.je.radius
        vi = self.je.center
        next_quad = np.array([
            vi + r * x_ax - r * y_ax,
            vi + r * x_ax + r * y_ax,
            vi - r * x_ax + r * y_ax,
            vi - r * x_ax - r * y_ax,

        ])

        if self.je.neighbors.shape[0] > 2:
            existing_quad = self.je.retrieve_edge_quad(self.id)
            next_quad = align_vertices(existing_quad, next_quad)

        self.end_quad = next_quad
        self.js.compute_propagation_frame(self.id)


        # Propagate frame forward
        if end_degree == 2:
            for edge in self.je.id_neighbor_map:
                if edge != self.id:
                    self.je.propagation_frame[edge] = np.array([x_ax, y_ax, z_ax]).T
            self.je.subdivisions[0] = self.je.subdivisions[1] = next_quad

    def convert_to_mesh(self):
        prev_quad = np.array(self.js.retrieve_edge_quad(self.id))
        end_quad = self.end_quad
        # if self.js.neighbors.shape[0] > 2:      # Starting from a generated quad
        #     import pdb
        #     pdb.set_trace()
        #     prev_quad = align_vertices(prev_quad, end_quad, force_project=True)
        #
        # if self.je.neighbors.shape[0] > 2:
        #
        #
        #
        #     existing_quad = self.je.compute_propagation_frame(self.id)
        #     end_quad = align_vertices(existing_quad, end_quad)

        verts = np.concatenate([prev_quad, end_quad])
        faces = [
            [0, 1, 5], [0, 5, 4],
            [0, 3, 7], [0, 7, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
        ]

        if self.je.neighbors.shape[0] == 1:     # Branch end
            faces.extend([[4, 5, 7], [5, 7, 6]])

        faces = np.array(faces)

        return verts, faces


def process_skeleton(graph, default_radius = 0.01):

    from skeletonization import get_edge_str, split_graph_into_segments, convert_skeleton_to_graph
    import networkx as nx
    if not isinstance(graph, nx.Graph):
        graph = convert_skeleton_to_graph(graph)

    joints = {}
    for node in graph.nodes:
        neighbors = [n for n in graph[node]]
        radius = np.array([graph.edges[(node, neighbor)].get('radius', default_radius) for neighbor in neighbors]).mean()
        joints[node] = SphereJoint(np.array(node), radius, np.array(neighbors),
                                        [get_edge_str(node, n) for n in neighbors])

    for node, deg in graph.degree:
        if deg < 3:
            continue
        joints[node].divide_sphere()

    all_data = []
    for segment in split_graph_into_segments(graph):

        if len(graph[segment[0]]) < 3:
            segment = segment[::-1]
        if len(graph[segment[0]]) < 3:
            print('Temporarily skipping this segment...')
            continue

        sleeves = [Sleeve(joints[a], joints[b], get_edge_str(a, b)) for a, b in zip(segment[:-1], segment[1:])]
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
    return output







def project_line_onto_plane(line_start, line_vec, plane_pt, plane_vec):

    p1 = project_pt_onto_plane(line_start, plane_pt, plane_vec)
    p2 = project_pt_onto_plane(line_start + line_vec, plane_pt, plane_vec)
    return p1, p2-p1


def project_pt_onto_plane(pt, plane_pt, plane_vec):

    plane_vec = plane_vec / np.linalg.norm(plane_vec)

    vec = pt - plane_pt
    dist = vec.dot(plane_vec)
    return pt - dist.reshape((-1, 1)).dot(plane_vec.reshape(1,-1))

def normalize(v):
    return v / np.linalg.norm(v)

def arccos(x, edge_tolerance=0.00001):
    if isclose(abs(x), 1, abs_tol=edge_tolerance):
        x = 1.0 if x > 0 else -1.0
    return np.arccos(x)


def get_sphere_line_intersection(radius, line_point, line_vec, center=None):
    # Computes the intersection of a sphere (centered at 0 by default) and a line
    if center is None:
        center = np.array([0,0,0])

    line_point = line_point - center
    line_vec = line_vec / np.linalg.norm(line_vec)

    b = line_vec.dot(line_point)
    c = (line_point ** 2).sum() - radius ** 2

    discriminant = b ** 2 - c
    if discriminant < 0:
        return []
    else:
        d1 = -b + np.sqrt(discriminant)
        d2 = -b - np.sqrt(discriminant)
        return (line_point + center) + line_vec * d1, (line_point + center) + line_vec * d2


def get_circumcenter(a, b, c):
    # https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
    c_a = c - a
    b_a = b - a
    mag_sq_c_a = (c_a ** 2).sum()
    mag_sq_b_a = (b_a ** 2).sum()
    numerator = mag_sq_c_a * np.cross(np.cross(b_a, c_a), b_a) + mag_sq_b_a * np.cross(np.cross(c_a, b_a), c_a)
    denom = 2 * (np.cross(b_a, c_a) ** 2).sum()
    return a + numerator / denom

def align_vertices(to_permute, base, allow_reverse=True, force_project=False):



    roll_vals = [0, 1, 2, 3]
    reverse_vals = [1]
    if allow_reverse:
        reverse_vals.append(-1)

    orig_permute = to_permute
    if force_project:
        base_centroid = base.mean(axis=0)
        orig_centroid = to_permute.mean(axis=0)

        import pdb
        pdb.set_trace()

        to_permute = project_pt_onto_plane(orig_permute, base_centroid, orig_centroid-base_centroid)

    current_best = 1, 0
    dist = np.inf

    for order, r in product(reverse_vals, roll_vals):
        rotated = np.roll(to_permute, r, axis=0)
        current_dist = np.linalg.norm(base - rotated, axis=1).sum()
        if current_dist < dist:
            dist = current_dist
            current_best = order, r

    return np.roll(orig_permute[::current_best[0]], current_best[1], axis=0)

def midpoint_wrapped(a, b, abs_bound=np.pi):

    if a <= b:
        return (a+b)/2

    # If b > a, shift a over by 2*bound and take the average; then take average, check which side of bound it's on
    wrapped_mid = (a + 2*abs_bound + b) / 2
    if wrapped_mid > abs_bound:
        return wrapped_mid - 2 * abs_bound

    return wrapped_mid

def split_interval_between(a, b, interval):

    orig_a = a

    if a == b:
        raise ValueError("Can't divide interval when a and b are the same!")

    i0, i1 = interval

    # "Order" a and b on the interval
    if i0 <= i1:
        if a > b:
            a, b = b, a
    elif (a >= i0 and b >= i0) or (a <= i1 and b <= i1):
        if a > b:
            a, b = b, a
    elif b >= i0 and a <= i1:
        a, b = b, a
    else:
        raise ValueError('Your input values are not contained in the given interval!')

    mid = midpoint_wrapped(a, b)
    intervals = [(i0, mid), (mid, i1)]
    if a < orig_a < mid:
        flip = False
    else:
        flip = True

    return intervals, flip



def is_wrapped_between(a, interval):
    b0, b1 = interval
    if b0 <= b1:
        return b0 <= a <= b1
    return a > b0 or a <= b1

def create_mesh_from_pickle(file_path):
    import cPickle as pickle
    with open(file_path, 'rb') as fh:
        vf = pickle.load(fh)

    v = vf['v']
    f = vf['f']
    import pymesh
    mesh = pymesh.form_mesh(v, f)
    pymesh.save_mesh('test_tree.obj', mesh)


if __name__ == '__main__':

    c = np.array([0, 0, 0])
    b1a = np.array([1, 0, 0])
    b1b = np.array([2, 0, 0])
    b2 = np.array([-0.5, 0, 0.5])
    b3 = np.array([-0.5, 0, -0.5])
    b4 = np.array([-1, 0, 0])

    cj = SphereJoint(c, 0.05, np.array([b1a, b2, b3, b4]), ['c1', 'c2', 'c3', 'c4'])
    b1aj = SphereJoint(b1a, 0.05, np.array([c, b1b]), ['c1', '1ab'])
    b1bj = SphereJoint(b1b, 0.05, np.array([b1b]), ['1ab'])
    b2j = SphereJoint(b2, 0.05, np.array([c]), ['c2'])
    b3j = SphereJoint(b3, 0.05, np.array([c]), ['c3'])
    b4j = SphereJoint(b4, 0.05, np.array([c]), ['c4'])

    s1a = Sleeve(cj, b1aj, 'c1')
    s1b = Sleeve(b1aj, b1bj, '1ab')
    s2 = Sleeve(cj, b2j, 'c2')
    s3 = Sleeve(cj, b3j, 'c3')
    s4 = Sleeve(cj, b4j, 'c4')

    cj.divide_sphere()

    all_data = []
    for s in [s1a, s1b, s2, s3, s4]:
        s.generate_faces()
        all_data.append(s.convert_to_mesh())

    v, f = zip(*all_data)
    counter = 0
    for i, faces in enumerate(f):
        faces += counter
        counter += v[i].shape[0]

    v = np.concatenate(v)
    f = np.concatenate(f)

    import pymesh
    mesh = pymesh.form_mesh(v, f)
    pymesh.save_mesh('test_tree.obj', mesh)

    # for i in range(10):
    #     print('Test {}'.format(i))
    #     arr = np.random.normal(0, 10, (3,3))
    #     a = arr[0]
    #     b = arr[1]
    #     c = arr[2]
    #
    #     cent = get_circumcenter(a, b, c)
    #     print('Dists: {:.1f}, {:.1f}, {:.1f}'.format(*np.linalg.norm(arr-cent, axis=1)))
    #
    #
