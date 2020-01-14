#!/usr/bin/env python3


from pypcd import pypcd as pypcd

import numpy as np
import pickle
from Cylinder import Cylinder
from test_pts import Best_pts, Bad_pts
from list_read_write import ReadWrite
from bitarray import bitarray


class MyPointCloud(ReadWrite):
    def __init__(self):
        super(MyPointCloud, self). __init__("MYPOINTCLOUD")
        self.offsets = []
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    self.offsets.append([ix, iy, iz])

        # Set in load point cloud
        self.pc_data = np.zeros([1,3])
        self.min_pt = [1e30, 1e30, 1e30]
        self.max_pt = [-1e30, -1e30, -1e30]

        # Filled in by creating the bins
        self.radius_neighbor = 0.01  # Defines neighbor radius
        self.div = 0.005             # Defines width of bin
        self.start_xyz = [0.0, 0.0, 0.0]  # bottom left corner of box containing points
        self.n_bins_xyz = [1, 1, 1]       # Number of bins in each direction
        self.bin_offset = [1, 1, 1]       # for mapping bin xi,yi,zi to unique number

        # For mapping bin xi,yi,zi to single indes
        self.bin_ipts = []  # xi,yi,zi for each point
        self.bin_ids = []   # bin index for each point
        self.bin_list = {}  # bins, key is bin index, list of points in bin

        # Filled in by find_neighbors
        self.neighbors = []  # List of lists of neighbors for each point

    def _bin_index_pt(self, p):
        """
        Map the point into a bin
        :param p: 3D point
        :return: xi, yi, zi index of bin
        """
        ipt = [0, 0, 0]
        for i in range(0, 3):
            fx = (p[i] - self.start_xyz[i]) / self.div
            ipt[i] = int(np.floor(fx))
        return ipt

    def _bin_index_map(self, ipt):
        """
        Map the bin index into a single index
        :param ipt: xi, yi, zi index of bin
        :return: zi + yi (size x) + ix (size y and z)
        """
        return ipt[2] + ipt[1] * self.bin_offset[0] + ipt[0] * self.bin_offset[1]

    def _bin_center(self, index):
        """
        The x,y,z point in real space that is the bin center
        :param index: bin index
        :return: x,y,z
        """
        zi = index % self.n_bins_xyz[2]
        xy_index = int((index - zi) / self.n_bins_xyz[2])
        yi = xy_index % self.n_bins_xyz[1]
        x_index = xy_index - yi
        xi = int(x_index / self.n_bins_xyz[1])
        ipt = [xi, yi, zi]

        pt = []
        for i in range(0, 3):
            pt.append(self.start_xyz[i] + (ipt[i] + 0.5) * self.div)

        #  check_ipt = self._bin_index_pt(pt)
        #  check_index = self._bin_index_map(ipt)
        return np.array(pt)

    @staticmethod
    def dist(p, q):
        v = [(q[i] - p[i]) * (q[i] - p[i]) for i in range(0, 3)]
        return np.sqrt(sum(v))

    @staticmethod
    def dist_norm(p, q, pn, qn):
        dist_pt = MyPointCloud.dist(p, q)
        norm_align = np.dot(pn, qn)
        if norm_align < 0:
            return 2
        return dist_pt + 1 - norm_align

    def find_neighbors(self):
        """
        Search the neighboring bins for points within the given radius
        :return: list of ids of neighbors
        """
        self.neighbors = []

        # For each point, get the points in the bins around it
        avg_n = 0
        for i, p in enumerate(self.pc_data):
            ipt = self.bin_ipts[i]
            qs = []
            for o in self.offsets:
                ipt_search = [ipt[i] + o[i] for i in range(0, 3)]
                i_bin = self._bin_index_map(ipt_search)
                if i_bin in self.bin_list:
                    qs.extend(self.bin_list[i_bin])

            ns = []
            for qi in qs:
                if i != qi:
                    dist_q = self.dist(p, self.pc_data[qi])
                    if dist_q < self.radius_neighbor:
                        ns.append((qi, dist_q))
            self.neighbors.append(ns)
            avg_n += len(ns)
        return avg_n / len(self.neighbors)

    def find_connected(self, pi_start, in_radius=0.05):
        """
        Find all the points that are within the radius AND connected (breadth first search)
        :param pi_start: Point index to start at
        :param in_radius: Maximum Euclidean distance allowed
        :return: Dictionary of neighbors containing distances to neighbors
        """
        visited = {}
        to_visit = {pi_start: 0.0}
        ret_list = {pi_start: 0.0}
        p = self.pc_data[pi_start]
        while to_visit:
            # Get next item off the list
            (v, d) = to_visit.popitem()
            visited[v] = True
            # Check its neighbors
            for (q, dn) in self.neighbors[v]:
                if not (q in visited) and not (q in to_visit):
                    d_new = self.dist(p, self.pc_data[q])
                    # if close enough, add
                    if d_new < in_radius:
                        to_visit[q] = d_new
                        ret_list[q] = d_new
                    else:
                        visited[q] = True  # no sense in looking at this again
        return ret_list

    def _reorder_pts_in_bins(self):
        """
        Put the point that is closest to the center of the bin first in the list
        :return: None
        """
        for k, b in self.bin_list.items():
            pt_center = self._bin_center(k)
            dist_to_center = []
            for p_id in b:
                dist_to_center.append((p_id, MyPointCloud.dist(pt_center, self.pc_data[p_id])))
            dist_to_center.sort(key=lambda list_item: list_item[1])

            for i in range(0, len(b)):
                b[i] = dist_to_center[i][0]

    def create_bins(self, in_smallest_branch_width=0.01):
        """
        Make bins and put points in the bins. Aim for bins that are 1/3 radius of smallest branch width
        :param in_smallest_branch_width: Smallest branch width
        :return:
        """
        max_width = 0
        for i in range(0, 3):
            max_width = max(max_width, self.max_pt[i] - self.min_pt[i])
        self.radius_neighbor = 2.0 * in_smallest_branch_width / 3.0
        self.div = self.radius_neighbor / 2.0
        # pad by one row of bins
        self.start_xyz = [self.min_pt[i] - self.div - 0.0001 for i in range(0, 3)]
        self.n_bins_xyz = [int(np.ceil((self.max_pt[i] + self.div - self.start_xyz[i]) / self.div)) for i in range(0, 3)]
        # For mapping bin xi,yi,zi to single indes
        self.bin_offset = [self.n_bins_xyz[2], self.n_bins_xyz[2] * self.n_bins_xyz[1]]

        # Put each point in its bin
        self.bin_ipts = []
        self.bin_ids = []
        self.bin_list = {}
        min_ipt = [1000, 1000, 1000]
        max_ipt = [0, 0, 0]

        for i, p in enumerate(self.pc_data):
            ipt = self._bin_index_pt(p)
            for j in range(0, 3):
                min_ipt[j] = min(min_ipt[j], ipt[j])
                max_ipt[j] = max(max_ipt[j], ipt[j])
            index = self._bin_index_map(ipt)
            self.bin_ipts.append(ipt)
            self.bin_ids.append(index)
            try:
                list_in_bin = self.bin_list[index]
                list_in_bin.append(i)
                self.bin_list[index] = list_in_bin
            except KeyError:
                self.bin_list[index] = [i]

        # Stats on bin occupancy
        n_pts_in_bin_avg = 0
        n_count = 0
        max_count = 0

        bins_as_bit_arrays = {}
        for k, pts_in_bin in self.bin_list.items():
            n_count += 1
            n_pts_in_bin_avg += len(pts_in_bin)
            max_count = max(max_count, len(pts_in_bin))

            """
            #  bin_bit_array = len(my_pcd.pc_data) * bitarray('0')
            bin_bit_array = []
            for i in pts_in_bin:
                bin_bit_array.append(i)
                
            #  bins_as_bit_arrays[k] = bin_bit_array
            """

        for i in range(0, 3):
            if min_ipt[i] == 0 or max_ipt[i] > self.n_bins_xyz[i] - 1:
                raise ValueError("No padding")

        self._reorder_pts_in_bins()
        print("Count {0} avg {1} total {2} max {3}".format(n_count, n_pts_in_bin_avg / n_count, len(self.bin_ids), max_count))

        return self.bin_list

    def write(self, file_name):
        with open(file_name, 'wb') as fid:
            for_dump = list(self.bin_list.items())
            self.bin_list = for_dump
            pickle.dump(self, fid)
        self.fix_bin_list()

    def fix_bin_list(self):
        bin_list = {}
        for k in self.bin_list:
            bin_list[k[0]] = k[1]
        self.bin_list = bin_list
        return self

    def load_point_cloud(self, file_name):
        pcd_data = pypcd.point_cloud_from_path(file_name)
        self.pc_data = []
        for p in pcd_data.pc_data:
            if p[1] > 1.25 or p[2] > 1.7:
                continue
            self.pc_data.append(p)

        self.pc_data = np.array(self.pc_data)

        #  Find bounding box
        self.min_pt = [1e30, 1e30, 1e30]
        self.max_pt = [-1e30, -1e30, -1e30]
        for p in self.pc_data:
            for i in range(0, 3):
                self.min_pt[i] = min(self.min_pt[i], p[i])
                self.max_pt[i] = max(self.max_pt[i], p[i])


def load(in_fname):
    with open(in_fname, 'rb') as fid:
        my_pcd_load: MyPointCloud = pickle.load(fid)
    my_pcd_load.fix_bin_list()
    return my_pcd_load


if __name__ == '__main__':
    b_read = False

    name_file = "point_clouds/bag_0/cloud_final.pcd"
    #  name_file = "final_fused.pcd"
    fname_read = "data/MyPointCloud.pickle"
    if b_read:
        my_pcd = load("data/MyPointCloud.pickle")

        cyl = Cylinder()
        cyl_pts = Best_pts()
        cyl_pts.update(Bad_pts())

        for pt_id, label in cyl_pts.items():
            radius = 0.06
            if label.startswith("Thin"):
                radius = 0.05
            elif label.startswith("Trunk"):
                radius = 0.1

            pt_ids = my_pcd.find_connected(pt_id, radius)
            cyl.set_fit_pts(pt_id, pt_ids.keys(), my_pcd.pc_data)
            fname = "data/cyl_{0}.txt".format(pt_id)
            with open(fname, "w") as f:
                cyl.write(f, write_pts=True)

            print("Type {0}".format(label))
            print(cyl)
    else:
        my_pcd = MyPointCloud()
        my_pcd.load_point_cloud("data/" + name_file)
        smallest_branch_width_apple = 0.06
        smallest_branch_width_cherry = 0.04
        print("Creating bins: ", end="")
        my_pcd.create_bins(2 * smallest_branch_width_cherry / 4.0)
        print("{0}".format(my_pcd.div))
        print("Point cloud width/height: ", end=" ")
        vec = np.array(my_pcd.max_pt) - np.array(my_pcd.min_pt)
        print(vec)
        print("Expected bin width {0:0.4f}".format(np.linalg.norm(vec) / 500))
        print("Finding neighbors")
        n_avg = my_pcd.find_neighbors()
        print("Avg neigh {0}".format(n_avg))
        my_pcd.write(fname_read)


