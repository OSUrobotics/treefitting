#!/usr/bin/env python3

import pymesh
import numpy as np
import scipy.signal as signal
from list_read_write import ReadWrite

class MyPointCloud(ReadWrite):
    def __init__(self):
        super(MyPointCloud, self). __init__("MYPOINTCLOUD")
        self.offsets = []
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    self.offsets.append([ix, iy, iz])

        # Set in load point cloud
        self.points = np.zeros((0,3))
        self.min_pt = np.array([1e30, 1e30, 1e30])
        self.max_pt = np.array([-1e30, -1e30, -1e30])
        self.file_name = None

        # Filled in by creating the bins
        self.div = 0.005                  # Defines width of bin
        self.start_xyz = [0.0, 0.0, 0.0]  # bottom left corner of box containing points
        self.n_bins_xyz = [1, 1, 1]       # Number of bins in each direction
        self.bin_offset = [1, 1, 1]       # for mapping bin xi,yi,zi to unique number

        # For mapping bin xi,yi,zi to single indes
        self.bin_ids = []   # bin index for each point, can recover ix, iy, iz from _bin_ipt
        self.bin_list = {}  # bins, key is bin index, list of points in bin

    def __getitem__(self, i):
        return self.points[i]

    def __len__(self):
        return self.points.shape[0]

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

    def bin_ipt(self, index):
        """
        The x,y,z point in real space that is the bin center
        :param index: bin index
        :return: xi,yi,zi
        """
        zi = index % self.n_bins_xyz[2]
        xy_index = int((index - zi) / self.n_bins_xyz[2])
        yi = xy_index % self.n_bins_xyz[1]
        x_index = xy_index - yi
        xi = int(x_index / self.n_bins_xyz[1])
        return [xi, yi, zi]

    def _bin_center(self, index):
        """
        The x,y,z point in real space that is the bin center
        :param index: bin index
        :return: x,y,z
        """
        ipt = self.bin_ipt(index)

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

    def find_neighbor_bins(self, pi_index, radius_search):
        """
        Search the neighboring bins for points within the given radius
        :param: pi_index: index of point
        :param: radius_search: maximum radius to try
        :return: list of ids of neighbor bin ids within radius_search
        """
        p = self[pi_index]
        last_count = 0
        ret_list = [self.bin_ids[pi_index]]
        visited = {self.bin_ids[pi_index]: 0.0}
        while len(ret_list) > last_count:
            ipt = self.bin_ipt(ret_list[last_count])
            for o in self.offsets:
                ipt_search = [ipt[i] + o[i] for i in range(0, 3)]
                i_bin = self._bin_index_map(ipt_search)
                if i_bin in self.bin_list and i_bin not in visited:
                    q = self._bin_center(i_bin)
                    dist_to_bin = self.dist(p, q)
                    visited[i_bin] = dist_to_bin
                    if dist_to_bin < radius_search:
                        ret_list.append(i_bin)
            last_count += 1
        return ret_list

    def find_connected(self, pi_start, in_radius=0.05):
        """
        Find all the points that are within the radius AND connected (breadth first search)
        :param pi_start: Point index to start at
        :param in_radius: Maximum Euclidean distance allowed
        :return: Dictionary of neighbors containing distances to neighbors (id, pt, dist)
        """
        bin_id_list = self.find_neighbor_bins(pi_start, in_radius + self.div)
        p = self[pi_start]
        ret_list = []
        for i_bin in bin_id_list:
            for q_id in self.bin_list[i_bin]:
                q = self[q_id]
                d_pq = self.dist(p, q)
                if d_pq < in_radius:
                    ret_list.append((q_id, q, d_pq))
        return ret_list

    def _reorder_pts_in_bins(self):
        """
        Put the point that is closest to the center of the bin first in the list
        :return: None
        """
        for k, bl in self.bin_list.items():
            pt_center = self._bin_center(k)
            dist_to_center = []
            for p_id in bl:
                dist_to_center.append((p_id, MyPointCloud.dist(pt_center, self[p_id])))
            dist_to_center.sort(key=lambda list_item: list_item[1])

            for i in range(0, len(bl)):
                bl[i] = dist_to_center[i][0]

    def smallest_branch_width(self):
        return 3.0 * self.div

    def create_bins(self, in_smallest_branch_width=0.01):
        """
        Make bins and put points in the bins. Aim for bins that are 1/3 radius of smallest branch width
        :param in_smallest_branch_width: Smallest branch width
        :return: bin list
        """
        max_width = 0
        for i in range(0, 3):
            max_width = max(max_width, self.max_pt[i] - self.min_pt[i])
        self.div = in_smallest_branch_width / 3.0
        # pad by one row of bins
        self.start_xyz = [self.min_pt[i] - self.div - 0.0001 for i in range(0, 3)]
        self.n_bins_xyz = [int(np.ceil((self.max_pt[i] + self.div - self.start_xyz[i]) / self.div)) for i in range(0, 3)]
        # For mapping bin xi,yi,zi to single indes
        self.bin_offset = [self.n_bins_xyz[2], self.n_bins_xyz[2] * self.n_bins_xyz[1]]

        # Put each point in its bin
        self.bin_ids = []
        self.bin_list = {}
        min_ipt = [1000, 1000, 1000]
        max_ipt = [0, 0, 0]

        print("Finding bins n points: {0}, div {1:0.4f}".format(len(self), self.div))
        for i, p in enumerate(self.points):
            if i % 1000 == 0:
                if i % 10000 == 0:
                    print("{0} ".format(i), end='')

            ipt = self._bin_index_pt(p)
            for j in range(0, 3):
                min_ipt[j] = min(min_ipt[j], ipt[j])
                max_ipt[j] = max(max_ipt[j], ipt[j])
            index = self._bin_index_map(ipt)
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

        # bins_as_bit_arrays = {}
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

    def load_point_cloud(self, file_name=None):

        if isinstance(file_name, str):
            self.points = pymesh.load_mesh(file_name).vertices
            self.file_name = file_name

            # TODO Remove this dependency by pre-filtering .ply files
            import tree_model
            self.points = tree_model.preprocess_point_cloud(self.points, downsample=50000)

        else:   # Super hack!
            self.points = file_name

        #  Find bounding box
        self.min_pt = np.min(self.points, axis=0)
        self.max_pt = np.max(self.points, axis=0)

    def read(self, fid):
        self.check_header(fid)
        self.read_class_members(fid)
        self.load_point_cloud(self.file_name)

    def write(self, fid):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), MyPointCloud, ["points"])
        self.write_footer(fid)


if __name__ == '__main__':
    b_read = True

    my_pcd = MyPointCloud()

    name_file = "data/point_clouds/bag_3/cloud_final.ply"
    # name_file = "final_fused.pcd"
    fname_rw = "data/MyPointCloud.txt"

    if b_read:
        with open(fname_rw, "r") as fid_in:
            my_pcd.read(fid_in)

        print("Reading: n bins: {0}, bin width {1:0.4f}".format(len(my_pcd.bin_list), my_pcd.div))
        print("Point cloud width/height: ", end=" ")
        vec = np.array(my_pcd.max_pt) - np.array(my_pcd.min_pt)
        print(vec)
        print(my_pcd.min_pt)
        print(my_pcd.max_pt)
        print("Expected bin width {0:0.4f}".format(np.linalg.norm(vec) / 500))
        bin_count_bds = [0, 5, 15, 25, 45, 100, 1000000]
        bin_count = [0, 0, 0, 0, 0, 6]
        for _, b in my_pcd.bin_list.items():
            b_len = len(b)
            for bi, bd in enumerate(bin_count_bds[:-1]):
                if bd <= b_len < bin_count_bds[bi+1]:
                    bin_count[bi] += 1

        print("Bin counts:")
        for bi, bd in enumerate(bin_count_bds[:-1]):
            print("{0} - {1}, {2}".format(bd, bin_count_bds[bi+1], bin_count[bi]))
    else:
        my_pcd.load_point_cloud(name_file)
        smallest_branch_width_apple = 0.06
        smallest_branch_width_cherry = 0.04
        print("Creating bins: ", end="")
        my_pcd.create_bins(2 * smallest_branch_width_cherry / 4.0)
        print("{0}".format(my_pcd.div))
        print("Point cloud width/height: ", end=" ")
        vec = np.array(my_pcd.max_pt) - np.array(my_pcd.min_pt)
        print(vec)
        print("Expected bin width {0:0.4f}".format(np.linalg.norm(vec) / 500))

        with open(fname_rw, "w") as fid_out:
            my_pcd.write(fid_out)

    from Cylinder import Cylinder
    from test_pts import best_pts, bad_pts
    cyl_pts = best_pts()
    cyl_pts.update(bad_pts())

    cyl = Cylinder()
    for cyl_id, label in cyl_pts.items():
        ret_val = my_pcd.find_connected(cyl_id, my_pcd.div * 10.0)
        fname = "data/cyl_{0}.txt".format(cyl_id)
        cyl.set_fit_pts(cyl_id, [reg[0] for reg in ret_val], my_pcd.points)
        with open(fname, "w") as f:
            cyl.write(f, write_pts=True)

    for pid_rand in np.random.uniform(0, 1, 40):
        pid = int(np.floor(pid_rand * len(my_pcd)))
        ret_val = my_pcd.find_connected(pid, my_pcd.div * 10.0)
        print(ret_val)
