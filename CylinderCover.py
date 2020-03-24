#!/usr/bin/env python3

from MyPointCloud import MyPointCloud
from Cylinder import Cylinder
from list_read_write import ReadWrite
from copy import copy
import os

class PCAScore(ReadWrite):
    def __init__(self, pt_id=0):
        super(PCAScore, self).__init__("PCASCORE")
        self.pt_id = pt_id
        self.score = 1e6
        self.id_start = -1

    def is_set(self):
        if self.id_start == -1:
            return False
        return True


class CylinderCover(ReadWrite):
    def __init__(self, pcd=None):
        super(CylinderCover, self).__init__("CYLINDERCOVER")
        if pcd is None:
            self.my_pcd = MyPointCloud()
        else:
            self.my_pcd = pcd

        # Decent guesses for these values; set in fit, though
        self.radius_min = 0.015
        self.radius_max = 0.09
        self.height = 0.28
        self.radius_search = self.height / 2

        # Cylinders that come out of pca/radius fit, and final, fitted cylinders
        self.pt_score = []
        self.cyls_pca = []
        self.cyls_fitted = []

    def find_good_pca(self, mark_size, in_height, in_rad_min, in_rad_max):
        """
        Use just the pca and radius fit to find an initial set of branches/trunks
        :param mark_size: Mark off points in good cylinders so won't fit to them again
        :param in_height: Desired length of branch to fit to (should be at least 2 times rad_max)
        :param in_rad_min: Expected radius of smallest branches
        :param in_rad_max: Maximum radius of trunk
        :return: Number good cylinders found; stores cylinders in cyls_pca (sorted by score)
        """
        # Keep values
        self.radius_min = in_rad_min
        self.radius_max = in_rad_max
        self.height = in_height

        # Some wriggle room for fitting cylinders
        height_clip_low = max(4 * in_rad_min, in_height * 0.8)
        height_clip_high = min(4 * in_rad_max, in_height * 1.1)

        # Trim out those without a good ratio without fitting
        pca_clip_low = 1.6
        pca_clip_high = 40

        # Cap the nearest neighbor search
        self.radius_search = in_height * 0.5
        clip = mark_size * in_height   # Only mark points that are close to starting point

        # Store best fit for each point found so far
        pca_score = [PCAScore(i) for i in range(len(self.my_pcd))]

        # Track some stats on how many points we tried
        n_bins_tried = 0
        n_bad_bins = 0
        n_skipped = 0
        count_size_neighbor = [len(pca_score), 0, 0]
        print("Total bins {0}, total pts {1}".format(len(self.my_pcd.bin_list), len(self.my_pcd)))

        # Put the cylinders in here so we can sort them before storing
        cyl_list_to_sort = []



        # Go through all the bins...
        for b in self.my_pcd.bin_list.values():
            if pca_score[b[0]].id_start != -1 or len(b) < 5:
                n_skipped += 1
                continue

            n_bins_tried += 1
            if n_bins_tried % 50 == 0:
                print("{0}, {1}, skipped {2}".format(n_bins_tried, len(cyl_list_to_sort), n_skipped))

            region = self.my_pcd.find_connected(b[0], self.radius_search)

            # Keep some stats on number of points used for each fit
            n_in_region = len(region)
            count_size_neighbor[0] = min(count_size_neighbor[0], n_in_region)
            count_size_neighbor[1] += n_in_region
            count_size_neighbor[2] = max(count_size_neighbor[0], n_in_region)

            cyl = Cylinder()
            cyl.set_fit_pts(b[0], [reg[0] for reg in region], self.my_pcd.points)
            cyl.fit_pca()
            # Don't bother if the ratio is really, really off
            if not (pca_clip_low < cyl.pca_ratio() < pca_clip_high):
                n_bad_bins = n_bad_bins + 1
                continue

            # If height is not close to target means we have a blob of unconnected points
            cyl.fit_radius()
            if not height_clip_low < cyl.height < height_clip_high:
                n_bad_bins += 1
                continue

            if cyl.radius < in_rad_min or cyl.radius > in_rad_max:
                n_bad_bins += 1
                continue

            # Look for good pca ratios (roughly 4)
            score = cyl.fit_radius_2d_err

            # Mark neighborhood points as covered
            b_use = False
            for [p_id, _, p_val] in region:
                if p_val < clip:
                    if pca_score[p_id].is_set() is False or pca_score[p_id].score > score:
                        pca_score[p_id].id_start = b[0]
                        pca_score[p_id].score = score
                        b_use = True
            if b_use:
                cyl_list_to_sort.append((cyl, score))

        count_bad = 0  # Number of points not in any cylinder
        self.pt_score = []
        for s in pca_score:
            if s.id_start == -1:
                self.pt_score.append(10.0)
                count_bad += 1
            else:
                self.pt_score.append(pca_score[s.pt_id].score)

        cyl_list_to_sort.sort(key=lambda l_item: l_item[1])
        for cyl, score in cyl_list_to_sort:
            self.cyls_pca.append(cyl)

        # Print out stats on number of cylinders/bins tried
        print("Neighbor size: min {0} max {1} avg {2:2f}".format(count_size_neighbor[0], count_size_neighbor[2], count_size_neighbor[1]/n_bins_tried))
        print("Tried {0} bins, {1} bad bins, {2} not covered, skipped {3} total {4}".format(n_bins_tried, n_bad_bins, count_bad, n_skipped, len(cyl_list_to_sort)))
        return len(cyl_list_to_sort)

    def optimize_cyl(self, mark_size=0.9):
        """
        Optimize the cylinders from the find_good_pca; eliminate bad ones
        :param mark_size: Size of region to mark off of covered
        :param err_max: Maximum error from err_fit in Cylinder allowed
        :return:
        """
        covered = [False for _ in range(len(self.my_pcd))]

        clip = mark_size * self.height

        # Already sorted - just do optimize fit
        for i, cyl in enumerate(self.cyls_pca):
            if covered[cyl.id()]:
                continue

            cyl.optimize_ang(self.radius_min, self.radius_max)

            if cyl.percentage_in_err > 0.5:
                continue
            if cyl.percentage_out_err > 0.5:
                continue
            if cyl.height_distributon_err > 0.5:
                continue
            if cyl.radius_err > 0.5:
                continue

            print("Optimizing {0} of {1} found {2}".format(i, len(self.cyls_pca), len(self.cyls_fitted)))
            cyl_keep = copy(cyl)
            self.cyls_fitted.append(cyl_keep)

            for p_id in cyl.data.pts_ids:
                dist = MyPointCloud.dist(cyl.pt_center(), self.my_pcd.pt(p_id))
                if dist < clip:
                    self.pt_score[p_id] = cyl.err
                    covered[p_id] = True

        return len(self.cyls_fitted)

    def read(self, base_dir):
        cover_file = os.path.join(base_dir, 'cylinder_cover.txt')
        pcd_file = os.path.join(base_dir, 'pcd_file.txt')

        import ipdb
        ipdb.set_trace()


        with open(cover_file, 'r') as fid:

            self.check_header(fid)
            member_name, n_found = self.read_class_members(fid, ["cyls_pca", "cyls_fitted", "my_pcd"])
            pts = None
            try:
                with open(pcd_file, "r") as fid_pcd:
                    self.my_pcd.read(fid_pcd)
                pts = self.my_pcd.points
            except FileExistsError:
                print("File not found, not loading points {0}".format(self.my_pcd_file_name))

            if not member_name.startswith("cyls_pca"):
                raise ValueError("Should be cyls_pca, was {0}".format(member_name))

            self.cyls_pca = []
            for _ in range(0, n_found):
                self.cyls_pca.append(Cylinder())
                self.cyls_pca[-1].read(fid, all_pts=pts)

            l_str = fid.readline()
            l_strs = l_str.split()
            if not l_strs[0].startswith("cyls_fitted"):
                raise ValueError("Should be cyls_fitted, was {0}".format(l_str))

            self.cyls_fitted = []
            for index in range(0, int(l_strs[1])):
                self.cyls_fitted.append(Cylinder())
                self.cyls_fitted[-1].read(fid, all_pts=pts)

            l_str = fid.readline()
            self.check_footer(l_str)



    def write(self, base_dir):
        cover_file = os.path.join(base_dir, 'cylinder_cover.txt')
        pcd_file = os.path.join(base_dir, 'pcd_file.txt')
        with open(pcd_file, 'w') as fid:
            self.my_pcd.write(fid)

        with open(cover_file, 'w') as fid:

            self.write_header(fid)
            self.write_class_members(fid, dir(self), CylinderCover, ["cyls_pca", "cyls_fitted", "my_pcd"])

            fid.write("cyls_pca {0}\n".format(len(self.cyls_pca)))
            for c in self.cyls_pca:
                c.write(fid, write_pts=False)

            fid.write("cyls_fitted {0}\n".format(len(self.cyls_fitted)))
            for c in self.cyls_fitted:
                c.write(fid, write_pts=False)
            self.write_footer(fid)

#
# def test_cylinder_cover_rw():
#     with open("data/cyl_cover.txt", 'w') as fid:
#         my_cyl_cov.write(fid)
#
#     with open("data/cyl_cover.txt", 'r') as fid:
#         my_cyl_cov.read(fid)


# if __name__ == '__main__':
#
#     fname_read = "data/MyPointCloud.txt"
#     my_cyl_cov = CylinderCover(fname_read)
#
#     # test_cylinder_cover_rw()
#     # Apple
#     width_small_branch = 0.03  # Somewhere between 0.03 and 0.05
#     width_large_branch = 0.18  # somewhere between 0.15 and 0.2
#     height_cyl = 0.25  # Somewhere betwen 0.2 and 0.28
#     rad_min = width_small_branch / 2.0
#     rad_max = width_large_branch / 2.0
#
#     width_small_branch = 0.04  # Somewhere between 0.03 and 0.05
#     width_large_branch = 0.18  # somewhere between 0.15 and 0.2
#     height_cyl = 0.17  # Somewhere betwen 0.2 and 0.28
#     rad_min = width_small_branch / 2.0
#     rad_max = width_large_branch / 2.0
#
#     b_read = True
#     if b_read:
#         with open("data/cyl_cover_pca.txt", 'r') as f:
#             my_cyl_cov.read(f)
#
#         my_cyl_cov.optimize_cyl(mark_size=0.9)
#         with open("data/cyl_cover_all.txt", 'w') as f:
#             my_cyl_cov.write(f)
#     else:
#         my_cyl_cov.find_good_pca(mark_size=0.5, in_height=height_cyl, in_rad_min=rad_min, in_rad_max=rad_max)
#
#         with open("data/cyl_cover_pca.txt", 'w') as f:
#             my_cyl_cov.write(f)
#
#         my_cyl_cov.optimize_cyl(mark_size=0.9)
#         with open("data/cyl_cover_all.txt", 'w') as f:
#             my_cyl_cov.write(f)


