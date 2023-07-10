#!/usr/bin/env python3

# Fit a Bezier cylinder to a mask
#  Adds least squares fit to bezier_cyl_2d
#  Adds fitting the bezier_cyl_2d to the mask by trying to place the Bezier curve's middle in the center of
#    the mask. Also adjusts the width
#  Essentially, chunk up the mask into pieces, find the average center, then set up a LS fit that (gradually)
#    moves the center by using each chunk's recommendation for where the center should be

import numpy as np
import cv2
import json
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from line_seg_2d import LineSeg2D
from BaseStatsImage import BaseStatsImage
from HandleFileNames import HandleFileNames

class FitBezierCyl2D(BezierCyl2D):
    def __init__(self, pt):
        """ Read in the mask image, use the stats to start the quad fit, then fit the quad
        @param fname_mask_image: Mask image name
        @param stats_image: name the stats for this image
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _stats.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats
        self.stats_dict = BaseStatsImage(fname_mask_image, fname_calculated, fname_debug, b_recalc)

        # Now initialize bezier curve with infor from stats
        super(BezierCyl2D, self).__init__(self.stats_dict["lower_left"], self.stats_dict["upper_right"], 0.5 * self.stats_dict['width'])

        # Fit a quad to the mask, using the end points of the base image as a starting point
        print(f"Fitting bezier curve to mask image {fname_mask_image}")
        self.fname_bezier_cyl = None    # The actual quadratic bezier
        self.fname_params = None  # Parameters used to do the fit
        if fname_calculated:
            self.fname_bezier_cyl = fname_calculated + "bezier_cyl_mask.json"
            self.fname_params = fname_calculated + "bezier_cyl_params.json"

        self.quad = None

        # Current parameters for the vertical leader - will need to make this a parameter
        self.params = {"step_size": int(0.5 * self.stats_dict['width'] * 1.5), "width_mask": 1.4, "width": 0.25}

        if b_recalc or not fname_calculated or not exists(fname_calculated):
            if exists(self.fname_bezier_cyl) and not b_recalc:
                BezierCyl2D.read_json(self.fname_bezier_cyl, self)
                with open(self.fname_params, 'r') as f:
                    self.params = json.load(f)
            else:
                self.fit_quad_to_mask(self, self.stats_dict.mask_image, stats=self.stats_dict, params=self.params)
                self.write_json(self.fname_bezier_cyl)
                with open(self.fname_params, 'w') as f:
                    json.dump(self.params, f)

        if fname_debug:
            # Draw the edge and original image with the fitted quad and rects
            im_covert_back = cv2.cvtColor(self.stats_dict.mask_image, cv2.COLOR_GRAY2RGB)
            self.stats_dict.debug_image(im_covert_back)  # The eigen vec
            self.debug_image_quad_fit(im_covert_back)
            cv2.imwrite(fname_debug, im_covert_back)

        self.score = self.score_mask_fit()

    def _setup_least_squares(self, ts):
        """Setup the least squares approximation - ts is the number of constraints to add, also
           puts in a copule constraints to keep end points where they are
        @param ts - t values to use
        @returns A, B for Ax = b """
        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints = np.zeros((len(ts) + 3, 3))
        ts_constraints = np.zeros((len(ts) + 3))
        b_rhs = np.zeros((len(ts) + 3, 2))
        ts_constraints[-3] = 0.0
        ts_constraints[-2] = 0.5
        ts_constraints[-1] = 1.0
        ts_constraints[:-3] = np.transpose(ts)
        a_constraints[:, -3] = (1-ts_constraints) * (1-ts_constraints)
        a_constraints[:, -2] = 2 * (1-ts_constraints) * ts_constraints
        a_constraints[:, -1] = ts_constraints * ts_constraints
        for i, t in enumerate(ts_constraints):
            b_rhs[i, :] = self.pt_axis(ts_constraints[i])
        return a_constraints, b_rhs

    def _extract_least_squares(self, a_constraints, b_rhs):
        """ Do the actual Ax = b and keep horizontal/vertical end points
        @param a_constraints the A of Ax = b
        @param b_rhs the b of Ax = b
        @returns fit error L0 norm"""
        if a_constraints.shape[0] < 3:
            return 0.0

        #  a_at = a_constraints @ a_constraints.transpose()
        #  rank = np.rank(a_at)
        #  if rank < 3:
        #      return 0.0

        new_pts, residuals, rank, _ = np.linalg.lstsq(a_constraints, b_rhs, rcond=None)

        print(f"Residuals {residuals}, rank {rank}")
        b_rhs[1, :] = self.p1
        pts_diffs = np.sum(np.abs(new_pts - b_rhs[0:3, :]))

        # Don't let the end points contract
        if self.orientation is "vertical":
            new_pts[0, 1] = self.p0[1]
            new_pts[2, 1] = self.p2[1]
        else:
            new_pts[0, 0] = self.p0[0]
            new_pts[2, 0] = self.p2[0]
        self.p0 = new_pts[0, :]
        self.p1 = new_pts[1, :]
        self.p2 = new_pts[2, :]
        return pts_diffs

    @staticmethod
    def adjust_bezier_crv_by_mask(bezier_crv, im_mask, step_size=40, perc_width=1.2):
        """Replace the linear approximation with one based on following the mask
        @param im_mask - mask image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @param axs - optional axes to draw the cutout in
        @returns how much the points moved"""
        height = int(bezier_crv.radius_2d)
        rects, ts = bezier_crv.interior_rects(step_size=step_size, perc_width=perc_width)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = bezier_crv._setup_least_squares(ts)

        x_grid, y_grid = np.meshgrid(range(0, step_size), range(0, height))
        for i, r in enumerate(rects):
            b_rect_inside = BezierCyl2D._rect_in_image(im_mask, r, pad=2)

            im_warp, tform_inv = bezier_crv._image_cutout(im_mask, r, step_size=step_size, height=height)
            if b_rect_inside and np.sum(im_warp > 0) > 0:
                x_mean = np.mean(x_grid[im_warp > 0])
                y_mean = np.mean(y_grid[im_warp > 0])
                pt_warp_back = tform_inv @ np.transpose(np.array([x_mean, y_mean, 1]))
                print(f"{bezier_crv.pt_axis(ts[i])} ({x_mean}, {y_mean}), {pt_warp_back}")
                b_rhs[i, :] = pt_warp_back[0:2]
            else:
                print(f"Empty slice {r}")

        return bezier_crv._extract_least_squares(a_constraints, b_rhs)

    def set_end_pts(self, pt0, pt2):
        """ Set the end point to the new end point while trying to keep the curve the same
        @param pt0 new p0
        @param pt2 new p2"""
        l0 = LineSeg2D(self.p0, self.p1)
        l2 = LineSeg2D(self.p1, self.p2)
        t0 = l0.projection(pt0)
        t2 = l0.projection(pt2)

        ts_mid = np.array([0.25, 0.75])
        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = self._setup_least_squares(ts_mid)
        b_rhs[-3, :] = pt0.transpose()
        b_rhs[-2, :] = self.pt_axis(0.5 * (t0 + t2))
        b_rhs[-1, :] = pt2.transpose()
        for i, t in enumerate(ts_mid):
            t_map = (1-t) * t0 + t * t2
            b_rhs[i, :] = self.pt_axis(t_map)

        return self._extract_least_squares(a_constraints, b_rhs)

    @staticmethod
    def fit_bezier_crv_to_mask(bezier_crv, im_mask, stats, params):
        """ Fit a quad to the mask, edge image
        @param bezier_crv - the initial bezier curve
        @param im_mask - the image mask
        @param stats - the stats from BaseStatsImage
        @param params - the parameters to use in the fit
        @return fitted quad and parameters used in the fit"""

        # Fit a Bezier curve to the mask - this does a bit of tweaking to try to extend the end points as
        #  far as possible
        pt_lower_left = stats['center']
        vec_len = stats["Length"] * 0.4
        while pt_lower_left[0] > 2 + stats['x_min'] and pt_lower_left[1] > 2 + stats['y_min']:
            pt_lower_left = stats["center"] - stats["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        pt_upper_right = stats['center']
        vec_len = stats["Length"] * 0.4
        while pt_upper_right[0] < -2 + stats['x_max'] and pt_upper_right[1] < -2 + stats['y_max']:
            pt_upper_right = stats["center"] + stats["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        bezier_crv = BezierCyl2D(pt_lower_left, pt_upper_right, 0.5 * stats['width'])

        # Iteratively move the quad to the center of the mask
        print(f"Res: ", end="")
        for i in range(0, 5):
            res = FitBezierCyl2D.adjust_crv_by_mask(bezier_crv,
                                                    im_mask,
                                                    step_size=params["step_size"], perc_width=params["width_mask"])
            print(f"{res} ", end="")
        print("")
        return bezier_crv, params

    def find_edges_hough_transform(self, im_edge, step_size=40, perc_width=0.3, axs=None):
        """Find the hough transform of the images in the boxes; save the line orientations
        @param im_edge - edge image
        @param step_size - how many pixels to use in each Hough image
        @param perc_width - how wide a rectangle to use on the edges
        @param axs - matplot lib axes for debug image
        @returns center, angle for each box"""

        # Size of the rectangle(s) to cutout is based on the step size and the radius
        height = int(self.radius_2d)
        rect_destination = np.array([[0, 0], [step_size, 0], [step_size, height], [0, height]], dtype="float32")
        rects, ts = self.boundary_rects(step_size=step_size, perc_width=perc_width)

        if axs is not None:
            im_debug = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)
            self.draw_edge_rects(im_debug, step_size=step_size, perc_width=perc_width)
            axs.imshow(im_debug)

        ret_segs = []
        # For fitting y = mx + b
        line_abc_constraints = np.ones((3, 3))
        line_b = np.zeros((3, 1))
        line_b[2, 0] = 1.0
        for i_rect, r in enumerate(rects):
            b_rect_inside = BezierCyl2D._rect_in_image(im_edge, r, pad=2)

            im_warp, tform3_back = self._image_cutout(im_edge, r, step_size=step_size, height=height)
            i_seg = i_rect // 2
            i_side = i_rect % 2
            if i_side == 0:
                ret_segs.append([[], []])

            if axs is not None:
                im_debug = cv2.cvtColor(im_warp, cv2.COLOR_GRAY2RGB)

                i_edge = i_seg // 2
                if i_edge % 2:
                    p1_back = tform3_back @ np.transpose(np.array([1, height / 2, 1]))
                    p2_back = tform3_back @ np.transpose(np.array([step_size - 1, height / 2, 1]))
                else:
                    p1_back = tform3_back @ np.transpose(np.array([1, 1, 1]))
                    p2_back = tform3_back @ np.transpose(np.array([step_size - 1, 1, 1]))
                print(f"l {p1_back}, r {p2_back}")

            # Actual hough transform on the cut-out image
            lines = cv2.HoughLines(im_warp, 1, np.pi / 180.0, 10)

            if axs is not None:
                for i, p in enumerate(r):
                    p1_in = np.transpose(np.array([rect_destination[i][0], rect_destination[i][1], 1.0]))
                    p1_back = tform3_back @ p1_in
                    print(f"Orig {p}, transform back {p1_back}")

            if lines is not None and b_rect_inside:
                for rho, theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = x0 + 1000*(-b)
                    y1 = y0 + 1000*(a)
                    x2 = x0 - 1000*(-b)
                    y2 = y0 - 1000*(a)
                    if np.isclose(theta, 0.0):
                        line_abc = np.zeros((3, 1))
                        if np.isclose(rho, 1.0):
                            line_abc[0] = 1.0
                            line_abc[2] = -1.0
                        else:
                            line_abc[0] = -1.0 / (rho - 1.0)
                            line_abc[2] = 1.0 - line_abc[0]
                    else:
                        line_abc_constraints[0, 0] = x1
                        line_abc_constraints[1, 0] = x2
                        line_abc_constraints[0, 1] = y1
                        line_abc_constraints[1, 1] = y2

                        print(f"rho {rho} theta {theta}")
                        print(f"A {line_abc_constraints}")
                        print(f"b {line_b}")
                        line_abc = np.linalg.solve(line_abc_constraints, line_b)

                    check1 = line_abc[0] * x1 + line_abc[1] * y1 + line_abc[2]
                    check2 = line_abc[0] * x2 + line_abc[1] * y2 + line_abc[2]
                    if not np.isclose(check1, 0.0) or not np.isclose(check2, 0.0):
                        raise ValueError("Cyl Fit 2D: Making line, pts not on line")

                    # We only care about horizontal lines anyways, so ignore vertical ones
                    if not np.isclose(line_abc[0], 0.0):
                        # Get where the horizontal line crosses the left/right edge
                        y_left = -(line_abc[0] * 0.0 + line_abc[2]) / line_abc[1]
                        y_right = -(line_abc[0] * step_size + line_abc[2]) / line_abc[1]

                        # Only keep edges that DO cross the left/right edge
                        if 0 < y_left < height and 0 < y_right < height:
                            p1_in = np.transpose(np.array([0.0, y_left[0], 1.0]))
                            p2_in = np.transpose(np.array([step_size, y_right[0], 1.0]))
                            p1_back = tform3_back @ p1_in
                            p2_back = tform3_back @ p2_in
                            ret_segs[i_seg][i_side].append([p1_back[0:2], p2_back[0:2]])
                    if axs is not None:
                        cv2.line(im_debug, (x1,y1), (x2,y2), (255, 100, 100), 2)
                if axs is not None:
                    axs.imshow(im_debug, origin='lower')
                    print(f"Found {len(lines[0])} lines")
            else:
                if axs is not None:
                    axs.imshow(im_debug, origin='lower')
                    print(f"Found no lines")

            if axs is not None:
                axs.clear()

        return ts[0::2], ret_segs

    @staticmethod
    def adjust_quad_by_edge_image(im_edge, quad, params):
        """ Adjust the quad to the edge image using hough transform
        @param im_edge: The edge image
        @param quad: The original quad fit to the mask
        @param params: The params to use in the fit
        @return the quad and the params"""

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(im_edge, step_size=params["step_size"], perc_width=params["width"], axs=None)
            print(f"Res Hough {ret}")

        return quad

    def check_interior_depth(self, im_depth, step_size=40, perc_width=0.3):
        """ Find which pixels are valid depth and fit average depth
        @param im_depth - depth image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        height = int(self.radius_2d)
        rects, ts = self.interior_rects(step_size=step_size, perc_width=perc_width)

        stats = []
        perc_consistant = 0.0
        for i, r in enumerate(rects):
            b_rect_inside = BezierCyl2D._rect_in_image(im_depth, r, pad=2)

            im_warp, tform_inv = self._image_cutout(im_depth, r, step_size=step_size, height=height)

            stats_slice = {"Min": np.min(im_warp),
                           "Max": np.max(im_warp),
                           "Median": np.median(im_warp)}
            stats_slice["Perc_in_range"] = np.count_nonzero(np.abs(im_warp - stats_slice["Median"]) < 10) / (im_warp.size)
            perc_consistant += stats_slice["Perc_in_range"]
            stats.append(stats_slice)
        perc_consistant /= len(rects)
        return perc_consistant, stats

    @staticmethod
    def adjust_quad_by_flow_image(im_flow, quad, params):
        """ Not really useful now - fixes the mask by calculating the average depth in the flow mask under the bezier
        the trimming off mask pixels that don't belong
        @param im_flow: Flow or depth image (gray scale)
        @param quad: The quad to adjust
        @param params: Parameters to use for adjust
        @return the adjusted quad"""

        print("Quad adjust res: ", end="")
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im_flow,
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            print(f"{res} ")
        print(" done")

        return quad

    def score_quad(self, im_flow, quad):
        """ See if the quad makes sense over the optical flow image
        @quad - the quad
        """

        # Two checks: one, are the depth/optical fow values largely consistent under the quad center
        #  Are there boundaries in the optical flow image where the edge of the quad is?
        im_flow_mask = cv2.cvtColor(im_flow, cv2.COLOR_BGR2GRAY)
        perc_consistant, stats_slice = quad.check_interior_depth(im_flow_mask)

        diff = 0
        for i in range(1, len(stats_slice)):
            diff_slices = np.abs(stats_slice[i]["Median"] - stats_slice[i-1]["Median"])
            if diff_slices > 20:
                print(f"Warning: Depth values not consistant from slice {self.fname_quad} {i} {stats_slice}")
            diff += diff_slices
        if perc_consistant < 0.9:
            print(f"Warning: not consistant {self.fname_quad} {stats_slice}")
        return perc_consistant, diff / (len(stats_slice) - 1)

    def debug_image_quad_fit(self, image_debug):
        """ Draw the fitted quad on the image
        @param image_debug - rgb image"""
        # Draw the original, the edges, and the depth mask with the fitted quad
        self.quad.draw_bezier(image_debug)
        if self.quad.is_wire():
            LineSeg2D.(image_debug, self.quad.p0, (255, 0, 0), thickness=2, length=10)
            LineSeg2D.draw_cross(image_debug, self.quad.p2, (255, 0, 0), thickness=2, length=10)
        else:
            self.quad.draw_boundary(image_debug, 10)


if __name__ == '__main__':
    #path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = False
    for ind in all_files.loop_masks():
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        mask_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=True)
        if not b_do_debug:
            mask_fname_debug = ""

        mask_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        b_stats = FitBezierCyl2D(mask_fname, mask_fname_calculate, mask_fname_debug, b_recalc=b_do_recalc)
    from branchpointdetection import BranchPointDetection

    # Compute all the branch points/approximate lines for branches
    bp = BranchPointDetection("data/forcindy/", "0")

    # Read in/compute the additional images we need for debugging
    #   Original image, convert to canny edge
    #   Mask image
    #   Depth image
    im_orig = cv2.imread('data/forcindy/0.png')
    im_depth = cv2.imread('data/forcindy/0_depth.png')
    im_mask_color = cv2.imread('data/forcindy/0_trunk_0.png')
    im_mask = cv2.cvtColor(im_mask_color, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
    im_edge = cv2.Canny(im_gray, 50, 150, apertureSize=3)
    im_depth_color = cv2.cvtColor(im_depth, cv2.COLOR_BGR2RGB)
    im_covert_back = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)

    # Write out edge image
    cv2.imwrite('data/forcindy/0_edges.png', im_edge)

    # For the vertical leader...
    trunk_pts = bp.trunks[0]["stats"]
    # Fit a quad to the trunk
    quad = BezierCyl2D(trunk_pts['lower_left'], trunk_pts['upper_right'], 0.5 * trunk_pts['width'])

    # Current parameters for the vertical leader
    step_size_to_use = int(quad.radius_2d * 1.5)  # Go along the edge at 1.5 * the estimated radius
    perc_width_to_use = 0.3  # How "fat" to make the edge rectangles
    perc_width_to_use_mask = 1.4  # How "fat" a rectangle to cover the mask

    # Debugging image - draw the interior rects
    quad.draw_interior_rects(im_mask_color, step_size=step_size_to_use, perc_width=perc_width_to_use_mask)
    cv2.imwrite('data/forcindy/0_mask.png', im_mask_color)

    # For debugging images
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im_orig)
    axs[0, 1].imshow(im_mask_color)
    plt.tight_layout()

    # Iteratively move the quad to the center of the mask
    for i in range(0, 5):
        res = quad.adjust_quad_by_mask(im_mask,
                                       step_size=step_size_to_use, perc_width=perc_width_to_use_mask,
                                       axs=axs[1, 0])
        print(f"Res {res}")

    # Draw the original, the edges, and the depth mask with the fitted quad
    quad.draw_bezier(im_orig)
    quad.draw_boundary(im_orig, 10)
    quad.draw_bezier(im_covert_back)

    quad.draw_edge_rects(im_orig, step_size=step_size_to_use, perc_width=perc_width_to_use)
    quad.draw_edge_rects(im_covert_back, step_size=step_size_to_use, perc_width=perc_width_to_use)
    #quad.draw_edge_rects_markers(im_edge, step_size=step_size_to_use, perc_width=perc_width_to_use)
    quad.draw_interior_rects(im_depth_color, step_size=step_size_to_use, perc_width=perc_width_to_use)

    im_both = np.hstack([im_orig, im_covert_back, im_depth_color])
    cv2.imshow("Original and edge and depth", im_both)
    cv2.imwrite('data/forcindy/0_rects.png', im_both)

    # Now do the hough transform - first draw the hough transform edges
    for i in range(0, 5):
        ret = quad.adjust_quad_by_hough_edges(im_edge, step_size=step_size_to_use, perc_width=perc_width_to_use, axs=axs[1, 1])
        print(f"Res Hough {ret}")

    im_orig = cv2.imread('data/forcindy/0.png')
    quad.draw_bezier(im_orig)
    quad.draw_boundary(im_orig, 10)
    cv2.imwrite('data/forcindy/0_quad.png', im_orig)

    print("foo")
