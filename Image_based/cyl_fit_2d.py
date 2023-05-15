#!/usr/bin/env python3

# Read in masked images and estimate points where a side branch joins a leader (trunk)

import numpy as np
from matplotlib import pyplot as plt
import json
import cv2
# If this doesn't load, right click on Image_based folder on the LHS and select "Mark directory as...->sources root"
#   This just lets PyCharm know that it should look in the Image_based folders for Python files
from line_seg_2d import LineSeg2D


class Quad:
    def __init__(self, v1, v2, radius, mid_pt=None):
        self.bezier = np.zeros((2, 3))
        self.p0, self.p2, self.orientation = self._orientation((np.array(v1)), np.array(v2))
        if mid_pt is None:
            self.p1 = 0.5 * (self.p0 + self.p2)
        else:
            self.p1 = np.array(mid_pt)
        self.depth_values = []
        self.radius_2d = radius

    def _orientation(self, v1, v2):
        """Set the orientation and ensure left-right or right-left
        @param v1 vertex 1 from branchpointdetection
        @param v2 vertex 2
        @returns three points, orientation as a text string"""
        if abs(v1[1] - v2[1]) > abs(v1[0] - v2[0]):
            ret_orientation = "vertical"
            if v1[1] > v2[1]:
                p0 = v1
                p2 = v2
            else:
                p0 = v1
                p2 = v2
        else:
            ret_orientation = "horizontal"
            if v1[0] > v2[0]:
                p0 = v1
                p2 = v2
            else:
                p0 = v1
                p2 = v2
        return p0, p2, ret_orientation

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        pts = np.array([self.p0[i] * (1-t) ** 2 + 2 * (1-t) * t * self.p1[i] + t ** 2 * self.p2[i] for i in range(0, 2)])
        return pts.transpose()
        #return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2

    def tangent_axis(self, t):
        """ Return the tangent vec
        @param t in 0, 1
        @return 2d vec"""
        return  2 * t * (self.p0 - 2.0 * self.p1 + self.p2) - 2 * self.p0 + 2 * self.p1

    def edge_pts(self, t):
        """ Return the left edge of the tube
        @param t in 0, 1
        @return 2d pts, left and right edge"""
        pt = self.pt_axis(t)
        vec = self.tangent_axis(t)
        vec_step = self.radius_2d * vec / np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        left_pt = [pt[0] - vec_step[1], pt[1] + vec_step[0]]
        right_pt = [pt[0] + vec_step[1], pt[1] - vec_step[0]]
        return left_pt, right_pt

    @staticmethod
    def _rect_in_image(im, r, pad=2):
        """ See if the rectangle is within the image boundaries
        @im - image (for width and height)
        @r - the rectangle
        @pad - how much to allow for overlap
        @return True or False"""
        if np.min(r) < pad:
            return False
        if np.max(r[:, 0]) > im.shape[1] + pad:
            return False
        if np.max(r[:, 1]) > im.shape[0] + pad:
           return False
        return True

    def _rect_corners(self, t1, t2, perc_width=0.3):
        """ Get two rectangles covering the expected edges of the cylinder
        @param t1 starting t value
        @param t2 ending t value
        @param perc_width How much of the radius to move in/out of the edge
        @returns two rectangles"""
        vec_ts = self.tangent_axis(0.5 * (t1 + t2))
        edge_left1, edge_right1 = self.edge_pts(t1)
        edge_left2, edge_right2 = self.edge_pts(t2)

        vec_step = perc_width * self.radius_2d * vec_ts / np.sqrt(vec_ts[0] * vec_ts[0] + vec_ts[1] * vec_ts[1])
        rect_left = np.array([[edge_left1[0] + vec_step[1], edge_left1[1] - vec_step[0]],
                              [edge_left2[0] + vec_step[1], edge_left2[1] - vec_step[0]],
                              [edge_left2[0] - vec_step[1], edge_left2[1] + vec_step[0]],
                              [edge_left1[0] - vec_step[1], edge_left1[1] + vec_step[0]]], dtype="float32")
        rect_right = np.array([[edge_right2[0] - vec_step[1], edge_right2[1] + vec_step[0]],
                               [edge_right1[0] - vec_step[1], edge_right1[1] + vec_step[0]],
                               [edge_right1[0] + vec_step[1], edge_right1[1] - vec_step[0]],
                               [edge_right2[0] + vec_step[1], edge_right2[1] - vec_step[0]],
                               ], dtype="float32")
        return rect_left, rect_right

    def _rect_corners_interior(self, t1, t2, perc_width=0.3):
        """ Get a rectangle covering the expected interior of the cylinder
        @param t1 starting t value
        @param t2 ending t value
        @param perc_width How much of the radius to move in/out of the edge
        @returns two rectangles"""
        vec_ts = self.tangent_axis(0.5 * (t1 + t2))
        pt1 = self.pt_axis(t1)
        pt2 = self.pt_axis(t2)

        vec_step = perc_width * self.radius_2d * vec_ts / np.sqrt(vec_ts[0] * vec_ts[0] + vec_ts[1] * vec_ts[1])
        rect = np.array([[pt1[0] + vec_step[1], pt1[1] - vec_step[0]],
                         [pt2[0] + vec_step[1], pt2[1] - vec_step[0]],
                         [pt2[0] - vec_step[1], pt2[1] + vec_step[0]],
                         [pt1[0] - vec_step[1], pt1[1] + vec_step[0]]], dtype="float32")
        return rect

    def boundary_rects(self, step_size=40, perc_width=0.3):
        """ Get two rectangles covering the expected edges of the cylinder
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        @returns a list of pairs of left,right rectangles - evens are left, odds right"""

        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(1.0 / t_step))
        t_step_exact = 1.0 / (n_boxes)
        rects = []
        ts = []
        for t in np.arange(0, 1.0, step=t_step_exact):
            rect_left, rect_right = self._rect_corners(t, t + t_step_exact, perc_width=perc_width)
            rects.append(rect_left)
            rects.append(rect_right)
            ts.append(t + 0.5 * t_step_exact)
            ts.append(t + 0.5 * t_step_exact)
        return rects, ts

    def interior_rects(self, step_size=40, perc_width=0.3):
        """ Find which pixels are valid depth and fit average depth
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(1.0 / t_step))
        t_step_exact = 1.0 / (n_boxes)
        rects = []
        ts = []
        for t in np.arange(0, 1.0, step=t_step_exact):
            rect = self._rect_corners_interior(t, t + t_step_exact, perc_width=perc_width)
            rects.append(rect)
            ts.append(t + 0.5 * t_step_exact)
        return rects, ts

    def interior_rects_mask(self, image_shape, step_size=40, perc_width=0.3):
        """ Find which pixels are inside the quad
        @param image_shape - shape of image to fill mask with
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(1.0 / t_step))
        t_step_exact = 1.0 / (n_boxes)
        im_mask = np.zeros(image_shape, dtype=bool)
        for t in np.arange(0, 1.0, step=t_step_exact):
            rect = self._rect_corners_interior(t, t + t_step_exact, perc_width=perc_width)
            self.draw_rect_filled(im_mask, rect)

        return im_mask

    def _image_cutout(self, im, rect, step_size, height):
        """Cutout a warped bit of the image and return it
        @param im - the image rect is in
        @param rect - four corners of the rectangle to cut out
        @param step_size - the length of the destination rectangle
        @param height - the height of the destination rectangle
        @returns an image, and the reverse transform"""
        rect_destination = np.array([[0, 0], [step_size, 0], [step_size, height], [0, height]], dtype="float32")
        tform3 = cv2.getPerspectiveTransform(rect, rect_destination)
        tform3_back = np.linalg.pinv(tform3)
        return cv2.warpPerspective(im, tform3, (step_size, height)), tform3_back

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
            b_rect_inside = Quad._rect_in_image(im_depth, r, pad=2)

            im_warp, tform_inv = self._image_cutout(im_depth, r, step_size=step_size, height=height)

            stats_slice = {"Min": np.min(im_warp),
                           "Max": np.max(im_warp),
                           "Median": np.median(im_warp)}
            stats_slice["Perc_in_range"] = np.count_nonzero(np.abs(im_warp - stats_slice["Median"]) < 10) / (im_warp.size)
            perc_consistant += stats_slice["Perc_in_range"]
            stats.append(stats_slice)
        perc_consistant /= len(rects)
        return perc_consistant, stats

    def find_edges_hough_transform(self, im_edge, step_size=40, perc_width=0.3, axs=None):
        """Find the hough transform of the images in the boxes; save the line orientations
        @param im_edge - edge image
        @param step_size - how many pixels to use in the hough
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
            b_rect_inside = Quad._rect_in_image(im_edge, r, pad=2)

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
                    # print(f"Orig {p}, transform back {p1_back}")

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

                        # print(f"rho {rho} theta {theta}")
                        # print(f"A {line_abc_constraints}")
                        # print(f"b {line_b}")
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
                    # print(f"Found {len(lines[0])} lines")
            else:
                if axs is not None:
                    axs.imshow(im_debug, origin='lower')
                    print(f"Found no lines")

            if axs is not None:
                axs.clear()

        return ts[0::2], ret_segs

    def _time_step_from_im_step(self, step_size):
        """ How far to step along the curve to step that far in the image
        @param step_size how many pixels to use in the box"""
        crv_length = np.sqrt(np.sum((self.p2 - self.p0) ** 2))
        return min(1, step_size / crv_length)

    def _setup_least_squares(self, ts, ):
        """Setup the least squares approximation
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

        # print(f"Residuals {residuals}, rank {rank}")
        b_rhs[1, :] = self.p1
        pts_diffs = np.sum(np.abs(new_pts - b_rhs[0:3, :]))

        if self.orientation == "vertical":
            new_pts[0, 1] = self.p0[1]
            new_pts[2, 1] = self.p2[1]
        else:
            new_pts[0, 0] = self.p0[0]
            new_pts[2, 0] = self.p2[0]
        self.p0 = new_pts[0, :]
        self.p1 = new_pts[1, :]
        self.p2 = new_pts[2, :]
        return pts_diffs

    def adjust_quad_by_mask(self, im_mask, step_size=40, perc_width=1.2, axs=None):
        """Replace the linear approximation with one based on following the mask
        @param im_mask - mask image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @param axs - optional axes to draw the cutout in
        @returns how much the points moved"""
        height = int(self.radius_2d)
        rects, ts = self.interior_rects(step_size=step_size, perc_width=perc_width)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = self._setup_least_squares(ts)

        x_grid, y_grid = np.meshgrid(range(0, step_size), range(0, height))
        if axs is not None:
            axs.imshow(im_mask, origin='lower')
        for i, r in enumerate(rects):
            b_rect_inside = Quad._rect_in_image(im_mask, r, pad=2)

            im_warp, tform_inv = self._image_cutout(im_mask, r, step_size=step_size, height=height)
            if b_rect_inside and np.sum(im_warp > 0) > 0:
                x_mean = np.mean(x_grid[im_warp > 0])
                y_mean = np.mean(y_grid[im_warp > 0])
                pt_warp_back = tform_inv @ np.transpose(np.array([x_mean, y_mean, 1]))
                # print(f"{self.pt_axis(ts[i])} ({x_mean}, {y_mean}), {pt_warp_back}")
                b_rhs[i, :] = pt_warp_back[0:2]
            else:
                # print(f"Empty slice {r}")
                pass

            if axs is not None:
                axs.clear()
                axs.imshow(im_warp, origin='lower')

        return self._extract_least_squares(a_constraints, b_rhs)

    def _hough_edge_to_middle(self, p1, p2):
        """ Convert the two end points to an estimate of the mid-point and the pt on the spine
        @param p1 upper left [if left edge] or lower right (if right edge)
        @param p2
        returns mid_pt, center_pt"""
        mid_pt = 0.5 * (p1 + p2)
        vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        vec = vec / np.linalg.norm(vec)
        vec_in = np.array([vec[1], -vec[0]])
        pt_middle = mid_pt + vec_in * self.radius_2d
        return mid_pt, pt_middle

    def adjust_quad_by_hough_edges(self, im_edge, step_size=40, perc_width=0.3, axs=None):
        """Replace the linear approximation with one based on following the mask
        @param im_mask - mask image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @perc_orig - percent to keep original pts (weitght in least squares)
        @param axs - optional axes to draw the cutout in
        @returns how much the points moved"""

        # Find all the edge rectangles that have points
        ts, seg_edges = self.find_edges_hough_transform(im_edge, step_size=step_size, perc_width=perc_width, axs=axs)

        if axs is not None:
            axs.clear()
            im_show_lines = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = self._setup_least_squares(ts)

        radius_avg = []
        for i_seg, s in enumerate(seg_edges):
            s1, s2 = s

            if axs is not None:
                for pts in s1:
                    self.draw_line(im_show_lines, pts[0], pts[1], (120, 120, 255), 4)
                for pts in s2:
                    self.draw_line(im_show_lines, pts[0], pts[1], (120, 120, 255), 4)

            # No Hough edges - just keep the current estimate in the LS solver
            if s1 == [] and s2 == []:
                continue

            pt_from_left = np.zeros((1, 2))
            mid_pt_left = np.zeros((1, 2))
            for p1, p2 in s1:
                mid_pt, pt_middle = self._hough_edge_to_middle(p1, p2)
                pt_from_left += pt_middle
                mid_pt_left += mid_pt

            pt_from_right = np.zeros((1, 2))
            mid_pt_right = np.zeros((1, 2))
            for p1, p2 in s2:
                mid_pt, pt_middle = self._hough_edge_to_middle(p1, p2)
                pt_from_right += pt_middle
                mid_pt_right += mid_pt

            if len(s1) > 0 and len(s2) > 0:
                mid_pt_left = mid_pt_left / len(s1)
                mid_pt_right = mid_pt_right / len(s2)

                if axs is not None:
                    # print(f"{mid_pt_left.shape}, {mid_pt_right.shape}")
                    self.draw_line(im_show_lines, mid_pt_left, mid_pt_right, (180, 180, 180), 2)

                pt_mid = 0.5 * (mid_pt_left + mid_pt_right)
                # print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_mid}")
                b_rhs[i_seg, :] = pt_mid

                radius_avg.append(0.5 * np.linalg.norm(mid_pt_right - mid_pt_left))
            elif len(s1) > 0:
                pt_from_left = pt_from_left / len(s1)

                # print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_from_left}")
                b_rhs[i_seg, :] = pt_from_left
                if axs is not None:
                    self.draw_line(im_show_lines, mid_pt_left, pt_from_left, (250, 180, 250), 2)
            else:
                pt_from_right = pt_from_right / len(s2)

                # print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_from_right}")
                b_rhs[i_seg, :] = pt_from_right
                if axs is not None:
                    self.draw_line(im_show_lines, mid_pt_right, pt_from_right, (250, 180, 250), 2)

        if len(radius_avg) > 0:
            # print(f"Radius before {self.radius_2d}")
            self.radius_2d = 0.5 * self.radius_2d + 0.5 * np.mean(np.array(radius_avg))
            # print(f"Radius after {self.radius_2d}")
        if axs is not None:
            axs.imshow(im_show_lines)

        return self._extract_least_squares(a_constraints, b_rhs)

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

    def is_wire(self):
        """Determine if this is likely a wire (long, narrow, straight, and thin)
        @return True/False
        """
        rad_clip = 3
        if self.radius_2d > rad_clip:
            return False

        line_axis = LineSeg2D(self.p0, self.p2)
        pt_proj, _ = line_axis.projection(self.p1)

        dist_line = np.linalg.norm(self.p1 - pt_proj)
        if dist_line > rad_clip:
            return False

        return True

    def draw_quad(self, im):
        """ Set the pixels corresponding to the quad to white
        @im numpy array"""
        n_pts_quad = 6
        pts = self.pt_axis(np.linspace(0, 1, n_pts_quad))
        col_start = 125
        col_div = 120 // (n_pts_quad - 1)
        for p1, p2 in zip(pts[0:-1], pts[1:]):
            cv2.line(im, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (col_start, col_start, col_start), thickness=2)
            col_start += col_div
        """
        rr, cc = draw.bezier_curve(int(self.p0[0]), int(self.p0[1]),
                                   int(self.p1[0]), int(self.p1[1]),
                                   int(self.p2[0]), int(self.p2[1]), weight=2)
        im[rr, cc, 0:3] = (0.1, 0.9, 0.1)
        """

    def draw_boundary(self, im, step_size = 10):
        """ Draw the edge boundary"""
        t_step = self._time_step_from_im_step(step_size)
        max_n = max(2, int(1.0 / t_step))
        edge_pts_draw = [self.edge_pts(t) for t in np.linspace(0, 1, max_n )]
        col_start = 125
        col_div = 120 // (max_n)
        for p1, p2 in zip(edge_pts_draw[0:-1], edge_pts_draw[1:]):
            for i in range(0, 2):
                cv2.line(im, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),
                         (220 - i * 100, col_start, 20 + i * 100), thickness=2)
                """
                rr, cc = draw.line(int(pt1[i][0]), int(pt1[i][1]), int(pt2[i][0]), int(pt2[i][1]))
                rr = np.clip(rr, 0, im.shape[0]-1)
                cc = np.clip(cc, 0, im.shape[1]-1)
                im[rr, cc, 0:3] = (0.3, 0.4, 0.5 + i * 0.25)
                """
            col_start += col_div

    def draw_edge_rect(self, im, rect, col=(50, 255, 255)):
        """ Draw a rectangle in the image
        @param im - the image
        @param rect - the rect as a 4x2 np array
        """
        col_lower_left = (0, 255, 0)
        for i, p1 in enumerate(rect):
            p2 = rect[(i+1) % 4]
            if i == 0:
                col_to_use = col_lower_left
            else:
                col_to_use = col
            cv2.line(im, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col_to_use, thickness=1)
        """
        rr, cc = draw.polygon_perimeter([int(x) for x, _ in rect],
                                        [int(y) for _, y in rect],
                                        shape=im.shape, clip=True)
        rr = np.clip(rr, 0, im.shape[0]-1)
        cc = np.clip(cc, 0, im.shape[1]-1)
        im[rr, cc, 0:3] = (0.1, 0.9, 0.9)
        """

    def draw_rect_filled(self, im, rect, col=(50, 255, 255)):
        """ Fill in the rectangle in the image
        @param im - the image
        @param rect - the rect as a 4x2 np array
        """
        xs = [p[0] for p in rect]
        ys = [p[1] for p in rect]
        diff_x = max(xs) - min(xs)
        diff_y = max(ys) - min(ys)
        for s in np.linspace(0.0, 1.0, int(np.ceil(diff_x))):
            for t in np.linspace(0, 1, int(np.ceil(diff_y))):
                xy = (1-s) * (1-t) * rect[0] + (1-s) * t * rect[1] + s * t * rect[2] + s * (1-t) * rect[3]
                xy[0], xy[1] = xy[1], xy[0]
                if 0 < int(xy[0]) < im.shape[0] and 0 < int(xy[1]) < im.shape[1]:
                    im[int(xy[0]), int(xy[1])] = col

    def draw_edge_rects(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.boundary_rects(step_size, perc_width)
        col_incr = 255 // len(rects)
        for i, r in enumerate(rects):
            col = (i * col_incr, 100 + (i%2) * 100, i * col_incr)
            self.draw_edge_rect(im, r, col=col)

    def draw_edge_rects_markers(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.boundary_rects(step_size, perc_width)
        s1 = 0.25
        s2 = 0.5
        t = 0.25
        col_left = (200, 200, 125)
        col_right = (250, 250, 250)
        for i, r in enumerate(rects):
            p1 = ((1-s1) * (1-t) * r[0] +
                  (s1) * (1-t) * r[1] +
                  (s1) * (t) * r[2] +
                  (1-s1) * (t) * r[3])
            p2 = ((1-s2) * (1-t) * r[0] +
                  (s2) * (1-t) * r[1] +
                  (s2) * (t) * r[2] +
                  (1-s2) * (t) * r[3])
            if i % 2:
                self.draw_line(im, p1, p2, color=col_left, thickness=2)
            else:
                self.draw_line(im, p1, p2, color=col_right, thickness=2)

    def draw_interior_rects(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.interior_rects(step_size, perc_width)
        col_incr = 255 // len(rects)
        for i, r in enumerate(rects):
            col = (i * col_incr, 100 + (i%2) * 100, i * col_incr)
            self.draw_edge_rect(im, r, col=col)

    def write_json(self, fname):
        """Convert to array and write out
        @param fname file name to write to"""
        pts_dict = {"radius": self.radius_2d,
                    "p0": [self.p0[0], self.p0[1]],
                    "p1": [self.p1[0], self.p1[1]],
                    "p2": [self.p2[0], self.p2[1]]
                    }
        with open(fname, 'w') as f:
            json.dump(pts_dict, f)

    def read_json(self, fname):
        """ Read back in from json file
        @param fname file name to read from"""
        with open(fname, 'r') as f:
            pts_dict = json.load(f)
        self.radius_2d = pts_dict["radius"]
        self.p0 = np.array(pts_dict["p0"])
        self.p1 = np.array(pts_dict["p1"])
        self.p2 = np.array(pts_dict["p2"])


class BranchInImage:
    def __init__(self, path, image_name, quad):
        self.quad = quad


def process_image(path, im_name):
    """Do branch point detection followed by fit to mask followed by quad adjust by hough
    @param path - path to find files in
    @param im_name - name of image"""

    return 0


def check_one():
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
    quad = Quad(trunk_pts['lower_left'], trunk_pts['upper_right'], 0.5 * trunk_pts['width'])

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
        # print(f"Res {res}")

    # Draw the original, the edges, and the depth mask with the fitted quad
    quad.draw_quad(im_orig)
    quad.draw_boundary(im_orig, 10)
    quad.draw_quad(im_covert_back)

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
        # print(f"Res Hough {ret}")

    im_orig = cv2.imread('data/forcindy/0.png')
    quad.draw_quad(im_orig)
    quad.draw_boundary(im_orig, 10)
    cv2.imwrite('data/forcindy/0_quad.png', im_orig)

    # print("foo")


if __name__ == '__main__':
    check_one()
