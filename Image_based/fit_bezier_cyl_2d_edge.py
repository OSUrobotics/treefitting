#!/usr/bin/env python3

# Fit a Bezier cylinder to a mask
#  Adds least squares fit to bezier_cyl_2d
#  Adds fitting the bezier_cyl_2d to the mask by trying to place the Bezier curve's middle in the center of
#    the mask. Also adjusts the width
#  Essentially, chunk up the mask into pieces, find the average center, then set up a LS fit that (gradually)
#    moves the center by using each chunk's recommendation for where the center should be
# If no edge image, calculate edge image
#  a) Fit the curve to the masked area
#        Extend to boundaries of image if possible
#  b) Calculate IoU for mask and fitted curve
#    b.1) % pixels in center 80% of Bezier curve mask that are in original mask
#    b.2) % pixels outside of 1.1 * Bezier curve mask that are in original mask
#  c) Output revised mask
#  d) Output edge curve boundaries
#    d.1) Cut out piece of boundary
#    b.2) Map to a rectangle
#    b.3) See if edges
#         If edges, use edge cut out
#         Else use center fitted curve
import numpy as np
import cv2
import json
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from line_seg_2d import LineSeg2D
from BaseStatsImage import BaseStatsImage
from HandleFileNames import HandleFileNames

class FitBezierCyl2DEdge:
    def __init__(self, fname_mask_image, fname_calculated=None, fname_debug=None, b_recalc=False):
        # TODO: Make this look like fit_bezier_cyl_2d_mask, creating a FitBezierCyl2DMask then using the
        # output of that to do the fit to the edge process


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
        # TODO Make this look like fit_bezier_crv_to_mask in FitBezierCyl2DMask
        # TODO rename quad to Beier
        # TODO Like adjust, create a FitBezierCrv2D from an input Bezier crv (the setup_least_squares etc will work)
        # TODO change self.ls etc to fit_bezier_crv
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
                    print(f"{mid_pt_left.shape}, {mid_pt_right.shape}")
                    self.draw_line(im_show_lines, mid_pt_left, mid_pt_right, (180, 180, 180), 2)

                pt_mid = 0.5 * (mid_pt_left + mid_pt_right)
                print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_mid}")
                b_rhs[i_seg, :] = pt_mid

                radius_avg.append(0.5 * np.linalg.norm(mid_pt_right - mid_pt_left))
            elif len(s1) > 0:
                pt_from_left = pt_from_left / len(s1)

                print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_from_left}")
                b_rhs[i_seg, :] = pt_from_left
                if axs is not None:
                    self.draw_line(im_show_lines, mid_pt_left, pt_from_left, (250, 180, 250), 2)
            else:
                pt_from_right = pt_from_right / len(s2)

                print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_from_right}")
                b_rhs[i_seg, :] = pt_from_right
                if axs is not None:
                    self.draw_line(im_show_lines, mid_pt_right, pt_from_right, (250, 180, 250), 2)

        if len(radius_avg) > 0:
            print(f"Radius before {self.radius_2d}")
            self.radius_2d = 0.5 * self.radius_2d + 0.5 * np.mean(np.array(radius_avg))
            print(f"Radius after {self.radius_2d}")
        if axs is not None:
            axs.imshow(im_show_lines)

        return self._extract_least_squares(a_constraints, b_rhs)

    def set_end_pts(self, pt0, pt2):
        """ TODO: pass in the curve and set the crv's end points
        Set the end point to the new end point while trying to keep the curve the same
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

    def debug_image_edge_fit(self, image_debug):
        """ Draw the fitted quad on the image
        @param image_debug - rgb image"""
        # Draw the original, the edges, and the depth mask with the fitted quad
        self.quad.draw_quad(image_debug)
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
