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
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from fit_bezier_cyl_2d import FitBezierCyl2D
from line_seg_2d import LineSeg2D
from HandleFileNames import HandleFileNames
from fit_bezier_cyl_2d_mask import FitBezierCyl2DMask


class FitBezierCyl2DEdge:
    # For fitting y = mx + b in Hough transform
    _line_abc_constraints = np.ones((3, 3))
    _line_b = np.array([0.0, 0.0, 1.0])
    _line_abc = np.zeros((3, 1))

    def __init__(self, fname_rgb_image, fname_edge_image, fname_mask_image, fname_calculated=None, fname_debug=None, b_recalc=False):
        """ Read in the mask image, use the stats to start the Bezier fit, then fit the Bezier to the mask
        @param fname_rgb_image: Original rgb image name (for making edge name if we have to)
        @param fname_edge_image: Create and store the edge image, or read it in if it exists
        @param fname_mask_image: Mask image name - for the base stats image
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _crv.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc. Set to None if no debug image
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the base mask image
        self.mask_crv = FitBezierCyl2DMask(fname_mask_image, fname_calculated, fname_debug, b_recalc)

        # Read in the RGB image
        self.image_rgb = cv2.imread(fname_rgb_image)

        # Now calculate the edge image, if it doesn't exist
        if exists(fname_edge_image):
            im_edge_color = cv2.imread(fname_edge_image)
            self.image_edge = cv2.cvtColor(im_edge_color, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2GRAY)
            self.image_edge = cv2.Canny(im_gray, 50, 150, apertureSize=3)
            cv2.imwrite(fname_edge_image, self.image_edge)

        # Create the calculated file names
        b_debug = False
        if fname_debug:
            print(f"Fitting bezier curve to edge image {fname_edge_image}")
            b_debug = True

        self.fname_bezier_cyl_edge = None    # The actual quadratic bezier
        self.fname_params = None  # Parameters used to do the fit
        # Create the file names for the calculated data that we'll store (initial curve, curve fit to mask, parameters)
        if fname_calculated:
            self.fname_bezier_cyl_edge = fname_calculated + "_bezier_cyl_edge.json"
            self.fname_params = fname_calculated + "_bezier_cyl_edge_params.json"

        #   This is the curve that will be fit to the edge
        self.bezier_crv_fit_to_edge = None

        # Current parameters for the vertical leader fit
        # TODO make this a parameter in the init function
        self.params = {"step_size": int(self.mask_crv.bezier_crv_fit_to_mask.radius(0.5) * 1.5), "width_mask": 1.4, "width": 0.25}

        # Fit the curve to the edges
        if b_recalc or not fname_calculated or not exists(self.fname_bezier_cyl_edge):
            # Recalculate and write
            self.bezier_crv_fit_to_edge =\
                FitBezierCyl2DEdge.fit_bezier_crv_to_edge(self.mask_crv.bezier_crv_fit_to_mask,
                                                          self.image_edge, self.params, b_debug)
            if fname_calculated:
                self.bezier_crv_fit_to_edge.write_json(self.fname_bezier_cyl_edge)
        else:
            # Read in the pre-calculated curve
            self.bezier_crv_fit_to_edge = BezierCyl2D.read_json(self.fname_bezier_cyl_edge)

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            im_rgb = np.copy(self.image_rgb)
            self.mask_crv.bezier_crv_fit_to_mask.draw_bezier(im_rgb)
            self.mask_crv.bezier_crv_fit_to_mask.draw_boundary(im_rgb)

            # Convert the edge image to an rgb image and draw the fitted bezier curve on it
            im_covert_back = cv2.cvtColor(self.image_edge, cv2.COLOR_GRAY2RGB)
            self.debug_image_edge_fit(im_covert_back)

            # Stack them both together
            im_both = np.hstack([im_rgb, im_covert_back])
            cv2.imwrite(fname_debug + "_edge.png", im_both)

        # self.score = self.score_mask_fit(self.stats_dict.mask_image)
        # print(f"Mask {mask_fname}, score {self.score}")

    @staticmethod
    def _hough_edge_to_middle(bezier_crv, p1, p2, t):
        """ Convert the two end points to an estimate of the mid-point and the pt on the spine
        Two points are the end points of where the edge crosses the rectangle
        @param bezier_crv - the bezier curve
        @param p1 upper left [if left edge] or lower right (if right edge)
        @param p2 Other end point
        @param t along curve
        returns mid_pt, center_pt"""
        # The point on the fit line in the middle of the seg
        mid_pt_on_edge = 0.5 * (p1 + p2)
        vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        vec = vec / np.linalg.norm(vec)
        vec_in = np.array([vec[1], -vec[0]])
        # Take the normal to the line and point it in the distance to the radius
        pt_on_spine = mid_pt_on_edge + vec_in * bezier_crv.radius(t)
        return mid_pt_on_edge, pt_on_spine

    @staticmethod
    def get_horizontal_lines_from_hough(lines, tform3_back, width, height, b_debug=False):
        """ Get left and right edge points from the lines returned from Hough transform
        @param lines - the rho, theta returned from Hough
        @param tform3_back - undo the warp transform
        @param width - width of the cutout
        @param height - height of the cutout
        @param b_debug - print out debug messages if True
        @return list of pairs of left,right points where line crosses cutout image"""
        # For fitting y = mx + b

        ret_pts = []
        # Lines are returned as an angle and a distance from the origin
        for rho, theta in lines[0]:
            # Convert from r, theta to ax + by + c form
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Assuming image is not bigger than 1000,1000
            x1 = x0 + 1000 * (-b)
            y1 = y0 + 1000 * a
            x2 = x0 - 1000 * (-b)
            y2 = y0 - 1000 * a
            if np.isclose(theta, 0.0):
                if np.isclose(rho, 1.0):
                    FitBezierCyl2DEdge._line_abc[0] = 1.0
                    FitBezierCyl2DEdge._line_abc[1] = 0.0
                    FitBezierCyl2DEdge._line_abc[2] = -1.0
                else:
                    FitBezierCyl2DEdge._line_abc[0] = -1.0 / (rho - 1.0)
                    FitBezierCyl2DEdge._line_abc[1] = 0.0
                    FitBezierCyl2DEdge._line_abc[2] = 1.0 - FitBezierCyl2DEdge._line_abc[0]
            else:
                FitBezierCyl2DEdge._line_abc_constraints[0, 0] = x1
                FitBezierCyl2DEdge._line_abc_constraints[1, 0] = x2
                FitBezierCyl2DEdge._line_abc_constraints[0, 1] = y1
                FitBezierCyl2DEdge._line_abc_constraints[1, 1] = y2

                if b_debug:
                    print(f"rho {rho} theta {theta}")
                    print(f"A {FitBezierCyl2DEdge._line_abc_constraints}")
                    print(f"b {FitBezierCyl2DEdge._line_b}")
                FitBezierCyl2DEdge._line_abc = np.linalg.solve(FitBezierCyl2DEdge._line_abc_constraints, FitBezierCyl2DEdge._line_b)

            check1 = FitBezierCyl2DEdge._line_abc[0] * x1 + FitBezierCyl2DEdge._line_abc[1] * y1 + FitBezierCyl2DEdge._line_abc[2]
            check2 = FitBezierCyl2DEdge._line_abc[0] * x2 + FitBezierCyl2DEdge._line_abc[1] * y2 + FitBezierCyl2DEdge._line_abc[2]
            if not np.isclose(check1, 0.0) or not np.isclose(check2, 0.0):
                raise ValueError("FitBezierCyl2DEdge: Making line, pts not on line")

            # We only care about horizontal lines anyways, so ignore vertical ones
            #   Don't have to check for a divide by 0
            if not np.isclose(FitBezierCyl2DEdge._line_abc[1], 0.0):
                # Get where the horizontal line crosses the left/right edge
                y_left = -(FitBezierCyl2DEdge._line_abc[0] * 0.0 + FitBezierCyl2DEdge._line_abc[2]) / FitBezierCyl2DEdge._line_abc[1]
                y_right = -(FitBezierCyl2DEdge._line_abc[0] * width + FitBezierCyl2DEdge._line_abc[2]) / FitBezierCyl2DEdge._line_abc[1]

                # Only keep edges that DO cross the left/right edge
                if 0 < y_left < height and 0 < y_right < height:
                    p1_in = np.transpose(np.array([0.0, y_left, 1.0]))
                    p2_in = np.transpose(np.array([width, y_right, 1.0]))
                    # Get the point back in the original image
                    p1_back = tform3_back @ p1_in
                    p2_back = tform3_back @ p2_in
                    # iseg is per segment, iside is left and right
                    ret_pts.append([p1_back[0:2], p2_back[0:2]])

        if b_debug:
            print(f"Found {len(lines[0])} lines")
        return ret_pts

    @staticmethod
    def find_edges_hough_transform(bezier_crv, im_edge, step_size=40, perc_width=0.3, b_debug=False):
        """Find the hough transform of the images in the boxes; save the line orientations
        @param bezier_crv - curve
        @param im_edge - edge image
        @param step_size - how many pixels to use in each Hough image
        @param perc_width - how wide a rectangle to use on the edges
        @param b_debug - print out debug messages True/False
        @returns center, angle for each box"""

        # Size of the rectangle(s) to cutout is based on the step size and the radius
        height = int(bezier_crv.radius(0.5))
        rect_destination = np.array([[0, 0], [step_size, 0], [step_size, height], [0, height]], dtype="float32")
        rects, ts = bezier_crv.boundary_rects(step_size=step_size, perc_width=perc_width)

        """ If you need to, pring out the edge boundary image
        im_rgb = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)
        bezier_crv.draw_interior_rects(im_rgb, step_size=step_size, perc_width=0.25)
        bezier_crv.draw_edge_rects(im_rgb, step_size=step_size, perc_width=perc_width)
        cv2.imwrite("./data/forcindy/DebugImages/0_trunk_0_fit_edge_bdry.png", im_rgb)
        """

        ret_segs = []
        # For fitting y = mx + b
        line_abc_constraints = np.ones((3, 3))
        line_b = np.zeros((3, 1))
        line_b[2, 0] = 1.0
        # Loop through all the rectangles on the edge
        for i_rect, r in enumerate(rects):
            # Don't do this if the rectangle lies outside of the original image
            b_rect_inside = BezierCyl2D.rect_in_image(im_edge, r, pad=2)

            # Cut out the bit of image from the edge rectangle as a rectilinear image
            im_warp, tform3_back = bezier_crv.image_cutout(im_edge, r, step_size=step_size, height=height)
            i_seg = i_rect // 2
            i_side = i_rect % 2
            # Convert from ever other rectangle being left one to a pair of segs
            if i_side == 0:
                ret_segs.append([[], []])

            # Actual hough transform on the cut-out image
            lines = cv2.HoughLines(im_warp, 1, np.pi / 180.0, 10)

            if lines is not None and b_rect_inside:
                ret_segs[i_seg][i_side] = \
                    FitBezierCyl2DEdge.get_horizontal_lines_from_hough(lines, tform3_back, step_size, height, b_debug)
            else:
                if b_debug:
                    print(f"No lines or rect outside {r} im shape {im_edge.shape}")

        # Grab every other t value, since we're returning left right edges in pairs
        return ts[0::2], ret_segs

    @staticmethod
    def _adjust_bezier_by_hough_edges(fit_bezier_crv, im_edge, step_size=40, perc_width=0.3, b_debug=False):
        """Fit the bezier curve bounaries to the edges in the edge image
        @param fit_bezier_crv - the bezier fit curve
        @param im_edge - edge image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @param b_debug - do debug True/False
        @returns how much the points moved"""

        # Find all the edge rectangles that have points
        ts, seg_edges = FitBezierCyl2DEdge.find_edges_hough_transform(fit_bezier_crv, im_edge,
                                                                      step_size=step_size, perc_width=perc_width,
                                                                      b_debug=b_debug)

        # im_rgb = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = fit_bezier_crv.setup_least_squares(ts)

        # Keeping all the radius guesses to fit the radius to
        radius_avg = []
        for i_seg, s in enumerate(seg_edges):
            # Left and right list of points
            s1, s2 = s

            # No Hough edges - just keep the current estimate in the LS solver
            if s1 == [] and s2 == []:
                continue

            pt_on_spine_from_left = np.zeros((1, 2))
            mid_pt_on_edge_left = np.zeros((1, 2))
            for p1, p2 in s1:
                mid_pt_edge, pt_on_spine = FitBezierCyl2DEdge._hough_edge_to_middle(fit_bezier_crv, p1, p2, ts[i_seg])
                pt_on_spine_from_left += pt_on_spine
                mid_pt_on_edge_left += mid_pt_edge

                # LineSeg2D.draw_box(im_rgb, pt_on_spine_from_left, color=(255, 0, 0), width=3)
                # LineSeg2D.draw_cross(im_rgb, mid_pt_on_edge_left, color=(0, 255, 0))

            pt_on_spine_from_right = np.zeros((1, 2))
            mid_pt_on_edge_right = np.zeros((1, 2))
            for p1, p2 in s2:
                mid_pt_edge, pt_on_spine = FitBezierCyl2DEdge._hough_edge_to_middle(fit_bezier_crv, p1, p2, ts[i_seg])
                pt_on_spine_from_right += pt_on_spine
                mid_pt_on_edge_right += mid_pt_edge
                # LineSeg2D.draw_box(im_rgb, pt_on_spine_from_right, color=(0, 255, 0), width=3)
                # LineSeg2D.draw_cross(im_rgb, mid_pt_on_edge_right, color=(0, 0, 255))

            if len(s1) > 0 and len(s2) > 0:
                # We had both a left and right fit edge - average the middle points
                #  Place the spine point at the mid-point of the two
                mid_pt_on_edge_left = mid_pt_on_edge_left / len(s1)
                mid_pt_on_edge_right = mid_pt_on_edge_right / len(s2)

                pt_on_spine = 0.5 * (mid_pt_on_edge_left + mid_pt_on_edge_right)
                if b_debug:
                    print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_on_spine}")
                b_rhs[i_seg, :] = pt_on_spine
                # LineSeg2D.draw_box(im_rgb, pt_on_spine, color=(255, 255, 255), width=5)

                radius_avg.append([0.5 * np.linalg.norm(mid_pt_on_edge_right - mid_pt_on_edge_left), ts[i_seg]])
            elif len(s1) > 0:
                # Only have the left point
                pt_on_spine_from_left = pt_on_spine_from_left / len(s1)
                mid_pt_on_edge_left = mid_pt_on_edge_left / len(s1)

                if b_debug:
                    print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_on_spine_from_left}")
                b_rhs[i_seg, :] = pt_on_spine_from_left
                # LineSeg2D.draw_box(im_rgb, pt_on_spine_from_left, color=(255, 255, 0), width=5)

                radius_avg.append([np.linalg.norm(pt_on_spine_from_left - mid_pt_on_edge_left), ts[i_seg]])
            elif len(s2) > 0:
                pt_on_spine_from_right = pt_on_spine_from_right / len(s2)
                mid_pt_on_edge_right = mid_pt_on_edge_right / len(s2)

                if b_debug:
                    print(f"rhs {b_rhs[i_seg, :]}, new pt {pt_on_spine_from_right}")
                b_rhs[i_seg, :] = pt_on_spine_from_right
                # LineSeg2D.draw_box(im_rgb, pt_on_spine_from_right, color=(255, 0, 255), width=5)
                radius_avg.append([np.linalg.norm(pt_on_spine_from_right - mid_pt_on_edge_right), ts[i_seg]])

        if len(radius_avg) > 0:
            # [(1-t), t] * [r_start, r_end]^t = radius_avg
            a_radius_constraints = np.zeros((len(radius_avg), 2))
            b_radius_rhs = np.zeros((len(radius_avg), 1))
            for i_row, (r, t) in enumerate(radius_avg):
                a_radius_constraints[i_row, 0] = 1 - t
                a_radius_constraints[i_row, 1] = t
                b_radius_rhs[i_row] = r

            new_radii, residuals, rank, _ = np.linalg.lstsq(a_radius_constraints, b_radius_rhs, rcond=None)
            if b_debug:
                print(f"Residuals {residuals}, rank {rank}")

            if b_debug:
                print(f"Radius before {fit_bezier_crv.start_radius}, {fit_bezier_crv.end_radius}")
            fit_bezier_crv.start_radius = 0.5 * (fit_bezier_crv.start_radius + new_radii[0][0])
            fit_bezier_crv.end_radius = 0.5 * (fit_bezier_crv.end_radius + new_radii[1][0])
            if b_debug:
                print(f"Radius after {fit_bezier_crv.start_radius}, {fit_bezier_crv.end_radius}")

        # cv2.imwrite("./data/forcindy/DebugImages/0_trunk_0_fit_edge.png", im_rgb)

        return fit_bezier_crv.extract_least_squares(a_constraints, b_rhs)

    @staticmethod
    def fit_bezier_crv_to_edge(bezier_crv, im_edge, params, b_debug=False):
        """ Adjust the quad to the edge image using hough transform
        @param bezier_crv: Bezier curve to start with
        @param im_edge: The edge image
        @param params: The params to use in the fit
        @param b_debug: Print debug statements y/n
        @return the quad and the params"""

        # Make a fit bezier curve - this copies the data into fit_bezier curve
        fit_bezier_crv = FitBezierCyl2D(bezier_crv)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = FitBezierCyl2DEdge._adjust_bezier_by_hough_edges(fit_bezier_crv, im_edge,
                                                                   step_size=params["step_size"], perc_width=params["width"],
                                                                   b_debug=b_debug)
            if b_debug:
                print(f"Res Hough {ret}")

        # This will get the data back out
        return fit_bezier_crv.get_copy_of_2d_bezier_curve()

    def debug_image_edge_fit(self, image_debug):
        """ Draw the fitted quad on the image
        @param image_debug - rgb image"""
        # Draw the original, the edges, and the depth mask with the fitted quad
        self.bezier_crv_fit_to_edge.draw_bezier(image_debug)
        if self.bezier_crv_fit_to_edge.is_wire():
            LineSeg2D.draw_cross(image_debug, self.bezier_crv_fit_to_edge.p0, (255, 0, 0), thickness=2, length=10)
            LineSeg2D.draw_cross(image_debug, self.bezier_crv_fit_to_edge.p2, (255, 0, 0), thickness=2, length=10)
        else:
            self.bezier_crv_fit_to_edge.draw_boundary(image_debug, 10)


if __name__ == '__main__':
    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = False
    for ind in all_files.loop_masks():
        rgb_fname = all_files.get_image_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_add_tag=True)
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            edge_fname_debug = None

        edge_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        if not exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")

        edge_crv = FitBezierCyl2DEdge(rgb_fname, edge_fname, mask_fname, edge_fname_calculate, edge_fname_debug, b_recalc=b_do_recalc)
    print("foo")
