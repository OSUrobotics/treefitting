#!/usr/bin/env python3

# a 2D quadratic Bezier with additional radius information
#  - also keeps primary orientation (left, right or up, down)
#  - Assumes this is a curve in the image; all coordinates are in image coordinates
#
# Primary code:
#   a bezier curve is defined by f(t)->[x,y] for t going from 0 to 1
#   Currently a fixed radius along the cylinder/tube
#   Using the tangent and the normal (orthogonal to tangent) we can define rectangles
#     1) Rectangles that follow the axis of the curve and cover the interior of the curve
#        - parameter is how much of the middle to cover
#     2) Rectangles that follow the boundary of the tube
#        - parameter is how much to extend in/out of the edge
#
# Also includes a lot of methods for drawing the rectangles in the image (filled or not) and also
#   "cutting out" pieces of an image given a rectangle

# TODO: Make radius be different at end points

import numpy as np
import json
import cv2
# If this doesn't load, right click on Image_based folder on the LHS and select "Mark directory as...->sources root"
#   This just lets PyCharm know that it should look in the Image_based folders for Python files
from draw_routines.image_draw_geom_utils import LineSeg2D


class BezierCyl2D:

    def __init__(self, start_pt=None, end_pt=None, radius=1, mid_pt=None):
        """ Create a bezier from 2 points or an optional mid point
        @param start_pt - start point
        @param end_pt - end point
        @param radius - width in the image
        @param mid_pt - optional mid point (otherwise set to halfway between start_pt and end_pt)"""

        if start_pt is None or end_pt is None:
            self.p0 = np.array([0, 0])
            self.p2 = np.array([1, 0])
            self.orientation = "Horizontal"
            self.p0, self.p2, self.orientation = self._orientation(np.array([0, 0]), np.array([1, 0]))
        else:
            self.p0, self.p2, self.orientation = self._orientation(np.array(start_pt), np.array(end_pt))
        if mid_pt is None:
            self.p1 = 0.5 * (self.p0 + self.p2)
        else:
            self.p1 = np.array(mid_pt)
        self.start_radius = radius
        self.end_radius = radius

    def radius(self, t):
        """ Radius is a linear interpolation of two end radii
        @param t - t between 0 and 1"""
        return (1 - t) * self.start_radius + t * self.end_radius

    def curve_length(self, t_step=0.1):
        """ Approximate length of curve
        @param t_step - t values to sample at
        @return approximate length of curve"""
        pts = self.pt_axis(np.linspace(0, 1, int(1.0 / t_step)))
        pts_diff_sq = (pts[1:, :] - pts[0:-1, :]) ** 2
        norm_sq = np.sum(pts_diff_sq, axis=1)
        return np.sum(np.sqrt(norm_sq))

    @staticmethod
    def _orientation(start_pt, end_pt):
        """Set the orientation and ensure left-right or down-up
        swaps the end points if need be
        @param start_pt current start point from BaseStatsImage
        @param end_pt current end point from BaseStatsImage
        @returns three points, orientation as a text string"""
        if abs(start_pt[1] - end_pt[1]) > abs(start_pt[0] - end_pt[0]):
            ret_orientation = "vertical"
            if start_pt[1] > end_pt[1]:
                p0 = start_pt
                p2 = end_pt
            else:
                p0 = start_pt
                p2 = end_pt
        else:
            ret_orientation = "horizontal"
            if start_pt[0] > end_pt[0]:
                p0 = start_pt
                p2 = end_pt
            else:
                p0 = start_pt
                p2 = end_pt
        return p0, p2, ret_orientation

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        pts = np.array([self.p0[i] * (1-t) ** 2 + 2 * (1-t) * t * self.p1[i] + t ** 2 * self.p2[i] for i in range(0, 2)])
        return pts.transpose()
        # return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2

    def tangent_axis(self, t):
        """ Return the tangent vec
        @param t in 0, 1
        @return 2d vec"""
        return 2 * t * (self.p0 - 2.0 * self.p1 + self.p2) - 2 * self.p0 + 2 * self.p1

    def norm_axis(self, t, dir):
        """ Normal vector (unit length
        @param t - t value along the curve (in range 0, 1)
        @param direction - 'Left' is the left direction, 'Right' is the right direction
        @return numpy array x,y """
        vec_tang = self.tangent_axis(t)
        vec_length = np.sqrt(vec_tang[0] * vec_tang[0] + vec_tang[1] * vec_tang[1])
        if dir.lower() == "left":
            return np.array([-vec_tang[1] / vec_length, vec_tang[0] / vec_length])
        return np.array([vec_tang[1] / vec_length, -vec_tang[0] / vec_length])

    def edge_pts(self, t):
        """ Return the left and right edge of the tube as points
        @param t in 0, 1
        @return 2d pts, left and right edge"""
        pt = self.pt_axis(t)
        vec = self.tangent_axis(t)
        vec_step = self.radius(t) * vec / np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        left_pt = [pt[0] - vec_step[1], pt[1] + vec_step[0]]
        right_pt = [pt[0] + vec_step[1], pt[1] - vec_step[0]]
        return left_pt, right_pt

    def edge_offset_pt(self, t, perc_in_out, direction):
        """ Go in/out of the edge point a given percentage
        @param t - t value along the curve (in range 0, 1)
        @param perc_in_out - if 1, get point on edge. If 0.5, get halfway to centerline. If 2.0 get 2 width
        @param direction - 'Left' is the left direction, 'Right' is the right direction
        @return numpy array x,y """
        pt_edge = self.pt_axis(t)
        vec_norm = self.norm_axis(t, direction)
        return pt_edge + vec_norm * (perc_in_out * self.radius(t))

    @staticmethod
    def rect_in_image(im, r, pad=2):
        """ See if the rectangle is within the image boundaries
        @im - image (for width and height)
        @r - the rectangle
        @pad - a bit of padding -make sure the rectangle is not within pad of image boundary
        @return True or False"""
        if np.min(r) < pad:
            return False
        if np.max(r[:, 0]) > im.shape[1] + pad:
            return False
        if np.max(r[:, 1]) > im.shape[0] + pad:
            return False
        return True

    def _rect_corners(self, t1, t2, perc_width=0.3):
        """ Get two rectangles covering the expected left/right edges of the cylinder/tube
        @param t1 starting t value
        @param t2 ending t value
        @param perc_width How much of the radius to move in/out of the edge
        @returns two rectangles"""
        vec_ts = self.tangent_axis(0.5 * (t1 + t2))
        edge_left1, edge_right1 = self.edge_pts(t1)
        edge_left2, edge_right2 = self.edge_pts(t2)

        vec_step = perc_width * self.radius(t1) * vec_ts / np.sqrt(vec_ts[0] * vec_ts[0] + vec_ts[1] * vec_ts[1])
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

        vec_step = perc_width * self.radius(t1) * vec_ts / np.sqrt(vec_ts[0] * vec_ts[0] + vec_ts[1] * vec_ts[1])
        rect = np.array([[pt1[0] + vec_step[1], pt1[1] - vec_step[0]],
                         [pt2[0] + vec_step[1], pt2[1] - vec_step[0]],
                         [pt2[0] - vec_step[1], pt2[1] + vec_step[0]],
                         [pt1[0] - vec_step[1], pt1[1] + vec_step[0]]], dtype="float32")
        return rect

    def boundary_rects(self, step_size=40, perc_width=0.3, offset=False):
        """ Get a set of rectangles covering the left/right expected edges of the cylinder/tube
           March along the edges at the given image step size and produce rectangles in pairs
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        @param offset - if True, start at 0.5 of step_size and end at 1-0.5
        @returns a list of pairs of left,right rectangles - evens are left, odds right"""

        t_step = self._time_step_from_im_step(step_size)
        n_boxes = int(max(1.0, 1.0 / t_step))
        t_step_exact = 1.0 / n_boxes
        rects = []
        ts = []
        t_start = 0
        t_end = 1
        if offset:
            t_start = 0.5 * t_step_exact
            t_end = 1.0 - 0.5 * t_step_exact
        for t in np.arange(t_start, t_end, step=t_step_exact):
            rect_left, rect_right = self._rect_corners(t, t + t_step_exact, perc_width=perc_width)
            rects.append(rect_left)
            rects.append(rect_right)
            ts.append(t + 0.5 * t_step_exact)
            ts.append(t + 0.5 * t_step_exact)
        return rects, ts

    def interior_rects(self, step_size=40, perc_width=0.3):
        """ March along the interior of the tube and produce one rectangle for approximately step_size image pixels
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        @return a list of rectangles covering the interior
        """
        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(1.0 / t_step))
        t_step_exact = 1.0 / n_boxes
        rects = []
        ts = []
        for t in np.arange(0, 1.0, step=t_step_exact):
            rect = self._rect_corners_interior(t, t + t_step_exact, perc_width=perc_width)
            rects.append(rect)
            ts.append(t + 0.5 * t_step_exact)
        return rects, ts

    def interior_rects_mask(self, image_shape, step_size=40, perc_width=0.3):
        """ Overlay the interior rectangles on the image and set any pixels in the interior of the rectangle
         to be one. Essentially makes a mask of the quad
        @param image_shape - shape of image to fill mask with
        @param step_size how many pixels to cover with each rectangle
        @param perc_width How much of the radius to move in/out of the edge. 0.5 will cover entire cylinder
        @return image with pixels set to 256 where quad covers them
        """
        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(1.0 / t_step))
        t_step_exact = 1.0 / n_boxes
        ret_im_mask = np.zeros(image_shape, dtype=bool)
        for t in np.arange(0, 1.0, step=t_step_exact):
            rect = self._rect_corners_interior(t, t + t_step_exact, perc_width=perc_width)
            self.draw_rect_filled(ret_im_mask, rect)

        return ret_im_mask

    @staticmethod
    def image_cutout(im, rect, step_size, height):
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

    def _time_step_from_im_step(self, step_size):
        """ How far to step along the curve to step that far in the image
        @param step_size how many pixels to use in the box
        @return delta t to use"""
        crv_length = np.sqrt(np.sum((self.p2 - self.p0) ** 2))
        return min(1, step_size / crv_length)

    def is_wire(self):
        """Determine if this is likely a wire (long, narrow, straight, and thin)
        @return True/False
        """
        rad_clip = 3
        if self.end_radius > rad_clip or self.start_radius > rad_clip:
            return False

        line_axis = LineSeg2D(self.p0, self.p2)
        pt_proj, _ = line_axis.projection(self.p1)

        dist_line = np.linalg.norm(self.p1 - pt_proj)
        if dist_line > rad_clip:
            return False

        return True

    def draw_bezier(self, im):
        """ Set the pixels corresponding to the quad to white
        @im numpy array as image"""
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

    def draw_boundary(self, im, step_size=10):
        """ Draw the edge boundary"""
        t_step = self._time_step_from_im_step(step_size)
        max_n = max(2, int(1.0 / t_step))
        edge_pts_draw = [self.edge_pts(t) for t in np.linspace(0, 1, max_n)]
        col_start = 125
        col_div = 120 // max_n
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

    @staticmethod
    def draw_edge_rect(im, rect, col=(50, 255, 255)):
        """ Draw a rectangle in the image
        @param im - the image
        @param col - rgb color as triple 0-255
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

    @staticmethod
    def draw_rect_filled(im, rect, col=(50, 255, 255)):
        """ Fill in the rectangle in the image
        @param im - the image
        @param rect - the rect as a 4x2 np array
        @param col - the color to use
        """
        points = np.int32(rect)
        cv2.fillPoly(im, pts=[points], color=col)

    def draw_edge_rects(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.boundary_rects(step_size, perc_width)
        col_incr = 255 // len(rects)
        for i, r in enumerate(rects):
            col = (i * col_incr, 100 + (i % 2) * 100, i * col_incr)
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
                  s1 * (1 - t) * r[1] +
                  s1 * t * r[2] +
                  (1-s1) * t * r[3])
            p2 = ((1-s2) * (1-t) * r[0] +
                  s2 * (1 - t) * r[1] +
                  s2 * t * r[2] +
                  (1-s2) * t * r[3])
            if i % 2:
                LineSeg2D.draw_line(im, p1, p2, color=col_left, thickness=2)
            else:
                LineSeg2D.draw_line(im, p1, p2, color=col_right, thickness=2)

    def draw_interior_rects(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.interior_rects(step_size, perc_width)
        col_incr = 255 // len(rects)
        for i, r in enumerate(rects):
            col = (i * col_incr, 100 + (i % 2) * 100, i * col_incr)
            self.draw_edge_rect(im, r, col=col)

    def draw_interior_rects_filled(self, im, b_solid=True, col_solid=(255, 255, 255), step_size=40, perc_width=0.5):
        """ Draw the edge rectangles
        @param im - the image
        @param b_solid - use a solid color or alternate in order to see rects and order
        @param col_solid - the solid color to use.
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.interior_rects(step_size, perc_width)
        col_incr = 128 // len(rects)
        for i, r in enumerate(rects):
            if b_solid:
                col = col_solid
            else:
                col = (128 + i * col_incr, 100 + (i % 2) * 100, 128 + i * col_incr)
            self.draw_rect_filled(im, r, col=col)

    def draw_boundary_rects_filled(self, im, b_solid=True, col_solid=(255, 255, 255), step_size=40, perc_width=0.5):
        """ Draw the edge rectangles filled
        @param im - the image
        @param b_solid - use a solid color or alternate in order to see rects and order
        @param col_solid - the solid color to use.
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        rects, _ = self.boundary_rects(step_size, perc_width)
        col_incr = 128 // len(rects)
        for i, r in enumerate(rects):
            if b_solid:
                col = col_solid
            else:
                col = (128 + i * col_incr, 100 + (i % 4) * 50, 128 + i * col_incr)
            self.draw_rect_filled(im, r, col=col)

    def make_mask_image(self, im_mask, step_size=20, perc_fuzzy=0.2):
        """ Create a mask that is white in the middle, grey along the boundaries
        @param im_mask - the image
        @param step_size how many pixels to move along the boundary
        @param perc_fuzzy How much of the boundary to make fuzzy
        """
        self.draw_interior_rects_filled(im_mask, b_solid=True,
                                        col_solid=(255, 255, 255),
                                        step_size=step_size,
                                        perc_width=1.0)
        self.draw_boundary_rects_filled(im_mask, b_solid=True,
                                        col_solid=(128, 128, 128),
                                        step_size=step_size,
                                        perc_width=perc_fuzzy)

    def write_json(self, fname):
        """Convert to array and write out
        @param fname file name to write to"""
        fix_nparray = []
        for k, v in self.__dict__.items():
            try:
                if v.size > 1:
                    fix_nparray.append([k, v])
                    setattr(self, k, [float(x) for x in v])
            except AttributeError:
                pass

        with open(fname, "w") as f:
            json.dump(self.__dict__, f, indent=2)

        for fix in fix_nparray:
            setattr(self, fix[0], fix[1])

    @staticmethod
    def read_json(fname, bezier_crv=None):
        """ Read back in from json file
        @param fname file name to read from
        @param bezier_crv - an existing bezier curve to put the data in"""
        with open(fname, 'r') as f:
            my_data = json.load(f)
            if not bezier_crv:
                bezier_crv = BezierCyl2D([0, 0], [1, 1], 1)
            for k, v in my_data.items():
                try:
                    if len(v) == 2:
                        setattr(bezier_crv, k, np.array(v))
                    else:
                        setattr(bezier_crv, k, v)
                except TypeError:
                    setattr(bezier_crv, k, v)

        return bezier_crv


if __name__ == '__main__':
    # Make a horizontal curve
    bezier_crv_horiz = BezierCyl2D([10, 130], [620, 60], 40, [320, 190])
    assert(bezier_crv_horiz.orientation == "horizontal")
    assert(not bezier_crv_horiz.is_wire())
    # TODO set the two radii to be different and check that it renders corectly
    # Make a vertical curve
    bezier_crv_vert = BezierCyl2D([320, 30], [290, 470], 40, [310, 210])
    assert(bezier_crv_vert.orientation == "vertical")
    assert(not bezier_crv_vert.is_wire())

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2)
    perc_width_interior = 0.5
    perc_width_edge = 0.2
    for i_row, crv in enumerate([bezier_crv_horiz, bezier_crv_vert]):
        im_debug = np.zeros((480, 640, 3), np.uint8)
        crv.draw_bezier(im_debug)
        crv.draw_boundary(im_debug)
        crv.draw_edge_rects(im_debug, step_size=40, perc_width=perc_width_edge)
        crv.draw_interior_rects(im_debug, step_size=40, perc_width=perc_width_interior)
        axs[0, i_row].imshow(im_debug)
        axs[0, i_row].set_title(crv.orientation)

        im_debug = np.zeros((480, 640, 3), np.uint8)
        crv.draw_bezier(im_debug)
        crv.draw_boundary(im_debug)
        crv.draw_interior_rects_filled(im_debug, b_solid=False, step_size=40, perc_width=perc_width_interior)
        axs[1, i_row].imshow(im_debug)
        axs[1, i_row].set_title(crv.orientation + f" filled {perc_width_interior}")

        im_debug = np.zeros((480, 640, 3), np.uint8)
        crv.make_mask_image(im_debug, perc_fuzzy=0.25)
        axs[2, i_row].imshow(im_debug)
        axs[2, i_row].set_title(crv.orientation + f" mask 0.25")

        fname_test = "./data/test_crv.json"
        crv.write_json(fname_test)

        read_back_in_crv = BezierCyl2D.read_json(fname_test)
    plt.tight_layout()
    plt.show()

    print("Done")
