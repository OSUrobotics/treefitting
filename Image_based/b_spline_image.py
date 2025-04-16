#!/usr/bin/env python3

# A 2D B-spline curve (tree_geometry/b_spline_cyl) in an image
#  - also keeps primary orientation (left-right or up-down)
#  - Assumes this is a curve in the image; all coordinates are in image coordinates
#
# Primary code:
#   A bspline curve is defined by f(t)->[x,y] for t going from 0 to 1
#   Radius changes quadtratically along the branch
#   Using the tangent and the normal (orthogonal to tangent) we can define rectangles
#     1) Rectangles that follow the axis of the curve and cover the interior of the curve
#        - parameter is how much of the middle to cover
#     2) Rectangles that follow the boundary of the tube
#        - parameter is how much to extend in/out of the edge
#
# Also includes a lot of methods for drawing the rectangles in the image (filled or not) and also
#   "cutting out" pieces of an image given a rectangle
#
# Note: This code is largely copied from bezier_cyl_2d.py

import numpy as np
from typing import Union
import cv2
# If this doesn't load, right click on Image_based folder on the LHS and select "Mark directory as...->sources root"
#   This just lets PyCharm know that it should look in the Image_based folders for Python files
from tree_geometry.b_spline_cyl import BSplineCyl
#from draw_routines.image_draw_geom_utils import LineSeg2D



class BSplineCylImage(BSplineCyl):
    def __init__(self, ctrl_pts: Union[list[np.ndarray], np.ndarray, list[list]], degree: str = "quadratic", radii: Union[float, list[float]] = 1.0) -> None:
        """BSpline with radii initialization
        :param ctrl_pts: control points, list of numpy array points of desired dimension
        :param degree: degree of spline, defaults to "quadratic"
        :param radii: radii, either a single radii value for the whole curve or a list of up to 3 radii values
        """
        super().__init__(ctrl_pts=ctrl_pts, degree=degree, radii=radii)

    def reverse_direction(self):
        """ Reverse direction of curve"""
        super().reverse_direction()

    def orientation(self):
        """Return the orientation
        @returns -1,0 if right-left, 1,0 if left-right, 0,1 if down-up, 0,-1 if up_down, +-1,+-1 if mostly diagonal"""
        start_pt = self.eval_crv(0.0)
        end_pt = self.eval_crv(self.max_t())
        dir = end_pt - start_pt
        ang = np.arctan2(dir[1], dir[0])
        ret_orientation_x = 1
        if dir[0] < 0.0:
            ret_orientation_x = -1
        ret_orientation_y = 1
        if dir[1] < 0.0:
            ret_orientation_y = -1
        if np.fabs(np.fabs(ang) - np.pi / 4.0) < np.pi / 8.0 or np.fabs(np.fabs(ang) - 3.0 * np.pi / 4.0) < np.pi / 8.0:
            # Diagonal
            return (ret_orientation_x, ret_orientation_y)
        if np.fabs(ang) < np.pi / 8.0 or np.fabs(np.fabs(ang) - np.pi) < np.pi / 8.0:
            # Horizontal
            return (ret_orientation_x, 0)
        return (0, ret_orientation_y)

    def orient_left_right_down_up(self):
        """ make curve have increasing x and/or increasing y"""
        orient = self.orientation()
        if orient[0] < 0 and orient[1] == 0:
            self.reverse_direction()
        elif orient[1] < 0 and orient[0] == 0:
            self.reverse_direction()
        elif orient[0] < 0 and orient[1] < 0:
            self.reverse_direction()

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

    def _rect_corners_interior(self, t1, t2, perc_width=0.3):
        """ Get a rectangle covering the expected interior of the cylinder
           Rectangle starts at t = t1, left side, and goes clockwise
        @param t1 starting t value
        @param t2 ending t value
        @param perc_width How much of the radius to move in/out of the edge
        @returns rectangle as 4x2 numpy array"""
        edge_left_inner = self.edge_pts(np.linspace(t1, t2, 2), perc_width)
        edge_right_inner = self.edge_pts(np.linspace(t1, t2, 2), -1.0 + perc_width)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0, :] = edge_left_inner[0]
        rect[1, :] = edge_left_inner[1]
        rect[2, :] = edge_right_inner[1]
        rect[3, :] = edge_right_inner[0]

        return rect

    def boundary_rects(self, step_size=40, perc_width=0.3, offset=False):
        """ Get a set of rectangles covering the left/right expected edges of the cylinder/tube
           March along the edges at the given image step size and produce rectangles in pairs
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        @param offset - if True, start at 0.5 of step_size and end at 1-0.5
        @returns a list of pairs of left,right rectangles - evens are left, odds right"""

        t_step = self._time_step_from_im_step(step_size)
        n_boxes = int(max(self.max_t(), 1.0 / t_step))
        t_step_exact = self.max_t() / n_boxes
        left_rects = []
        right_rects = []
        t_start = 0
        t_end = self.max_t()
        if offset:
            t_start = 0.5 * t_step_exact
            t_end = self.max_t() - 0.5 * t_step_exact

        ts = np.linspace(0.0, self.max_t(), n_boxes)

        left_edge_pts_outer = self.edge_pts(ts, 1.0 + perc_width)
        left_edge_pts_inner = self.edge_pts(ts, 1.0 - perc_width)
        right_edge_pts_outer = self.edge_pts(ts, -1.0 - perc_width)
        right_edge_pts_inner = self.edge_pts(ts, -1.0 + perc_width)
        for i in range(0, len(left_edge_pts_inner) - 1):
            # go clockwise, starting with the t = t, inner point
            left_rect = [left_edge_pts_inner[i], left_edge_pts_outer[i], left_edge_pts_outer[i+1], left_edge_pts_inner[i+1]]
            right_rect = [right_edge_pts_inner[i], right_edge_pts_inner[i+1], right_edge_pts_outer[i+1], right_edge_pts_outer[i]]

            left_rects.append(left_rect)
            right_rects.append(right_rect)
        return left_rects, right_rects, ts

    def interior_rects(self, step_size=40, perc_width=0.3):
        """ March along the interior of the tube and produce one rectangle for approximately step_size image pixels
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        @return a list of rectangles covering the interior
        """
        t_step = self._time_step_from_im_step(step_size)
        n_boxes = max(1, int(self.max_t() / t_step))
        t_step_exact = self.max_t() / n_boxes
        rects = []

        ts = np.linspace(0.0, self.max_t(), n_boxes)
        left_edge_pts = self.edge_pts(ts,  perc_width)
        right_edge_pts = self.edge_pts(ts, -perc_width)
        for i in range(0, ts.shape[0]-1):
            # go clockwise, starting with the t = t, left point
            rect = [left_edge_pts[i], left_edge_pts[i+1], right_edge_pts[i+1], right_edge_pts[i]]
            rects.append(rect)
        return rects, ts

    def interior_rects_mask(self, image_shape, step_size=40, perc_width=0.3):
        """ Overlay the interior rectangles on the image and set any pixels in the interior of the rectangle
         to be one. Essentially makes a mask of the quad
        @param image_shape - shape of image to fill mask with
        @param step_size how many pixels to cover with each rectangle
        @param perc_width How much of the radius to move in/out of the edge. 0.5 will cover entire cylinder
        @return image with pixels set to 256 where quad covers them
        """
        rects, _ = self.interior_rects(step_size=step_size, perc_width=perc_width)

        ret_im_mask = np.zeros(image_shape, dtype=bool)
        for r in rects:
            self.draw_rect_filled(ret_im_mask, r)

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
        crv_length = super().curve_length()
        return min(1, step_size / crv_length)

    def draw_curve(self, im):
        """ Set the pixels corresponding to the axis to grey, going from dark grey to white
        @im numpy array as image"""
        n_pts_quad = 6
        pts = self.eval_crv(np.linspace(0, 1, n_pts_quad))
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
        max_n = max(2, int(self.max_t() / t_step) + 1)

        ts = np.linspace(0, self.max_t(), max_n)
        edge_pts = self.edge_pts(t=ts, perc_in_out=1.0)
        for dir in [-1, 1]:
            edge_pts_draw = [self.edge_pts(t=t, perc_in_out=dir) for t in np.linspace(0, self.max_t(), max_n)]
            col_start = 125
            col_div = 120 // max_n
            for i, (p1, p2) in enumerate(zip(edge_pts_draw[0:-1], edge_pts_draw[1:])):
                cv2.line(im, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (220 - i * 100, col_start, 20 + i * 100), thickness=2)
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
        left_rects, right_rects, _ = self.boundary_rects(step_size, perc_width)
        col_incr = 255 // len(left_rects)
        for i, (lr, rr) in enumerate(zip(left_rects, right_rects)):
            left_col = (i * col_incr, 100, i * col_incr)
            self.draw_edge_rect(im, lr, col=left_col)
            right_col = (i * col_incr, 200, i * col_incr)
            self.draw_edge_rect(im, rr, col=right_col)

    def draw_edge_rects_markers(self, im, step_size=40, perc_width=0.3):
        """ Draw the edge rectangles
        @param im - the image
        @param step_size how many pixels to move along the boundary
        @param perc_width How much of the radius to move in/out of the edge
        """
        left_rects, right_rects, _ = self.boundary_rects(step_size, perc_width)
        s1 = 0.25
        s2 = 0.5
        t = 0.25
        col_left = (200, 200, 125)
        col_right = (250, 250, 250)
        left_rects.extend(right_rects)
        for i, r in enumerate(left_rects):
            p1 = ((1-s1) * (1-t) * r[0] +
                  s1 * (1 - t) * r[1] +
                  s1 * t * r[2] +
                  (1-s1) * t * r[3])
            p2 = ((1-s2) * (1-t) * r[0] +
                  s2 * (1 - t) * r[1] +
                  s2 * t * r[2] +
                  (1-s2) * t * r[3])
            if i > len(right_rects):
                col = col_left
            else:
                col = col_right
            cv2.line(im, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=col, thickness=2)

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
        left_rects, right_rects, _ = self.boundary_rects(step_size, perc_width)
        col_incr = 128 // len(left_rects)
        for i, (lr, rr) in enumerate(zip(left_rects, right_rects)):
            if b_solid:
                col = col_solid
            else:                
                col = (128 + i * col_incr, 100 + (i % 2) * 50, 128 + i * col_incr)
            self.draw_rect_filled(im, lr, col=col)
            self.draw_rect_filled(im, rr, col=col)

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


if __name__ == '__main__':
    # Make a horizontal curve 480x640
    im_width = 320
    im_height = 240
    bspline_crv_horiz = BSplineCylImage(ctrl_pts=[[0.1 * im_width, 0.4 * im_height], [0.5 * im_width, 0.6 * im_height], [0.8 * im_width, 0.5 * im_height]], radii=[40, 60])
    assert bspline_crv_horiz.orientation() == (1, 0)

    # Make a vertical curve
    bspline_crv_vert = BSplineCylImage(ctrl_pts=[[0.4 * im_width, 0.1 * im_height], [0.6 * im_width, 0.5 * im_height], [0.5 * im_width, 0.8 * im_height]], radii=[40, 60])
    assert bspline_crv_vert.orientation() == (0, 1)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3)
    perc_width_interior = 0.5
    perc_width_edge = 0.2
    step_size = im_width / 10

    for i_row, crv in enumerate([bspline_crv_horiz, bspline_crv_vert]):
        im_debug = np.zeros((im_height, im_width, 3), np.uint8)
        crv.draw_curve(im_debug)
        crv.draw_boundary(im_debug, step_size=step_size)
        crv.draw_edge_rects(im_debug, step_size=step_size, perc_width=perc_width_edge)
        crv.draw_boundary(im_debug, step_size=step_size)
        crv.draw_interior_rects(im_debug, step_size=step_size, perc_width=perc_width_interior)
        axs[i_row, 0].imshow(im_debug)
        axs[i_row, 0].set_title(f"{crv.orientation()}")

        im_debug = np.zeros((im_height, im_width, 3), np.uint8)
        crv.draw_boundary(im_debug)
        crv.draw_interior_rects_filled(im_debug, b_solid=False, step_size=step_size, perc_width=perc_width_interior)
        crv.draw_curve(im_debug)
        axs[i_row, 1].imshow(im_debug)
        axs[i_row, 1].set_title(f"{crv.orientation()} filled {perc_width_interior}")

        im_debug = np.zeros((im_height, im_width, 3), np.uint8)
        crv.make_mask_image(im_debug, perc_fuzzy=0.25)
        crv.draw_curve(im_debug)
        axs[i_row, 2].imshow(im_debug)
        axs[i_row, 2].set_title(f"{crv.orientation()} mask 0.25")

    plt.tight_layout()
    plt.show()

    print("Done")
