#!/usr/bin/env python3

# Fit a Bezier cylinder using least squares
#  Adds least squares fit to bezier_cyl_2d
#  Essentially, chunk up the mask into pieces, find the average center, then set up a LS fit that (gradually)
#    moves the center by using each chunk's recommendation for where the center should be

import numpy as np
from bezier_cyl_2d import BezierCyl2D
from line_seg_2d import LineSeg2D


class FitBezierCyl2D(BezierCyl2D):
    def __init__(self, bezier_crv_start=None):
        """ Read in the mask image, use the stats to start the quad fit, then fit the quad
        @param bezier_crv_start: Curve to copy to start with
        """

        # This is a bit of a weird way to do this, but the assumption is that we will be initializing this
        #   with the curve that we got from the previous step (eg, the one from the stats for fitting to the
        #   mask). So we want to copy, not =, so that we won't change the values in the original curve
        # We're assuming we are not going to make one of these from scratch
        if bezier_crv_start:
            for k, v in bezier_crv_start.__dict__.items():
                try:
                    setattr(self, k, v.copy())
                except AttributeError:
                    setattr(self, k, v)
        else:
            super(BezierCyl2D, self).__init__()

    def setup_least_squares(self, ts):
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

    def extract_least_squares(self, a_constraints, b_rhs):
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

    def set_end_pts(self, pt0, pt2):
        """ Set the end point to the new end point while trying to keep the curve the same
        @param pt0 new p0
        @param pt2 new p2"""
        l0 = LineSeg2D(self.p0, self.p1)
        l2 = LineSeg2D(self.p1, self.p2)
        _, t0 = l0.projection(pt0)
        _, t2 = l2.projection(pt2)

        ts_mid = np.array([0.25, 0.75])
        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = self.setup_least_squares(ts_mid)
        b_rhs[-3, :] = pt0.transpose()
        b_rhs[-2, :] = self.pt_axis(0.5 * (t0 + t2))
        b_rhs[-1, :] = pt2.transpose()
        for i, t in enumerate(ts_mid):
            t_map = (1-t) * t0 + t * t2
            b_rhs[i, :] = self.pt_axis(t_map)

        return self.extract_least_squares(a_constraints, b_rhs)


if __name__ == '__main__':
    # Make a horizontal curve
    bezier_crv_horiz = BezierCyl2D([50, 230], [600, 60], 40, [310, 190])
    assert(bezier_crv_horiz.orientation == "horizontal")
    assert(not bezier_crv_horiz.is_wire())
    # Make a vertical curve
    bezier_crv_vert = BezierCyl2D([320, 60], [290, 410], 40, [310, 210])
    assert(bezier_crv_vert.orientation == "vertical")
    assert(not bezier_crv_vert.is_wire())

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    perc_width_interior = 0.5
    perc_width_edge = 0.2
    for i_row, crv in enumerate([bezier_crv_horiz, bezier_crv_vert]):
        im_debug = np.zeros((480, 640, 3), np.uint8)
        crv_fit = FitBezierCyl2D(crv)
        crv_fit.draw_bezier(im_debug)
        crv_fit.draw_boundary(im_debug)
        axs[0, i_row].imshow(im_debug)
        axs[0, i_row].set_title(crv.orientation)

        crv_fit.set_end_pts(crv.p0 - np.array([20, 20]), crv.p2 + np.array([20, 20]))
        im_debug = np.zeros((480, 640, 3), np.uint8)
        crv_fit.draw_bezier(im_debug)
        crv_fit.draw_boundary(im_debug)
        axs[1, i_row].imshow(im_debug)
        axs[1, i_row].set_title(crv.orientation + f" end pts moved")

    plt.tight_layout()

    print("Done")
