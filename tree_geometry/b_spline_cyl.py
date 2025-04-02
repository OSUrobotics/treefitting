#!/usr/bin/env python3
"""
b_spline_cyl.py
Author: Luke Strohbehn
"""

import numpy as np
from typing import Union
from tree_geometry.b_spline_curve import BSplineCurve


# Add a radius to a BSpline curve
#  If points are 3D, then, supports generating a mesh
#. Radius is a quadratic b spline with one segment (3 points)
class BSplineCyl(BSplineCurve):
    def __init__(self, ctrl_pts: Union[list[np.ndarray], np.ndarray, list[list]], degree: str = "quadratic", radii: Union[float, list[float]] = 1.0) -> None:
        """BSpline with radii initialization
        :param ctrl_pts: control points, list of numpy array points of desired dimension
        :param degree: degree of spline, defaults to "quadratic"
        :param radii: radii, either a single radii value for the whole curve or a list of radii values
        """
        super().__init__(ctrl_pts=ctrl_pts, degree=degree)

        # The spline curve for the radii
        radii_pts = [[1.0], [1.0], [1.0]]

        if isinstance(radii, float):
            for ind in range(0, 3):
                radii_pts[ind][0] = radii
        elif len(radii) == 2:
            radii_pts[0][0] = radii[0]
            radii_pts[1][0] = 0.5 * (radii[0] + radii[1])
            radii_pts[2][0] = radii[1]
        else:
            # This should work for 1, 3, and > 3
            radii_pts[0][0] = radii[0]
            radii_pts[1][0] = radii[int(len(radii) / 2)]
            radii_pts[2][0] = radii[-1]

        # Quadratic, 1d curve
        self.radii_crv = BSplineCurve(radii_pts, "quadratic")

    def radius(self, t):
        """Return radius at a point t along the spline
        @param t - the t value in 0, max_t"""
        t_radii = t / self.max_t()
        return self.radii_crv.eval_crv(t_radii)[0]

    def edge_pts(self, t, perc_in_out=1.0):
        """ 
        Return the left and right edge of the tube as points
        If 2D, returns points along edge (+- norm * radius)
        If 3D, returns points along the bi-norm as well,
        :param t: parameter
        :param perc_in_out: percentage of radius in_out (1.0 gives on edge)
        :return: 2d pts, left and right edge
        """
        pt = self.eval_crv(t)
        vec = self.eval_norm(t)
        radius = perc_in_out * self.radius(t)

        return pt + vec * radius, pt - vec * radius


if __name__ == "__main__":

    # Need n+1 control points
    cntrl_hull = [[1, 1], [2, 1], [3, -1]]
    crv1 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=0.1)
    crv2 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=[0.1, 0.3])
    crv3 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=[0.1, 0.3, 0.6])

