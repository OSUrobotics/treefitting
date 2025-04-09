#!/usr/bin/env python3
"""
b_spline_cyl.py
Author: Cindy Grimm
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
        :param radii: radii, either a single radii value for the whole curve or a list of up to 3 radii values
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

    def reverse_direction(self):
        """ Reverse the direction of the curve"""
        self.radii_crv.reverse_direction()
        super().reverse_direction()

    def radius(self, t: Union[float, np.ndarray]):
        """Return radius at a point t along the spline
        @param t - the t value in 0, max_t"""
        t_radii = t / self.max_t()
        return self.radii_crv.eval_crv(t_radii)[0]

    def edge_pts(self, t: Union[float, np.ndarray], perc_in_out : float=1.0):
        """ 
        Return the left and right edge of the tube as points
        If 2D, returns points along edge (+- norm * radius)
        If 3D, returns points along the bi-norm as well,
        :param t: parameter
        :param perc_in_out: percentage of radius in_out (1.0 gives on edge, negative values give right points, positive left)
        :return: 2d pts along edge
        """
        pt = self.eval_crv(t)
        vec = self.eval_norm(t)
        radius = perc_in_out * self.radius(t)

        return pt + vec * radius

    def write_json(self):
        """Create a dictionary and return it"""
        my_dict = {"Name": "BSplineCyl", "radius": self.radii_crv.write_json(), "bsplinecrv": super().write_json()}

        return my_dict

    @staticmethod
    def read_json(json_dict, bspline_cyl_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param control_hull_instance - an existing of points list to put the data in"""
        if json_dict["Name"] != "BSplineCyl":
            raise ValueError(f"This is not a bspline cylinder dictionary {json_dict}")

        if not bspline_cyl_instance:
            degree_name_list = ["None", "linear", "quadratic", "cubic"]
            degree_name = degree_name_list[json_dict["bsplinecrv"]["degree"]]
            radius_crv = BSplineCurve.read_json(json_dict["radius"])
            bspline_cyl_instance = BSplineCyl(ctrl_pts=json_dict["bsplinecrv"]["crv_pts"]["cntrl_hull_pts"]["pts"],
                                              degree=degree_name,
                                              radii=radius_crv.points())
        else:
            bspline_cyl_instance.set_points(json_dict["bsplinecrv"]["crv_pts"]["cntrl_hull_pts"]["pts"])
            assert bspline_cyl_instance.degree() == json_dict["bsplinecrv"]["degree"]
            BSplineCurve.read_json(json_dict["radius"], bspline_cyl_instance.radii_crv)
        # Check
        bspline_cyl_instance.internal_check()
        return bspline_cyl_instance

    def internal_check(self):
        """ Check that all the data lines up"""
        assert self.radii_crv.internal_check()
        assert super().internal_check()
        return True


if __name__ == "__main__":

    # Need n+1 control points
    cntrl_hull = [[1, 1], [2, 1], [3, -1]]
    crv1 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=0.1)
    crv2 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=[0.1, 0.3])
    crv3 = BSplineCyl(ctrl_pts=cntrl_hull, degree="quadratic", radii=[0.1, 0.3, 0.6])

    # Just checking that no syntax error
    crv1.edge_pts(t=0.0, perc_in_out=1.0)
    ts_check = np.linspace(0.0, 1.0, 3)
    crv1.edge_pts(t=ts_check, perc_in_out=1.0)

    # Make a straight line and check left-right points
    cntrl_hull_straight = [[0, 0], [1, 0], [2, 0]]
    cntrl_hull_straight_radius = 0.1
    crv_straight = BSplineCyl(ctrl_pts=cntrl_hull_straight, degree="linear", radii=cntrl_hull_straight_radius)
    edge_pts_left = crv_straight.edge_pts(t=np.linspace(0.0, crv_straight.max_t(), 3), perc_in_out=1.0)
    edge_pts_right = crv_straight.edge_pts(t=np.linspace(0.0, crv_straight.max_t(), 3), perc_in_out=-0.5)
    assert np.isclose(edge_pts_left[0][0], 0.0)
    assert np.isclose(edge_pts_left[1][0], 1.0)
    assert np.isclose(edge_pts_left[2][0], 2.0)
    assert np.isclose(edge_pts_right[0][0], 0.0)
    assert np.isclose(edge_pts_right[1][0], 1.0)
    assert np.isclose(edge_pts_right[2][0], 2.0)
    for ind in range(0, edge_pts_left.shape[0]):
        assert np.isclose(edge_pts_left[ind][1], cntrl_hull_straight_radius)
        assert np.isclose(edge_pts_right[ind][1], cntrl_hull_straight_radius * -0.5)

    import json
    fname = "../Image_based/data/test_bspline_cyl.txt"
    with open(fname, "w") as f:
        json.dump(crv_straight.write_json(), f, indent=2)

    with open(fname, 'r') as f:
        my_data = json.load(f)

        check_read = BSplineCyl.read_json(my_data)

        assert check_read.n_points() == crv_straight.n_points()
        assert check_read.degree() == crv_straight.degree()
        for ind in range(0, check_read.n_points()):
            assert np.all(np.isclose(check_read.point(ind), crv_straight.point(ind)))

        for ind in range(0, check_read.radii_crv.n_points()):
            assert np.all(np.isclose(check_read.radii_crv.point(ind), crv_straight.radii_crv.point(ind)))

        # Checking no syntax errors
        BSplineCyl.read_json(my_data, check_read)
