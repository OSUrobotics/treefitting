#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from tree_geometry.line_segs import LineSeg
from typing import Union

# A control hull is a list of points
from tree_geometry.point_lists import PointList

# ControlHull - a polyline, more or less
#  - keeps points, line segments, and can return points as a dim X numpoints numpy array
# Supports
#   - Adding points/segments
#   - Projecting onto the hull
#   - Length of hull
class ControlHull(PointList):
    def __init__(self, initial_points: Union[np.ndarray, list[np.ndarray], list[list]]):
        """ A control hull has to have at least two points
        @param initial_points: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""

        # Keep these three forms of the points
        # - a list of numpy arrays of dimension dim
        # - the points as an n points x dim numpy array
        # - LineSeg2D for each edge

        self._polylines = None

        super().__init__(initial_points)

        self._set_polylines()

    def __deepcopy__(self, memo):
        """Deep copy constructor for ControlHull """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Will end up calling set_points, which does a deep copy
        return ControlHull(self._points)

    def reverse_direction(self):
        """ Reverse the order of the control points"""
        super().reverse_direction()
        self._set_polylines()

    def polylines(self):
        """ If there are n points, then there are n-1 LineSeg2D's """
        return self._polylines

    def _set_polylines(self):
        """ Assumes the points are set and just does the line segs"""
        self._polylines = [LineSeg(self._points[i], self._points[i + 1]) for i in range(len(self._points) - 1)]

    def set_points(self, new_pts: Union[np.ndarray, list[np.ndarray], list[list]]):
        """ Set the control null from a list of numpy arrays (or list of lists)
        Note: This does a deep copy, not a shallow one
        @param new_pts: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""
        super().set_points(new_pts)

        if self.n_points() < 2:
            raise ValueError("ControlHull: Need at least two points, got {self.n_points()}")

        self._set_polylines()

    def add_point(self, pt: Union[np.ndarray, list]):
        """ add the point to the list, along with a new polyline
        @param pt: the point to add"""
        super().add_point(pt)
        self._polylines.append(LineSeg(self._points[-2], self._points[-1]))

    def hull_length(self):
        """ Length of hull
        @return sum of lengths of line segments"""
        seg_lengths = [line_i.line_length() for line_i in self._polylines]
        return np.sum(np.array(seg_lengths))

    def project_on_hull(self, point):
        dist = 1e30
        min_t = 0.0
        min_seg = -1
        min_proj = None
        for i, line_seg in enumerate(self.polylines()):
            pt_proj, t = line_seg.projection(point)
            d = np.sqrt(np.sum((pt_proj - point) ** 2))
            if d < dist:
                min_proj = pt_proj
                min_seg = i
                min_t = t
                dist = d
        return min_t, min_proj, min_seg

    def write_json(self):
        """Create a dictionary and return it"""
        my_dict = {"Name": "ControlHull", "cntrl_hull_pts": super().write_json()}

        return my_dict

    @staticmethod
    def read_json(json_dict, control_hull_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param control_hull_instance - an existing of points list to put the data in"""
        if json_dict["Name"] != "ControlHull":
            raise ValueError(f"This is not a control hull dictionary {json_dict}")

        if not control_hull_instance:
            control_hull_instance = ControlHull(initial_points=json_dict["cntrl_hull_pts"]["pts"])
        else:
            control_hull_instance.set_points(json_dict["cntrl_hull_pts"]["pts"])
        # Check
        control_hull_instance.internal_check()
        return control_hull_instance

    def internal_check(self):
        """ Check that all the data lines up"""

        assert self._points_as_ndarray.shape[0] == len(self._polylines) + 1
        for i in range(0, self.n_points() - 2):
            for d in range(0, self.dim()):
                assert np.isclose(self._polylines[i + 1].p1[d], self._polylines[i].p2[d])

        for i in range(0, self.n_points() - 1):
            for d in range(0, self.dim()):
                assert np.isclose(self._polylines[i].p1[d], self._points[i][d])
                assert np.isclose(self._polylines[i].p2[d], self._points[i+1][d])

        return super().internal_check()


if __name__ == "__main__":

    # check control hull
    control_hull = ControlHull(np.array([[0, 0], [1, 0], [1, 1], [1, 0]]))
    assert control_hull.n_points() == 4
    assert control_hull.dim() == 2
    assert control_hull.hull_length() == 3.0
    assert control_hull.internal_check()

    # Add point
    control_hull.add_point([0.5, 0.5])
    res1 = control_hull.project_on_hull(point=np.array([0.5, -0.25]))
    assert np.isclose(res1[0], 0.5)
    assert np.isclose(res1[1][0], 0.5)
    assert np.isclose(res1[1][1], 0.0)
    assert res1[2] == 0
    assert control_hull.internal_check()

    control_hull.add_point([0.5, 0.5])
    res2 = control_hull.project_on_hull(point=np.array([0.6, 0.6]))
    assert (np.isclose(res2[0], 1.0))
    assert (np.isclose(res2[1][0], 0.5))
    assert (np.isclose(res2[1][1], 0.5))
    assert (res2[2] == 3)
    assert control_hull.internal_check()

    control_hull.reverse_direction()
    assert np.all(np.isclose(control_hull.point(0), [0.5, 0.5]))
    assert np.all(np.isclose(control_hull.point(-1), [0.0, 0.0]))
    assert control_hull.internal_check()

    import json
    fname = "../Image_based/data/test_ctrl_hull.txt"
    with open(fname, "w") as f:
        json.dump(control_hull.write_json(), f, indent=2)

    with open(fname, 'r') as f:
        my_data = json.load(f)

        check_read = ControlHull.read_json(my_data)

        assert check_read.n_points() == control_hull.n_points()
        for ind in range(0, check_read.n_points()):
            assert np.all(np.isclose(check_read.point(ind), control_hull.point(ind)))
