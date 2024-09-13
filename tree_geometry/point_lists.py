#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from tree_geometry.line_segs import LineSeg
from typing import Union

class PointList:
    def __init__(self, initial_points: Union[np.ndarray, list[np.ndarray], list[list]]):
        """ A list of ordered points, initialized with either a list of points or an ndarray of points
        Keeps both around (list and nd array)
        @param initial_points - list of points of dimension 1, 2, or 3"""

        self._points = None
        self._points_as_ndarray = None

        self.set_points(initial_points)

    def __deepcopy__(self, memo):
        """Deep copy constructor for ControlHull """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Will end up calling set_points, which does a deep copy
        return PointList(self._points)

    def dim(self):
        """ Dimension of the points"""
        return self._points_as_ndarray.shape[1]

    def n_points(self):
        """ Number of points"""
        return self._points_as_ndarray.shape[0]

    def points(self):
        """ Points is a list of numpy arrays of dimension dim"""
        return self._points

    def points_as_ndarray(self):
        """ Numerically the same as points, but stored as a numpy array of
          shape n_points() X dim() """
        return self._points_as_ndarray

    def set_points(self, new_pts: Union[np.ndarray, list[np.ndarray], list[list]]):
        """ Set the points from an nd array, a list of numpy arrays (or list of lists)
        Note: This does a deep copy, not a shallow one
        @param new_pts: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""

        self._points = []
        if isinstance(new_pts, list):
            if isinstance(new_pts[0], list):
                dim = len(new_pts[0])
            else:
                dim = new_pts[0].size
            self._points_as_ndarray = np.zeros((len(new_pts), len(new_pts[0])))
            for i, p in enumerate(new_pts):
                pt_as_array = np.zeros(dim)
                for d in range(0, dim):
                    pt_as_array[d] = p[d]
                self._points.append(pt_as_array)
                self._points_as_ndarray[i, :] = pt_as_array
        else:
            self._points_as_ndarray = deepcopy(new_pts)
            for i in range(0, self._points_as_ndarray.shape[0]):
                self._points.append(self._points_as_ndarray[i, :])

    def add_point(self, pt: Union[np.ndarray, list]):
        """ Point can be a 1xdim numpy array or a list with dim elements
        @param pt: the point to add"""
        pt_as_array = np.zeros(self.dim())
        for i in range(0, self.dim()):
            pt_as_array[i] = pt[i]

        self._points.append(pt_as_array)
        # append is not working here for whatever reason
        self._points_as_ndarray = np.append(self._points_as_ndarray, np.zeros((1, self.dim())), axis=0)
        self._points_as_ndarray[-1, :] = pt_as_array

    def internal_check(self):
        """ Just check that all the data is ok"""
        assert self._points_as_ndarray.shape[0] == len(self._points)
        for i in range(0, self.n_points()):
            assert self._points[i].size == self._points_as_ndarray.shape[1]
            for i_d in range(0, self.dim()):
                assert np.isclose(self._points[i][i_d], self._points_as_ndarray[i, i_d])
        return True


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


class PointListWithTs(PointList):
    def __init__(self, pts: Union[np.ndarray, list[np.ndarray], list[list]], ts: Union[np.ndarray, list] = None):
        """ List of points with t values for each
        @param pts: list of points, each point is a list of 2,3, etc dims
        @param ts: t values for each point. If None, will set to be between 1 and 1 by chord length"""

        self._ts = np.zeros(1)

        # this will call set_pts, which will initialize ts to chord length
        super().__init__(pts)

        if ts is not None:
            for i, t in enumerate(ts):
                self._ts[i] = t

    def __deepcopy__(self, memo):
        """Deep copy constructor for ControlHull """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Will end up calling set_points, which does a deep copy
        return PointListWithTs(self._points, self._ts)
    
    def __repr__(self) -> str:
        return f"pts: {self._points_as_ndarray}\nts: {self._ts}"

    @property
    def ts(self):
        """ Should be one for each point """
        return self._ts

    @ts.setter
    def ts(self, ts: Union[np.ndarray, list]):
        """Set to a new array of t values; checks for size and monotonically increasing
        @param ts list of t values of n_points size"""
        for i in range(0, self.n_points()):
            self._ts[i] = ts[i]

        for i in range(0, self.n_points() - 1):
            assert ts[i+1] >= ts[i]

    def set_points(self, new_pts: Union[np.ndarray, list[np.ndarray], list[list]]):
        """ Set the control null from a list of numpy arrays (or list of lists)
        Note: This does a deep copy, not a shallow one
        @param new_pts: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""
        super().set_points(new_pts)

        if self.ts.size != self.n_points():
            self.set_ts_chord_length()

    def add_point(self, pt: Union[np.ndarray, list]):
        """ add the point to the list, along with a new polyline
        @param pt: the point to add"""
        super().add_point(pt)
        self._ts = np.append(self._ts, self._ts[-1] + 1.0)

    def add_point_t(self, pt: Union[np.ndarray, list], t: float):
        """ add the point to the list, along with a new polyline
        @param pt: the point to add
        @param t: the t to add"""
        super().add_point(pt)
        self._ts = np.append(self._ts, t)

    def set_ts_chord_length(self) -> np.ndarray:
        """Get chord length parameterization of euclidean points
        :return: t values for each input point, starting at 0 and going to 1
        """
        distances = [np.linalg.norm(self.points()[i] - self.points()[i - 1]) for i in range(1, self.n_points())]
        # Make these a cummulative sum that starts at 0
        distances.insert(0, 0.0)
        ts_as_distances = np.cumsum(distances)
        self._ts = ts_as_distances / ts_as_distances[-1]

        return self._ts

    def normalize_ts(self, start_t: float = 0.0, end_t: float = 1.0) -> np.ndarray:
        """ make the ts go from start_t to end_t
        :param start_t: t value to start, defaults to 0
        :param end_t: t value to end, defaults to 1.
        :return: normalized t values
        """
        self._ts = start_t + (end_t - start_t) / (self._ts[-1] - self._ts[0]) * self._ts
        return self._ts

    def internal_check(self):
        """ Just check that the ts are monotonically increasing"""

        assert self._points_as_ndarray.shape[0] == self._ts.size
        for i in range(0, self.n_points() - 1):
            assert self._ts[i+1] >= self._ts[i]

        return super().internal_check()


if __name__ == "__main__":
    # Check points constructors
    pt_v1 = PointList([[2, 3], [4, 6], [7.2, 3]])
    assert pt_v1.dim() == 2
    assert pt_v1.n_points() == 3
    assert pt_v1.internal_check()

    pt_v1.add_point([3, 4])
    assert pt_v1.n_points() == 4
    assert pt_v1.internal_check()

    pt_new_v1 = np.zeros(2) + 3
    pt_v1.add_point(pt_new_v1)
    assert pt_v1.n_points() == 5
    assert pt_v1.internal_check()

    pt_v2 = PointList([pt_v1.points()[0], pt_v1.points()[1]])
    assert pt_v2.dim() == 2
    assert pt_v2.n_points() == 2
    assert pt_v2.internal_check()

    pt_v3 = PointList(pt_v1.points_as_ndarray())
    assert pt_v3.dim() == 2
    assert pt_v3.n_points() == pt_v1.n_points()
    assert pt_v3.internal_check()

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

    pts_with_ts = PointListWithTs(np.array([[0, 0], [1, 0], [1, 1], [1, 0]]))
    assert np.isclose(pts_with_ts.ts[0], 0.0)
    assert np.isclose(pts_with_ts.ts[-1], 1.0)
    assert pts_with_ts.internal_check()

    pts_with_ts.add_point_t(pt=[3, 4], t=1.5)
    assert np.isclose(pts_with_ts.ts[0], 0.0)
    assert np.isclose(pts_with_ts.ts[-1], 1.5)
    assert pts_with_ts.n_points() == 5

    pts_with_ts.normalize_ts(-1.0, 1.0)
    assert np.isclose(pts_with_ts.ts[0], -1.0)
    assert np.isclose(pts_with_ts.ts[-1], 1.0)
    assert pts_with_ts.internal_check()

    pts_with_ts_2 = PointListWithTs(np.array([[0, 0], [1, 0], [1, 1], [1, 0]]), [-2.0, 0.25, 0.5, 0.75])
    assert np.isclose(pts_with_ts_2.ts[0], -2.0)
    assert np.isclose(pts_with_ts_2.ts[-1], 0.75)
