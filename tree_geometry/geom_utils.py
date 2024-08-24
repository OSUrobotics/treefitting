import numpy as np
from copy import deepcopy


class LineSeg:
    """Line segment virtual class"""

    def __init__(self, p1, p2):
        """Line segment with Ax + By + C form for closest point
        @param p1: Pt 1, as list or numpy array
        @param p2: Pt 2, as list or numpy array"""

        self._p1 = np.array(p1)
        self._p2 = np.array(p2)

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, p):
        self._p1 = np.array(p)

    @property
    def p2(self):
        return self._p2

    @p2.setter
    def p2(self, p):
        self._p2 = np.array(p)

    def line_length(self):
        """Length of curve segment
        @return length"""
        return np.linalg.norm(self._p1 - self._p2)

    def eval(self, t):
        """Evaluate the line segment at t
        @param t: t between 0 (pt1) and 1 (pt2)
        @return nparray point"""
        return (1 - t) * self._p1 + t * self._p2

    def projection(self, pt):
        """Project the point onto the line and return the t value
        a ((1-t)p1x + t p2x) + b ((1-t)p1y + t p2y) + c = 0
        t (a(p2x-p1x) + b(p2y-p1y)) = -c - a (p1x + p2x) - b(p1y + p2y)
        @param pt - pt to project
        @return t of projection point"""

        # distance between p1 and p2, squared
        l2 = np.sum((self._p2 - self._p1) ** 2)
        if np.isclose(l2, 0.0):
            return self._p1, 0.5

        # The line extending the segment is parameterized as p1 + t (p2 - p1).
        # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

        t = max(min(np.dot((pt - self._p1), (self._p2 - self._p1)) / l2, 1), 0)

        pt_proj = self._p1 + t * (self._p2 - self._p1)
        dotprod = np.dot(pt - pt_proj, self._p2 - self._p1)
        if not (np.isclose(t, 0.0) or np.isclose(t, 1.0)):
            assert np.isclose(dotprod, 0.0)  # check perpendicular

        return pt_proj, t


class LineSeg1D(LineSeg):
    """Very boring one dimensional line segment"""

    def __init__(self, p1, p2):
        """Line segment with Ax + By + C form for closest point
        @param p1: Pt 1, as list or numpy array
        @param p2: Pt 2, as list or numpy array"""

        super().__init__(p1, p2)


class LineSeg2D(LineSeg):
    """Adding ax + by + c to line seg"""

    def __init__(self, p1, p2):
        """Line segment with Ax + By + C form for closest point
        @param p1: Pt 1, as list or numpy array
        @param p2: Pt 2, as list or numpy array"""

        super().__init__(p1, p2)
        self._a, self._b, self._c = self.line(self._p1, self._p2)
        check1 = self._a * p1[0] + self._b * p1[1] + self._c
        check2 = self._a * p2[0] + self._b * p2[1] + self._c
        if not np.isclose(check1, 0.0) or not np.isclose(check2, 0.0):
            raise ValueError("LineSeg2D: Making line, pts not on line")

    @LineSeg.p1.setter
    def p1(self, p):
        super(LineSeg, self).p1 = p
        self._a, self._b, self._c = self.line(self._p1, self._p2)

    @LineSeg.p2.setter
    def p2(self, p):
        super(LineSeg, self).p2 = p

        self._a, self._b, self._c = self.line(self._p1, self._p2)

    @staticmethod
    def line(p1, p2):
        """a line in implicit coordinates
        @param p1 end point one
        @param p2 end point two
        @return a x + b y + c"""
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        return a, b, c

    @staticmethod
    def intersection(l1, l2):
        """Line-line intersection
        @param l1 - line one in implicit coords
        @param l2 - line two in implicit coords
        @return x, y if intersection point, None otherwise"""
        d = l1.a * l2.b - l1.b * l2.a
        dx = l1.c * l2.b - l1.b * l2.c
        dy = l1.a * l2.c - l1.c * l2.a
        if abs(d) > 1e-10:
            x = -dx / d
            y = -dy / d
            check1 = l1.a * x + l1.b * y + l1.c
            check2 = l2.a * x + l2.b * y + l2.c
            if not np.isclose(check1, 0.0) or not np.isclose(check2, 0.0):
                raise ValueError("LineSeg2D: Making line, pts not on line")
            return x, y
        else:
            return None


class LineSeg3D(LineSeg):
    """3D seg; at some point need to add back in line line intersection"""

    def __init__(self, p1, p2):
        """Line segment with Ax + By + C form for closest point
        @param p1: Pt 1, as list or numpy array
        @param p2: Pt 2, as list or numpy array"""

        super().__init__(p1, p2)


class ControlHull:
    def __init__(self, initial_points):
        """ A control hull has to have at least two points
        @param initial_points: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""

        # Keep these three forms of the points
        # - a list of numpy arrays of dimension dim
        # - the points as an n points x dim numpy array
        # - LineSeg2D for each edge

        self._points = None
        self._points_as_ndarray = None
        self._polylines = None

        self.set_points(initial_points)

    def dim(self):
        return len(self._points[0])

    def n_points(self):
        return len(self._points)

    def points(self):
        """ Points is a list of numpy arrays of dimension dim"""
        return self._points

    def polylines(self):
        """ If there are n points, then there are n-1 LineSeg2D's """
        return self._polylines

    def points_as_ndarray(self):
        """ Numerically the same as points, but stored as a numpy array of
          shape n_points() X dim() """
        return self._points_as_ndarray

    def set_points(self, new_pts):
        """ Set the control null from a list of numpy arrays (or list of lists)
        @param points: list of points, each point is a list of 2,3, etc dims
           OR if point is an nxdim array, make a list of n points"""

        self._points = []
        try:
            self._points_as_ndarray = np.zeros((len(new_pts), len(new_pts[0])))
            for i, p in enumerate(new_pts):
                self._points.append(np.array(p))
                self._points_as_ndarray[i, :] = self._points[-1]
        except ValueError:
            for p in new_pts.shape[0]:
                self._points.append(p)
            self._points_as_ndarray = deepcopy(new_pts)

        np.array(self._points)
        if self.n_points() < 2:
            raise ValueError("ControlHull: Need at least two points, got {self.dim()}")

        self._polylines = [LineSeg(self._points[i], self._points[i + 1]) for i in range(len(self._points) - 1)]

    def add_point(self, pt):
        # Use __ to set the actual variable without calling setter
        self._points.append(pt)
        self._polylines.append(LineSeg(self._points[-2], self._points[-1]))
        self._points_as_ndarray = np.array(self._points)

    def hull_length(self):
        """ Length of hull
        @return sum of lengths of line segments"""
        seg_lengths = [l.line_length() for l in self._polylines]
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


if __name__ == "__main__":
    # Check line seg
    line = LineSeg2D(np.array([0, 0]), np.array([1, 0]))
    pt_check, t_check = line.projection(np.array([0.5, 0]))
    assert np.isclose(t_check, 0.5)
    assert np.isclose(pt_check[0], 0.5)
    assert np.isclose(pt_check[1], 0.0)
    assert(line.line_length() == 1.0)

    pt_check, t_check = line.projection(np.array([0.5, 1.0]))
    assert np.isclose(t_check, 0.5)
    assert np.isclose(pt_check[0], 0.5)
    assert np.isclose(pt_check[1], 0.0)

    # check control hull
    control_hull = ControlHull(np.array([[0, 0], [1, 0], [1, 1], [1, 0]]))
    assert (control_hull.n_points() == 4)
    assert (control_hull.dim() == 2)
    assert (control_hull.hull_length() == 3.0)

    # Add point
    control_hull.add_point([0.5, 0.5])
    res1 = control_hull.project_on_hull(point=np.array([0.5, -0.25]))
    assert (np.isclose(res1[0], 0.5))
    assert (np.isclose(res1[1][0], 0.5))
    assert (np.isclose(res1[1][1], 0.0))
    assert (res1[2] == 0)
    control_hull.add_point([0.5, 0.5])
    res2 = control_hull.project_on_hull(point=np.array([0.6, 0.6]))
    assert (np.isclose(res2[0], 1.0))
    assert (np.isclose(res2[1][0], 0.5))
    assert (np.isclose(res2[1][1], 0.5))
    assert (res2[2] == 3)
