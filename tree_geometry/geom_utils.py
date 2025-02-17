import numpy as np
from scipy.spatial import ConvexHull


class LineSeg2D:
    """Adapted from OSURobotics/treefitting"""

    def __init__(self, p1, p2):
        """Line segment with Ax + By + C form for closest point
        @param p1: Pt 1
        @param p2: Pt 2"""

        self.p1 = p1
        self.p2 = p2
        self.a, self.b, self.c = self.line(p1, p2)
        check1 = self.a * p1[0] + self.b * p1[1] + self.c
        check2 = self.a * p2[0] + self.b * p2[1] + self.c
        if not np.isclose(check1, 0.0) or not np.isclose(check2, 0.0):
            raise ValueError("LineSeg2D: Making line, pts not on line")

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

    def projection(self, pt):
        """Project the point onto the line and return the t value
        a ((1-t)p1x + t p2x) + b ((1-t)p1y + t p2y) + c = 0
        t (a(p2x-p1x) + b(p2y-p1y)) = -c - a (p1x + p2x) - b(p1y + p2y)
        @param pt - pt to project
        @return t of projection point"""

        # distance between p1 and p2, squared
        l2 = np.sum((self.p2 - self.p1) ** 2)
        if np.isclose(l2, 0.0):
            return self.p1, 0.5

        # The line extending the segment is parameterized as p1 + t (p2 - p1).
        # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

        t = max(min(np.dot((pt - self.p1), (self.p2 - self.p1)) / l2, 1), 0)

        pt_proj = self.p1 + t * (self.p2 - self.p1)
        check = self.a * pt_proj[0] + self.b * pt_proj[1] + self.c
        assert np.isclose(check, 0.0)  # check on line
        dotprod = np.dot(pt - pt_proj, self.p2 - self.p1)
        if not (np.isclose(t, 0.0) or np.isclose(t, 1.0)):
            assert np.isclose(dotprod, 0.0)  # check perpendicular

        return pt_proj, t

    # @staticmethod
    # def draw_line(im, p1, p2, color, thickness=1):
    #     """ Draw the line in the image using opencv
    #     @param im - the image
    #     @param p1 - first point
    #     @param p2 - second point
    #     @param color - rgb as an 0..255 tuple
    #     @param thickness - thickness of line
    #     """
    #     try:
    #         p1_int = [int(x) for x in p1]
    #         p2_int = [int(x) for x in p2]
    #         cv2.line(im, (p1_int[0], p1_int[1]), (p2_int[0], p2_int[1]), color, thickness)
    #     except TypeError:
    #         p1_int = [int(x) for x in np.transpose(p1)]
    #         p2_int = [int(x) for x in np.transpose(p2)]
    #         cv2.line(im, (p1_int[0], p1_int[1]), (p2_int[0], p2_int[1]), color, thickness)
    #         print(f"p1 {p1} p2 {p2}")
    #     """
    #     p0 = p1
    #     p1 = p2
    #     r0 = p0[0, 0]
    #     c0 = p0[0, 1]
    #     r1 = p1[0, 0]
    #     c1 = p1[0, 1]
    #     rr, cc = draw.line(int(r0), int(r1), int(c0), int(c1))
    #     rr = np.clip(rr, 0, im.shape[0]-1)
    #     cc = np.clip(cc, 0, im.shape[1]-1)
    #     im[rr, cc, 0:3] = (0.1, 0.9, 0.9)
    #     """

    # @staticmethod
    # def draw_cross(im, p, color, thickness=1, length=2):
    #     """ Draw the line in the image using opencv
    #     @param im - the image
    #     @param p - point
    #     @param color - rgb as an 0..255 tuple
    #     @param thickness - thickness of line
    #     @param length - how long to make the cross lines
    #     """
    #     LineSeg2D.draw_line(im, p - np.array([0, length]), p + np.array([0, length]), color=color, thickness=thickness)
    #     LineSeg2D.draw_line(im, p - np.array([length, 0]), p + np.array([length, 0]), color=color, thickness=thickness)

    # @staticmethod
    # def draw_box(im, p, color, width=6):
    #     """ Draw the line in the image using opencv
    #     @param im - the image
    #     @param p - point
    #     @param color - rgb as an 0..255 tuple
    #     @param width - size of box
    #     """
    #     for r in range(-width, width):
    #         LineSeg2D.draw_line(im, p - np.array([-r, width]), p + np.array([r, width]), color=color, thickness=1)

    # @staticmethod
    # def draw_rect(im, bds, color, width=6):
    #     """ Draw the line in the image using opencv
    #     @param im - the image
    #     @param bds - point
    #     @param color - rgb as an 0..255 tuple
    #     @param thickness - thickness of line
    #     """
    #     LineSeg2D.draw_line(im, [bds[0][0], bds[1][0]], [bds[0][0], bds[1][1]], color=color, thickness=1)
    #     LineSeg2D.draw_line(im, [bds[0][1], bds[1][0]], [bds[0][1], bds[1][1]], color=color, thickness=1)
    #     LineSeg2D.draw_line(im, [bds[0][0], bds[1][0]], [bds[0][1], bds[1][0]], color=color, thickness=1)
    #     LineSeg2D.draw_line(im, [bds[0][0], bds[1][1]], [bds[0][1], bds[1][1]], color=color, thickness=1)


class ControlHull:
    def __init__(self, points):
        self.dim = len(points[0])
        self.points = points
        self.polylines = [(i, i + 1) for i in range(len(points) - 1)]

    def parameteric_project(self, point):
        dist = np.Inf
        min_t = 0
        min_seg = None
        min_proj = None
        for pairs in self.polylines:
            p1 = self.points[pairs[0]]
            p2 = self.points[pairs[1]]
            ls = LineSeg2D(p1, p2)
            pt_proj, t = ls.projection(point)
            d = np.sqrt(np.sum((pt_proj - point) ** 2))
            if d < dist:
                min_proj = pt_proj
                min_seg = pairs
                min_t = t
                dist = d
        return (min_t, min_proj, min_seg)


class ConvexHullGeom(ConvexHull):
    """Construct and calculate spatial transforms on a convex hull. Relies on scipy.spatial for construction"""

    def __init__(self, points: np.ndarray) -> None:
        super().__init__(points)
        self.dim = len(points[0])

    def parameteric_project(self, point):
        # pt_idx_for_visibility = self.points.shape[0]
        # qhull_option = "QG" + str(pt_idx_for_visibility)
        # generators = np.vstack((self.points, point))
        # # only get the segments visible from the convex hull
        # nhull = ConvexHull(points=generators, qhull_options=qhull_option)
        dist = np.Inf
        min_t = 0
        min_seg = None
        min_proj = None
        for idx in self.simplices:
            p1 = self.points[idx[0]]
            p2 = self.points[idx[1]]
            ls = LineSeg2D(p1, p2)
            pt_proj, t = ls.projection(point)
            d = np.sum((pt_proj - point) ** 2)
            print(f"{point} checking {p1}, {p2}\ngets {pt_proj} d {d}")
            # print(d)
            if d < dist:
                min_proj = pt_proj
                min_seg = (idx[0], idx[1])
                min_t = t
                dist = d
        return (min_t, min_proj, min_seg)


if __name__ == "__main__":
    line = LineSeg2D(np.array([0, 0]), np.array([1, 0]))
    pt, t = line.projection(np.array([0.5, 0]))
    assert np.isclose(t, 0.5)
    assert np.isclose(pt[0], 0.5)
    assert np.isclose(pt[1], 0.0)

    pt, t = line.projection(np.array([0.5, 1.0]))
    assert np.isclose(t, 0.5)
    assert np.isclose(pt[0], 0.5)
    assert np.isclose(pt[1], 0.0)
