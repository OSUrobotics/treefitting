#!/usr/bin/env python3

import math as math


# Keep all of the sketching data until it is time to actually make a spline curve
#  There's the curve backbone (2 o4 more points)
#  Plus 1 (or more) perpendicular cross bars to indicate width
#
# Regular clicking makes more backbone points (needs to bee in order)
# Shift click does a left-right pair across the backbone
# Cntrl clicking over a point erases it
#  Keep all points in width/height - convert at the end

class SketchesForCurves:
    def __init__(self):
        """ Nothing in it to start"""
        self.backbone_pts = []
        self.cross_bars = []

    def add_backbone_point(self, x, y):
        """ Add the x,y point to the backbone
        @param x
        @param y
        """
        self.backbone_pts.append([x, y])

    def add_crossbar_point(self, x, y):
        """ Add the x,y point to the crossbar
        @param x
        @param y
        """
        if self.cross_bars == []:
            self.cross_bars.append([[x, y]])
        elif len(self.cross_bars[-1]) == 1:
            self.cross_bars[-1].append([x, y])
        else:
            self.cross_bars.append([[x, y]])

    def remove_point(self, x, y, eps=5):
        """ Remove either a backbone point or a cross bar point
        @param x
        @param y
        """
        d_closest = 1e30
        i_closest = -1
        for i, pt in enumerate(self.backbone_pts):
            d_dist = math.fabs(pt[0] - x) + math.fabs(pt[1] - y)
            if d_dist < d_closest:
                d_closest = d_dist
                i_closest = i

        d_closest_cross = 1e30
        i_closest_cross = -1
        for i, pts in enumerate(self.cross_bars):
            for pt in pts:
                d_dist = math.fabs(pt[0] - x) + math.fabs(pt[1] - y)
                if d_dist < d_closest_cross:
                    d_closest_cross = d_dist
                    i_closest_cross = i

        if i_closest == -1 and i_closest_cross == -1:
            return

        if d_closest > eps and d_closest_cross > eps:
            return

        if d_closest < d_closest_cross:
            self.backbone_pts.pop(i_closest)
        else:
            self.cross_bars.pop(i_closest_cross)


if __name__ == '__main__':
    sk = SketchesForCurves()
    sk.add_backbone_point(10, 10)
    sk.add_backbone_point(100, 20)
    sk.add_backbone_point(200, 30)
    sk.add_crossbar_point(-5, 10)
    sk.add_crossbar_point(25, 10)
    sk.add_crossbar_point(85, 30)
    sk.add_crossbar_point(120, 10)
    sk.remove_point(10, 10, 2)
    sk.remove_point(120, 10, 2)

    print(sk.backbone_pts)
    print(sk.cross_bars)


