#!/usr/bin/env python3

import math as math


# Keep all of the sketching data until it is time to actually make a spline curve
#  There's the curve backbone (2 or more points)
#  Plus 1 (or more) perpendicular cross bars to indicate width
#
# Regular clicking makes more backbone points (needs to bee in order)
# Shift click does a left-right pair across the backbone
# Cntrl clicking over a point erases it
#  Keep all points in width/height - convert at the end

class SketchedCurve:
    def __init__(self):
        """ Nothing in it to start"""
        self.backbone_pts = []
        self.cross_bars = []

    def n_points(self):
        return len(self.backbone_pts)

    def radii(self):
        """ Convert the crossbars to radii and return a list
        @return a list"""
        radii_vs = []
        radii_ctrs = []
        for pts in self.cross_bars:
            if len(pts) == 2:
                radius = math.sqrt(math.pow(pts[1][0] - pts[0][0], 2) + math.pow(pts[1][1] - pts[0][1], 2)) / 2.0
                radii_ctrs.append([0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1])])
                radii_vs.append(radius)

        if len(radii_vs) < 2:
            # 3 and 2 pixels
            radii_vs.append(3)
            radii_vs.append(2)
            if len(self.backbone_pts) > 1:
                # Start and end of curve
                radii_ctrs.append(self.backbone_pts[0])
                radii_ctrs.append(self.backbone_pts[-1])
            else:
                radii_ctrs.append([0, 0])
                radii_ctrs.append([1, 1])
        return radii_ctrs, radii_vs

    @staticmethod
    def convert_pt(pt, lower_left, upper_right, width, height):
        """ Convert the drawing window point to the image point
        @param pt x,y point as a list
        @param lower_left - lower left point of the image in the window
        @param upper_right - upper right point of the image in the window
        @param width - width of image
        @param height - height of image
        @returns pt in image coordinates"""
        x = int(width * (pt[0] - lower_left[0]) / (upper_right[0] - lower_left[0]))
        if x < 0: 
            x = 0
        if x >= width:
            x = width - 1
        y = int(height * (pt[1] - lower_left[1]) / (upper_right[1] - lower_left[1]))
        if y < 0:
            y = 0
        if y >= height:
            y = height - 1
        return x, y

    def convert_image(self, lower_left, upper_right, width, height):
        """ Make a copy of self with points in image coordinates, not screen
        @param lower_left - lower left point of the image in the window
        @param upper_right - upper right point of the image in the window
        @param width - width of image
        @param height - height of image
        @returns SketchedCurve in coordinate system
        """
        ret_sketch = SketchedCurve()
        for pt in self.backbone_pts:
            x, y = SketchedCurve.convert_pt(pt, lower_left, upper_right, width, height)
            ret_sketch.add_backbone_point(x, y)

        for pts in self.cross_bars:
            # Make sure pts has two elements
            if len(pts) == 2:
                for pt in pts:
                    x, y = SketchedCurve.convert_pt(pt, lower_left, upper_right, width, height)
                    ret_sketch.add_crossbar_point(x, y)
        return ret_sketch

    def clear(self):
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

    def copy(self):
        """ make a copy and return it"""
        copy_curve = SketchedCurve()
        copy_curve.backbone_pts = [p.copy() for p in self.backbone_pts]
        copy_curve.cross_bars = []
        for cb in self.cross_bars:
            copy_curve.cross_bars.append([p.copy() for p in cb])
        return copy_curve

    def write_json(self):
        """Convert to array and return dictionary
        @param fname file name to write to"""
        my_dict = {"Name": "SketchedCurve", "backbone_pts": self.backbone_pts, "cross_bars": self.cross_bars}
        return my_dict

    @staticmethod
    def read_json(my_dict, sketches_crv :object=None):
        """ Read back in from json file
        @param my_dict - json dictionary
        @param sketches_crv - an existing bezier SketchedCurve to put the data in"""
        assert my_dict["Name"] == "SketchedCurve"

        if not sketches_crv:
            sketches_crv = SketchedCurve()
        sketches_crv.backbone_pts = my_dict["backbone_pts"]
        sketches_crv.cross_bars = my_dict["cross_bars"]
        return sketches_crv


if __name__ == '__main__':
    sk = SketchedCurve()
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

    import json
    from os.path import exists

    fname = "test_sketches_for_crvs.json"
    with open(fname, "w") as f:
        json.dump(sk.write_json(), f, indent=2)

    with open(fname, 'r') as f:
        my_data = json.load(f)
        sk_check = SketchedCurve.read_json(my_data)
        SketchedCurve.read_json(my_data, sk_check)

    fname_save_crv = dirname + "save_crv.json"
    if exists(fname_save_crv):
        with open(fname_save_crv, 'r') as f:
            my_data = json.load(f)
            sk_read = SketchedCurve.read_json(my_data)
    else:
        sk_read = sk_check

    width_rgb_image = 640
    height_rgb_image = 480

    width_window = 1200
    height_window = 894

    # The rectangle of the image in window coordinates
    lower_left = [154, 116]
    upper_right = [154, 754]

    if height_window >  width_window:
        # Rectangle stretches across the image from left to right, height clipped to maintain aspect ratio
        pixs_missing = height_window - width_window
        lower_left[1] = int(pixs_missing * 0.5)
        upper_right[1] = height_window - int(pixs_missing * 0.5) - 1
    else:
        # Rectangle stretches across the image from top to bottom, width clipped to maintain aspect ratio
        pixs_missing_width = width_window - height_window
        pixs_missing_height = 0.5 * height_window * (height_rgb_image / width_rgb_image)
        lower_left[0] = int(pixs_missing_width * 0.5)
        lower_left[1] = int(pixs_missing_height * 0.5)
        upper_right[0] = width_window - int(pixs_missing_width * 0.5) - 1
        upper_right[1] = height_window - int(pixs_missing_height * 0.5) - 1

    # Actually convert the curve
    crv_in_image_coords = sk_read.convert_image(lower_left=lower_left, upper_right=upper_right, width=width_rgb_image, height=height_rgb_image)
    fname_convert_crv = dirname + "save_crv_in_image.json"
    with open(fname_convert_crv, "w") as f:
        json.dump(crv_in_image_coords.write_json(), f, indent=2)



