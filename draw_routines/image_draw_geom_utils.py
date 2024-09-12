#!/usr/bin/env python3

# Read in masked images and estimate points where a side branch joins a leader (trunk)

from point_lists import LineSeg2D, ControlHull
import numpy as np
import cv2


def draw_line(im, p1, p2, color, thickness=1):
    """ Draw the line in the image using opencv
    @param im - the image
    @param p1 - first point
    @param p2 - second point
    @param color - rgb as an 0..255 tuple
    @param thickness - thickness of line
    """
    try:
        p1_int = [int(x) for x in p1]
        p2_int = [int(x) for x in p2]
        cv2.line(im, (p1_int[0], p1_int[1]), (p2_int[0], p2_int[1]), color, thickness)
    except TypeError:
        p1_int = [int(x) for x in np.transpose(p1)]
        p2_int = [int(x) for x in np.transpose(p2)]
        cv2.line(im, (p1_int[0], p1_int[1]), (p2_int[0], p2_int[1]), color, thickness)
        print(f"p1 {p1} p2 {p2}")
    """
    p0 = p1
    p1 = p2
    r0 = p0[0, 0]
    c0 = p0[0, 1]
    r1 = p1[0, 0]
    c1 = p1[0, 1]
    rr, cc = draw.line(int(r0), int(r1), int(c0), int(c1))
    rr = np.clip(rr, 0, im.shape[0]-1)
    cc = np.clip(cc, 0, im.shape[1]-1)
    im[rr, cc, 0:3] = (0.1, 0.9, 0.9)
    """


def draw_line_seg(im, l:LineSeg2D, color, thickness=1):
    """ Draw the line in the image using opencv
    @param im - the image
    @param l - line
    @param color - rgb as an 0..255 tuple
    @param thickness - thickness of line
    """
    p1 = l.p1
    p2 = l.p2
    draw_line(im, p1, p2, color, thickness)


def draw_cross(im, p, color, thickness=1, length=2):
    """ Draw the line in the image using opencv
    @param im - the image
    @param p - point
    @param color - rgb as an 0..255 tuple
    @param thickness - thickness of line
    @param length - how long to make the cross lines
    """
    draw_line(im, p - np.array([0, length]), p + np.array([0, length]), color=color, thickness=thickness)
    draw_line(im, p - np.array([length, 0]), p + np.array([length, 0]), color=color, thickness=thickness)

def draw_box(im, p, color, width=6):
    """ Draw the line in the image using opencv
    @param im - the image
    @param p - point
    @param color - rgb as an 0..255 tuple
    @param width - size of box
    """
    for r in range(-width, width):
        draw_line(im, p - np.array([-r, width]), p + np.array([r, width]), color=color, thickness=1)

def draw_rect(im, bds, color, width=6):
    """ Draw the line in the image using opencv
    @param im - the image
    @param bds - point
    @param color - rgb as an 0..255 tuple
    @param thickness - thickness of line
    """
    draw_line(im, [bds[0][0], bds[1][0]], [bds[0][0], bds[1][1]], color=color, thickness=1)
    draw_line(im, [bds[0][1], bds[1][0]], [bds[0][1], bds[1][1]], color=color, thickness=1)
    draw_line(im, [bds[0][0], bds[1][0]], [bds[0][1], bds[1][0]], color=color, thickness=1)
    draw_line(im, [bds[0][0], bds[1][1]], [bds[0][1], bds[1][1]], color=color, thickness=1)

def draw_hull(im, hull:ControlHull, color, thickness=1):
    """
    Draw the convex hull in the image
    @param im - the image
    @param hull - Convex hull
    @param color - rgb as an 0..255 tuple
    @param thickness - thickness of line
    """
    for l in hull.polylines:
        draw_line_seg(im, l, color=color, thickness=thickness)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    im = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_line(im, np.array([10, 20]), np.array([410, 610]), color=[250, 0, 0], thickness=4)
    draw_cross(im, np.array([130, 140]), color=[250, 250, 0], thickness=4)
    draw_box(im, np.array([200, 10]), color=[250, 250, 250], width=6)
    line_draw = LineSeg2D([30, 200], [200, 20])
    draw_line_seg(im, line_draw, color=(255, 0, 0), thickness=1)

    control_hull = ControlHull(np.array([[30, 300], [340, 300], [400, 390], [30, 370]]))
    draw_hull(im, control_hull, color=(0, 255, 0), thickness=3)
    pt_close = np.array([100, 320])
    draw_cross(im, pt_close, color=(0, 255, 0), thickness=3)
    ret = control_hull.parameteric_project(pt_close)
    draw_cross(im, ret[1], color=(0, 255, 0), thickness=3)
    pt_check = control_hull.polylines[ret[2]].eval(ret[0])
    draw_box(im, np.array(pt_check), color=(0, 0, 255), width=2)
    draw_line_seg(im, control_hull.polylines[ret[2]], color=(0, 255, 0), thickness=1)

    plt.subplot()
    plt.imshow(im)
    plt.show()
    print("done")
