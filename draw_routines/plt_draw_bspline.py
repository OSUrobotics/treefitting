#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from tree_geometry.b_spline_curve import BSplineCurve

def plot_basis(plt, crv: BSplineCurve):
    """Plots the basis function in [0, 1)]"""
    tr = np.linspace(0, 0.99999, 100)
    bases = crv.eval_basis(crv._basis_matrix, t=tr)
    for i in range(0, bases.shape[1]):
        plt.plot(tr, bases[:, i].transpose())
    plt.set_xlabel("t values")


def plot_control_hull(plt, crv):
    x = []
    y = []
    for p in crv.points():
        x.append(p[0])
        y.append(p[1])

    plt.plot(x, y, "-g", label="control hull")


def plot_crv(plt, crv: BSplineCurve):
    ts = np.linspace(0, crv.max_t(), 20 * crv.n_points())
    pts = crv.eval_crv(t=ts)
    x = pts[:, 0].transpose()
    y = pts[:, 1].transpose()
    plt.plot(x, y, "-b", label="curve")


def onclick(event):
    """manages matplotlib interactive plotting

    :param event: _description_
    """
    # print(type(event))


def plot_curve_debug(ax, crv: BSplineCurve):
    plot_crv(ax, crv)
    plot_control_hull(ax, crv)
    ctrl_array = crv.points_as_ndarray()
    ax.plot(
        ctrl_array[:, 0], ctrl_array[:, 1], "ro", label="control points"
    )
    ax.plot(
        [min(-2, min(ctrl_array[:, 0] - 2)), max(3, max(ctrl_array[:, 0] + 2))],
        [0, 0],
        "-k",
    )  # x axis
    ax.plot(
        [0, 0],
        [min(-3, min(ctrl_array[:, 1] - 2)), max(3, max(ctrl_array[:, 1] + 2))],
        "-k",
    )
    ax.axis("equal")
    ax.grid()
    ax.legend()
    plt.draw()

class InteractiveCurve():
    def __init__(self, crv: BSplineCurve, figax):
        self.crv = crv
        self.fig, self.ax = figax
        plot_curve_debug(self.ax, self.crv)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.suptitle("Left click to add control point, Right click to project on curve")
        plt.show()
    
    def onclick(self, event):
        if event.button == 3:  # projection on convex hull RIGHT
            ix, iy = event.xdata, event.ydata
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            print(f"projecting x {ix} y {iy}")
            # project_ctrl_hull((ix, iy))
            _, pt_proj, __ = self.crv.project_to_curve((ix, iy))
            self.ax.clear()
            plot_curve_debug(self.ax, self.crv)
            self.ax.plot([ix, pt_proj[0]], [iy, pt_proj[1]], "o-", label="projection")
        elif event.button == 1:  # add control point LEFT
            ix, iy = event.xdata, event.ydata
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            self.crv.add_point(np.array((ix, iy)))
            print(f"x {ix} y {iy} added")
            self.ax.clear()
            plot_curve_debug(self.ax, self.crv)
        plt.tight_layout()
        ax.legend()
        plt.draw()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    _, axes = plt.subplots(2, 3)
    for i, deg in enumerate(['linear', 'quadratic', 'cubic']):
        axs = axes[0, i]
        crv = BSplineCurve([[0, 0], [1, 1], [2, 1], [3, 0]], degree=deg)
        plot_basis(axs, crv)
        axs.set_title(deg)
        plot_control_hull(axes[1, i], crv)
        plot_crv(axes[1, i], crv)

    plt.show()

    _, axes = plt.subplots(2, 3)
    for i, deg in enumerate(['linear', 'quadratic', 'cubic']):
        axs = axes[0, i]
        crv = BSplineCurve([[0, 0], [1, 1], [2, 1], [3, 0]], degree=deg)
        plot_basis(axs, crv)
        axs.set_title(deg)
        plot_control_hull(axes[1, i], crv)
        plot_crv(axes[1, i], crv)

    plt.show()

    fig, ax = plt.subplots()
    InteractiveCurve(BSplineCurve([[0, 0], [1, 1], [2, 1], [3, 0]], degree='cubic'), (fig, ax))
    print("done")
