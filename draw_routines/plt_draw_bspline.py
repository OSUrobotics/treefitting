#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from b_spline_curve import BSplineCurve


def plot_basis(plt, crv):
    """Plots the basis function in [0, 1)]"""
    tr = np.linspace(0, 0.99999, 100)
    bases = crv._eval_basis(crv._basis_matrix, t=tr)
    for i in range(0, bases.shape[0]):
        plt.plot(tr, bases[i, :].transpose())
    plt.set_xlabel("t values")


def plot_control_hull(plt, crv):
    x = []
    y = []
    for p in crv.points():
        x.append(p[0])
        y.append(p[1])

    plt.plot(x, y, "-g", label="control hull")


def plot_crv(plt, crv):
    ts = np.linspace(0, crv.max_t(), 20 * crv.n_points())
    pts = crv.eval_crv(t=ts)
    x = pts[:, 0].transpose()
    y = pts[:, 1].transpose()
    plt.plot(x, y, "-b", label="curve")


def plot_curve(self, fig=None, ax=None):
    """plot spline curve. do not pass fig or ax for using existing canvas

    :param fig: mpl figure to draw on, defaults to None
    :param ax: mpl axes to use, defaults to None
    :return: (ctrl_point_line, spline_line)
    """

    if fig == None and ax == None and self.fig == None and self.ax == None:
        print("Atleast pass figure and ax!")
    elif fig is not None and ax is not None:
        self.fig = fig
        self.ax = ax
    self.ax.clear()
    tr = np.linspace(0, len(self.ctrl_pts) - self.degree, 1000)
    tr = tr[:-1]  # [0, 1) range
    spline = self.eval_crv(tr)
    spline = np.array(spline)
    ctrl_array = np.reshape(self.ctrl_pts, (-1, self.dim))
    print(spline)
    # print(f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(ctrl_array)} points")
    (ln,) = self.ax.plot(
        ctrl_array[:, 0], ctrl_array[:, 1], "bo", label="control points"
    )
    (ln2,) = self.ax.plot(spline[:, 0], spline[:, 1], label="spline")
    self.ax.plot(
        [min(-2, min(ctrl_array[:, 0] - 5)), max(10, max(ctrl_array[:, 0] + 5))],
        [0, 0],
        "-k",
    )  # x axis
    self.ax.plot(
        [0, 0],
        [min(-10, min(ctrl_array[:, 1] - 5)), max(10, max(ctrl_array[:, 1] + 5))],
        "-k",
    )
    self.ax.axis("equal")
    self.ax.grid()
    plt.draw()
    return ln, ln2

def onclick(fig, crv, event):
    """manages matplotlib interactive plotting

    :param event: _description_
    """
    # print(type(event))
    if event.button == 1:  # projection on convex hull LEFT
        ix, iy = event.xdata, event.ydata
        if ix == None or iy == None:
            print("You didn't actually select a point!")
            return
        print(f"projecting x {ix} y {iy}")
        # self.project_ctrl_hull((ix, iy))
        pt_proj = crv.project_to_curve((ix, iy))
    elif event.button == 3:  # add control point RIGHT
        ix, iy = event.xdata, event.ydata
        if ix == None or iy == None:
            print("You didn't actually select a point!")
            return
        crv.add_ctrl_point(np.array((ix, iy)))
        print(f"x {ix} y {iy} added")
        plot_curve(fig, crv)

    def enable_onclick(self):
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

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
    print("done")
    # plot_control_hull(axes, BSplineCurve())
    # cid = fig.canvas.mpl_connect("button_press_event", onclick)
