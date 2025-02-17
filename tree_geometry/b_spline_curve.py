#!/usr/bin/env python3
import stat
from typing import Tuple, Union
from copy import deepcopy
import pprint

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import quad

from .geom_utils import ControlHull

np.set_printoptions(precision=3, suppress=True)
"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""


class BSplineCurve(object):
    degree_dict = dict(
        linear=1,
        quadratic=2,
        cubic=3,
    )
    # uniform knot vector only
    # General Matrix Representations for b-Splines, Kaihuai Qin
    # each matrix column represents a series of coeffs of increasing degree for a basis function
    basis_matrix_dict = {
    0: np.array([1]),
    1: np.array([[1, 0], [-1, 0]]),
    2: 1/2 * np.array([[1, 1, 0], 
                       [-2, 2, 0],
                       [1, -2, 1]]),
    3: 1/6 * np.array([[1, 4, 1, 0],
                       [-3, 0, 3, 0],
                       [3, -6, 3, 0], 
                       [-1, 3, -3, 1]]),
    }
    derivative_dict = {
    0: np.array([0]),
    1: np.array([[0, 0], [-1, 0]]),
    2: 1/2 * np.array([[0, 1, 0],
                       [-2, 2, 0],
                       [2, -4, 2]]),
    3: 1/6 * np.array([[1, 4, 1, 0],
                       [-3, 0, 3, 0],
                       [6, -12, 6, 0], 
                       [-3, 9, -9, 3]]),
    }


    # ROADMAP
    # curve tangent, normal binormal for frenet frames
    # apply a filter -> move to middle
    # integrate radius of curvature over the curve as metric for curviness

    def __init__(
        self,
        degree: str = "quadratic",
        dim: int = 2,
        ctrl_pts: list[np.ndarray] = [],
        figax=None,
    ) -> None:
        """BSpline initialization

        :param degree: degree of spline, defaults to "quadratic"
        :param dim: dimension of spline, defaults to 2
        :param ctrl_pts: control points, defaults to []
        :param figax: fig, ax tuple for interactive plotting, defaults to None
        """
        self.ctrl_pts: list[np.ndarray] = deepcopy(ctrl_pts)
        if ctrl_pts is not None and len(ctrl_pts) > 0:
            if len(ctrl_pts[0]) != dim:
                raise ValueError(
                    "Mismatch in control point dimension and initialized dim!"
                )
        self.dim = dim
        self.degree: int = self.degree_dict[degree]
        self.order = self.degree + 1
        self.basis_matrix: np.ndarray = BSplineCurve.basis_matrix_dict[self.degree]
        self.deriv_matrix: np.ndarray = BSplineCurve.derivative_dict[self.degree]

        # for interactive plotting
        if figax is not None:
            self.fig, self.ax = figax

    @staticmethod
    def from_existing(existing):
        """Create a new spline from an existing one

        :param existing: existing spline
        """
        return BSplineCurve(
            ctrl_pts = existing.ctrl_pts,  # Deep copy in init function
            dim = existing.dim,
            degree = existing.degree,
            figax = (existing.fig, existing.ax))

    @staticmethod
    def unflatten_dim(points, dim):
        """list of points converted to array of shape (points, dim)"""
        return np.reshape(np.array(points), (-1, dim))

    @property
    def max_t(self):
        return len(self.ctrl_pts) - self.degree

    @property
    def ctrl_hull_length(self):
        """Get control hull length"""
        distances = [
            np.linalg.norm(self.ctrl_pts[i] - self.ctrl_pts[i - 1])
            for i in range(1, len(self.ctrl_pts))
        ]
        distances.insert(0, 0.0)
        return float(np.cumsum(distances)[-1])

    @property
    def curve_length(self):
        """Get curve length using integration of the norm of derivative of curve
        we prefer to use fast approximate curve length provided by ctrl hull"""
        a, b = 0.0, self.max_t

        def f(x):
            return np.linalg.norm(self.derivative(x))

        length, _ = quad(f, a, b)
        return length
    
    @property
    def is_initialized(self):
        """Check if spline is initialized"""
        return len(self.ctrl_pts) > 0
    
    def pt_axis(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve at parameter t
        :param t: parameter
        :return: point on spline of dimension self.dim
        """
        return self.eval_crv(t)
    
    def tangent_axis(self, t: float) -> np.ndarray:
        """Get the tangent vector to the curve at parameter t
        :param t: parameter
        :return: tangent vector
        """
        return self.derivative(t)

    def add_ctrl_point(self, point: Union[Tuple[float], np.ndarray]) -> None:
        """Add a control point to the sequence"""
        if type(point) is not np.ndarray:
            point = np.array(point)
        if len(point) != self.dim:
            raise ValueError(
                f"Bad point dimension! Existing is {self.dim}, we got {len(point)}"
            )
        self.ctrl_pts.append(point)
        return

    def eval_basis(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """evaluate basis functions at t
        :param t: parameter
        :type t: float or np.ndarray
        :return: basis matrix at t, each column is a basis function
        :rtype: np.ndarray
        """
        return polyval(t, self.basis_matrix)

    def get_banded_matrix(
        self, t: np.ndarray, num_ctrl_pts: Union[None, int] = None
    ) -> np.ndarray:
        """get the banded matrix for the spline

        on multiplying this matrix with control points, we get the curve at t.
        it is zeroed out for control points that don't contribute to the curve at t
        :param t: t values
        :type t: np.ndarray
        :param num_ctrl_pts: optional number of control points, reuses saved ctrl_pts if None
        :type num_ctrl_pts: Union[None, int], optional
        :return: banded matrix
        :rtype: np.ndarray
        """
        if num_ctrl_pts is None:
            num_ctrl_pts = len(self.ctrl_pts)
        banded_basis_matrix = np.zeros((len(t), num_ctrl_pts), dtype=float)
        floor = np.floor(t)
        idxs = floor.astype(int)
        t_prime = t - floor
        evaluated_basis = self.eval_basis(t_prime).T

        # construct diagonal banded matrix
        for i in range(len(t)):
            if idxs[i] < 0:  # clip to left limit
                banded_basis_matrix[i, 0 : self.order] = self.eval_basis(0.0)
            elif idxs[i] >= (num_ctrl_pts - self.degree):  # clip to right limit
                banded_basis_matrix[i, -self.order :] = self.eval_basis(1.0)
            else:
                banded_basis_matrix[i, idxs[i] : (idxs[i] + self.order)] = (
                    evaluated_basis[i]
                )
        return banded_basis_matrix

    def plot_basis(self, plt):
        """Plots the basis function in [0, 1)]"""
        tr = np.linspace(0, 1.0, 100)
        tr = tr[0:-2]  # [0, 1) range
        basis = []
        for t in tr:
            basis.append(self.eval_basis(t=t))
        basis = np.array(basis)
        for i in range(0, self.order):
            plt.scatter(tr + (self.degree - i), basis[:, i])
        plt.xlabel("t values")
        plt.show()

    def eval_crv(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - parameter
        @return point on spline of dimension self.dim
        """
        if isinstance(t, float):
            t = [t]
        if self.ctrl_pts is None or len(self.ctrl_pts) <= self.degree:
            raise ValueError(
                f"Need atleast degree + 1: {self.order} control points for creating a bezier curve"
            )
        control_points = np.reshape(self.ctrl_pts, (-1, self.dim))
        try:
            res = np.matmul(self.get_banded_matrix(t), control_points)
            if res.shape[0] == 1:  # if we needed only point, return it as tuple
                return res[0]
            return res
        except ValueError as v:
            raise ValueError(
                f"Something went really wrong! {v} multiplying {self.get_banded_matrix(t).shape} with {control_points.shape}"
            )

    def _get_distance_from_curve(self, t: np.ndarray, pt: np.ndarray) -> np.ndarray:
        """Get distance from curve at param t for point, convenience function for using with scipy optimization

        :param t: t value
        :type t: np.ndarray
        :param pt: point to get distance from curve at
        :type pt: np.ndarray
        :return: distance
        :rtype: np.ndarray
        """
        res = self.eval_crv(t)
        return np.linalg.norm(res - pt)

    def project_ctrl_hull(self, pt) -> float:
        """Get t value for projecting point on hull

        :param pt: point to project
        :return: t value
        """
        self.hull = ControlHull(self.ctrl_pts)

        t, pt_proj, min_seg = self.hull.parameteric_project(pt)
        if min_seg is None or pt_proj is None:
            raise ValueError("Could not project")

        t_reindex = t + min_seg[0]

        # just drawing the hull
        x_ = []
        y_ = []
        for pairs in self.hull.polylines:
            x_seg = [self.ctrl_pts[pairs[0]][0], self.ctrl_pts[pairs[1]][0]]
            y_seg = [self.ctrl_pts[pairs[0]][1], self.ctrl_pts[pairs[1]][1]]
            x_.extend(x_seg)
            y_.extend(y_seg)
        self.ax.plot(x_, y_, "-g", label="control hull")
        # just drawing the line segments and projections
        self.ax.plot(
            [pt_proj[0], pt[0]],
            [pt_proj[1], pt[1]],
            marker="D",
            label=f"t_val: {t_reindex}",
            color="orange",
        )
        x_seg = [self.ctrl_pts[min_seg[0]][0], self.ctrl_pts[min_seg[1]][0]]
        y_seg = [self.ctrl_pts[min_seg[0]][1], self.ctrl_pts[min_seg[1]][1]]
        self.ax.plot(x_seg, y_seg, "-r", label="current")

        return t_reindex

    def project_to_curve(self, pt):
        """Project a point on the current spline

        :param pt: point to project
        """
        t = self.project_ctrl_hull(pt)
        t_result = fmin(
            self._get_distance_from_curve, np.array(t), args=(pt,)
        )  # limit to 10 TODO
        print(f"t val {t} refined to {t_result}")
        pt_proj = self.eval_crv(t_result)
        print(f"proj pt at real value {pt_proj}")
        self.ax.plot(
            [pt_proj[0], pt[0]],
            [pt_proj[1], pt[1]],
            marker="D",
            label=f"t_val: {t_result}",
            color="brown",
        )
        return pt

    def derivative(self, t: float) -> np.ndarray:
        """Get the value of the derivative of the spline at parameter t
        @param t - parameter
        @return derivative
        """
        idx = int(np.floor(t))
        t_prime = float(t) - float(idx)
        # print(f"idx {idx} t {t_prime}")

        if self.ctrl_pts is None or len(self.ctrl_pts) <= self.degree:
            raise ValueError(
                f"Need atleast degree + 1: {self.order} control points for creating a bezier curve"
            )
        try:
            control_points = np.reshape(self.ctrl_pts, (-1, self.dim))
            if idx < 0:  # return curve at left limit
                return np.matmul(
                    polyval(0.0, self.deriv_matrix),
                    control_points[: self.order, :],
                )
            if idx >= (len(self.ctrl_pts) - self.degree):  # return curve at right limit
                return np.matmul(
                    polyval(0.0, self.deriv_matrix),
                    control_points[len(self.ctrl_pts) - self.degree - 1 :, :],
                )

            return np.matmul(
                polyval(t_prime, self.deriv_matrix),
                control_points[idx : idx + self.order, :],
            )
        except ValueError as v:
            print(self.eval_basis(t_prime))
            print(np.reshape(self.ctrl_pts, (-1, self.dim)))
            raise ValueError(f"at {idx} Something went really wrong! {v}")

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

    def onclick(self, event):
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
            self.project_to_curve((ix, iy))
        elif event.button == 3:  # add control point RIGHT
            ix, iy = event.xdata, event.ydata
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            self.add_ctrl_point(np.array((ix, iy)))
            print(f"x {ix} y {iy} added")
            self.plot_curve()
        print("plotted")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.draw()

    def enable_onclick(self):
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
