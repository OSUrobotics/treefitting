#!/usr/bin/env python3
from typing import Union

"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.optimize import fmin
from scipy.integrate import quad

from geom_utils import ControlHull


class BSplineCurve(ControlHull):
    # For initializing with a name instead of a dimension
    _degree_dict = dict(
        linear=1,
        quadratic=2,
        cubic=3,
    )
    # uniform knot vector only
    # General Matrix Representations for B-Splines, Kaihuai Qin
    # each matrix column represents a series of coeffs of increasing degree for a basis function
    _basis_matrix_dict = {
        0: np.array([1]),             # Note: These are COLUMNS, with constant value on top
        1: np.array([[ 1,  0],         # 1-t
                     [-1,  1]]),       # t
        2: 1 / 2 * np.array([[1, 1, 0],    # t^2 - 2t + 1
                             [-2, 2, 0],   # -2 t^2 + 2t + 1
                             [1, -2, 1]]), # t^2
        3: 1 / 6 * np.array([[ 1,  4,  1, 0],   # -t^3 + 3t^2 -3t   + 1
                             [-3,  0,  3, 0],   # 3t^3 - 6t^2       + 4
                             [ 3, -6,  3, 0],   # -3t^3 + 3t^2 + 3t + 1
                             [-1,  3, -3, 1]]), # t^3
    }
    _derivative_dict = {
        0: np.array([0]),
        1: np.array([[-1, 1],
                     [ 0, 0]]),
        2: 1 / 2 * np.array([[-2,   2, 0],
                             [1/2, -2/2, 1/2],
                             [0, 0, 0]]),
        3: 1 / 6 * np.array([[-3, 0, 3, 0],
                             [3/2, -6/2, 3/2, 0],
                             [-1/3, 3/3, -3/3, 1/3],
                             [0, 0, 0, 0]]),
    }

    # ROADMAP
    # curve tangent, normal binormal for frenet frames
    # apply a filter -> move to middle
    # integrate radius of curvature over the curve as metric for curviness

    def __init__(self, ctrl_pts: list[np.ndarray], degree: str="quadratic") -> None:
        """BSpline initialization
        :param ctrl_pts: control points, list of numpy array points of desired dimension
        :param degree: degree of spline, defaults to "quadratic"
        """
        # Sets control points
        super().__init__(ctrl_pts)

        self._degree: int = self._degree_dict[degree]

        if self.n_points() < self._degree + 1:
            raise ValueError(f"Must have at least {self._degree + 1} control points")

        # Set the bases matrics for the given degree
        self._basis_matrix: np.ndarray = BSplineCurve._basis_matrix_dict[self._degree]
        self._deriv_matrix: np.ndarray = BSplineCurve._derivative_dict[self._degree]

    def degree(self):
        return self._degree

    def order(self):
        return self._degree + 1

    def max_t(self):
        """ parameterization goes from 0 to number of control points minus degree"""
        return self.n_points() - self._degree

    def _eval_basis(self, basis_to_use, t: Union[float, np.ndarray]) -> np.ndarray:
        """evaluate basis functions at t
        @param basis_to_use: basis functions to evaluate (regular or deriv
        :param t: parameter
        :type t: float or np.ndarray
        :return: basis matrix at t, each column is a basis function
        :rtype: np.ndarray
        """
        #  This is how polyval evaluates a matrix
        # for i in range(0, basis_to_use.shape[0]):
        #     print(polyval(t, basis_to_use[:, i]))

        return polyval(t, basis_to_use)

    def get_banded_matrix(self, basis_to_use, t: np.ndarray) -> np.ndarray:
        """get the banded matrix for the spline

        on multiplying this matrix with control points, we get the curve at t.
        it is zeroed out for control points that don't contribute to the curve at t
        @param basis_to_use: basis functions to evaluate (regular or deriv)
        :param t: t values
        :type t: np.ndarray
        :return: banded matrix
        :rtype: np.ndarray
        """

        banded_basis_matrix = np.zeros((len(t), self.n_points()), dtype=float)
        floor = np.floor(t)
        idxs = floor.astype(int)
        t_prime = t - floor

        # construct diagonal banded matrix
        ids_safe = floor.astype(int)
        ids_safe[idxs < 0] = 0
        ids_safe[idxs >= (self.n_points() - self.degree())] = self.n_points() - self.order()
        t_prime[idxs < 0] = 0.0
        t_prime[idxs >= (self.n_points() - self.degree())] = 1.0
        evaluated_basis = self._eval_basis(basis_to_use, t_prime).T
        # print(f"t basis {evaluated_basis}")
        for i in range(len(t)):
            # dsum = np.sum(evaluated_basis[i, :])
            # if not np.isclose(dsum, 1.0):
                # print(f"Bad basis {dsum} {evaluated_basis[i, :]}")
            banded_basis_matrix[i, ids_safe[i]: (ids_safe[i] + self.order())] = evaluated_basis[i, :]
        return banded_basis_matrix

    def eval_crv(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - float or list of floats; only values between 0 and max_t are valid pts on curve
        @return point on spline of dimension self.dim
        """
        if isinstance(t, float):
            t = [t]

        try:
            res = np.matmul(self.get_banded_matrix(self._basis_matrix, t), self.points_as_ndarray())
            if res.shape[0] == 1:  # if we needed only point, return it as tuple
                return res[0]
            return res
        except ValueError as v:
            raise ValueError(
                f"Something went really wrong! {v} multiplying {self.get_banded_matrix(self._basis_matrix, t).shape} with {self.points_as_ndarray().shape}"
            )

    def eval_deriv(self, t: float) -> np.ndarray:
        """Get the value of the derivative of the spline at parameter t
        @param t - parameter
        @return derivative
        """
        if isinstance(t, float):
            t = [t]

        res = np.matmul(self.get_banded_matrix(self._deriv_matrix, t), self.points_as_ndarray())
        if res.shape[0] == 1:  # if we needed only point, return it as tuple
            return res[0]
        return res

    def curve_length(self):
        """Get curve length using integration of the norm of derivative of curve
        we prefer to use fast approximate curve length provided by ctrl hull"""
        a, b = 0.0, self.max_t()

        def f(t):
            return np.linalg.norm(self.eval_deriv(t))

        length, _ = quad(f, a, b)
        return length

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
        t, _, min_seg = self.project_on_hull(pt)
        if min_seg == -1:
            raise ValueError(f"Could not project {pt} on hull")

        return t + min_seg

    def project_to_curve(self, pt):
        """Project a point on the current spline
        :param pt: point to project
        """
        # Best guess from control hull
        # t = self.project_ctrl_hull(pt)

        ts = np.linspace(0, self.max_t(), self.n_points() * 4)
        pts = self.eval_crv(ts)
        for d in range(0, self.dim()):
            pts[:, d] = np.pow(pts[:, d] - pt[d], 2)
        dists = np.sum(pts, axis=1)
        d_min = np.min(dists)
        indx = np.where(dists == d_min)
        t_start = ts[indx[0][0]]

        # Standard fmin search for distance from curve
        t_result = fmin(self._get_distance_from_curve, np.array(t_start), args=(pt,), disp=False)  # limit to 10 TODO

        pt_proj = self.eval_crv(t_result)
        return t_result[0], pt_proj


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    for dim_check in range(1, 3):
        for deg_check in ['linear', 'quadratic', 'cubic']:
            print(f"dimension {dim_check} degree {deg_check}")
            cntrl_hull = []
            for i in range(0, 4):
                cntrl_hull.append(i * np.ones(dim_check))
            crv_check = BSplineCurve(ctrl_pts=cntrl_hull, degree=deg_check)
            pts = crv_check.eval_crv(np.array([0.0, 0.25, 0.5, 0.75, 0.999999999]))

            pt_mid = crv_check.eval_crv(crv_check.max_t() / 2)
            crv_check.eval_crv(np.linspace(0.0, crv_check.max_t(), 20))
            deriv_vecs = crv_check.eval_deriv(np.array([0.0, 0.25, 0.5, 0.75, 0.999999999]))
            pt_mid = crv_check.eval_crv(0.4)
            pt_mid_next = crv_check.eval_crv(0.41)
            vec_check = (pt_mid_next - pt_mid) / 0.01
            mid_vec = crv_check.eval_deriv(0.4)

            assert( np.isclose(vec_check, mid_vec).all() )

            assert( crv_check.curve_length() <= crv_check.hull_length() )

            res = crv_check.project_to_curve(pt_mid)

            print(f" pt mid {pt_mid} res t {res[0]}, res p {res[1]} {np.linalg.norm(res[1] - pt_mid)}")
            print(f"   Deriv {deriv_vecs[0]}")
            assert( np.isclose(pt_mid, res[1], atol=0.01).all() )
            assert( np.isclose(0.4, res[0], atol=0.01).all() )

        print("\n")

