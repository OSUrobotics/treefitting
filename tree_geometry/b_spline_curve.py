#!/usr/bin/env python3
from typing import Union, Any

from numpy import floating

"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.optimize import fmin
from scipy.integrate import quad

from point_lists import ControlHull


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
        2: 1 / 2 * np.array([[1, 1, 0],       # t^2 - 2t + 1
                             [-2, 2, 0],      # -2 t^2 + 2t + 1
                             [1, -2, 1]]),    # t^2
        3: 1 / 6 * np.array([[ 1,  4,  1, 0],   # -t^3 + 3t^2 -3t   + 1
                             [-3,  0,  3, 0],   # 3t^3 - 6t^2       + 4
                             [ 3, -6,  3, 0],   # -3t^3 + 3t^2 + 3t + 1
                             [-1,  3, -3, 1]]), # t^3
    }
    _derivative_dict = {
        0: np.array([0]),                  # Shift the basis up by 1 and divide
        1: np.array([[-1, 1],              # by power of t
                     [ 0, 0]]),
        2: 1 / 2 * np.array([[-2,   2, 0],
                             [1/2, -2/2, 1/2],
                             [0, 0, 0]]),
        3: 1 / 6 * np.array([[-3, 0, 3, 0],
                             [3/2, -6/2, 3/2, 0],
                             [-1/3, 3/3, -3/3, 1/3],
                             [0, 0, 0, 0]]),
    }

    def __init__(self, ctrl_pts: Union[list[np.ndarray], np.ndarray], degree: str = "quadratic") -> None:
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

    @property
    def basis_matrix(self):
        return self._basis_matrix

    @property
    def deriv_matrix(self):
        return self._deriv_matrix

    def degree_name(self):
        """ Convert degree back to name"""
        for i, k in enumerate(BSplineCurve._degree_dict.keys()):
            if i == self._degree - 1:
                return k
        return "Uknown degree"

    def order(self):
        return self._degree + 1

    def max_t(self):
        """ parameterization goes from 0 to number of control points minus degree"""
        return self.n_points() - self._degree

    @staticmethod
    def eval_basis(basis_to_use, t: Union[float, np.ndarray]) -> np.ndarray:
        """evaluate basis functions at t
        @param basis_to_use: basis functions to evaluate (regular or deriv
        @param t: t values, must be between 0 and 1
        @return: basis matrix at t, each column is a basis function
        """
        #  This is how polyval evaluates a matrix
        # for i in range(0, basis_to_use.shape[0]):
        #     print(polyval(t, basis_to_use[:, i]))
        # Return transpose to get each row corresponding to t value for each basis
        #   One row for each t value given
        return polyval(t, basis_to_use).T

    def clamp_t(self, t_in):
        """ Clamp the t value between 0 and max_t
        @param t_in: float or np.ndarray
        @return t_in between 0 and max_t"""
        if isinstance(t_in, float):
            return min(max(t_in, 0.0), self.max_t() - 0.000001)

        return np.clip(t_in, a_min=0.0, a_max=self.max_t() - 0.0000001)

    def get_banded_matrix(self, basis_to_use, t: np.ndarray) -> np.ndarray:
        """get the banded matrix for the spline
        on multiplying this matrix with control points, we get the curve at t.
        it is zeroed out for control points that don't contribute to the curve at t
        @param basis_to_use: basis functions to evaluate (regular or deriv)
        @param t: t values
        @return: banded matrix, one row for each t, one column for each control point
        """

        banded_basis_matrix = np.zeros((len(t), self.n_points()), dtype=float)
        # Make sure t's are in valid range - clamp to [0, max_t)
        t_clip = self.clamp_t(t)

        # Which index to start at
        idxs = np.floor(t_clip).astype(int)
        # t in the range 0..1 (for calling eval_basis)
        t_prime = t_clip - idxs

        # The basis functions evaluated at each t value
        evaluated_basis = self.eval_basis(basis_to_use, t_prime)
        # construct diagonal banded matrix -
        for i in range(len(t)):
            banded_basis_matrix[i, idxs[i]: (idxs[i] + self.order())] = evaluated_basis[i, :]
        return banded_basis_matrix

    def eval_crv(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - float or list of floats; only values between 0 and max_t are valid pts on curve
        @return point on spline of dimension self.dim
        """
        t_clamp = self.clamp_t(t)
        idx = np.floor(t_clamp).astype(int)
        eval_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._basis_matrix, t=t_clamp - idx)
        if isinstance(t_clamp, float):
            return eval_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]

        res_pts = np.zeros((len(t_clamp), self.dim()))
        for i_row, cp_id in enumerate(idx):
            res_pts[i_row, :] = eval_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
        return res_pts

    def eval_deriv(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Get the value of the derivative of the spline at parameter t
        @param t - parameter
        @return derivative
        """
        t_clamp = self.clamp_t(t)
        idx = np.floor(t_clamp).astype(int)
        eval_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._deriv_matrix, t=t_clamp - idx)
        if isinstance(t_clamp, float):
            return eval_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]

        res_vecs = np.zeros((len(t_clamp), self.dim()))
        for i_row, cp_id in enumerate(idx):
            res_vecs[i_row, :] = eval_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
        return res_vecs

    def curve_length(self):
        """Get curve length using integration of the norm of derivative of curve"""
        def f(t):
            return np.linalg.norm(self.eval_deriv(t))

        res = quad(f, a=0.0, b=self.max_t())
        return res[0]

    def get_distance_from_curve(self, t: np.ndarray, pt: np.ndarray) -> floating[Any]:
        """Get distance from curve at param t for point, convenience function for using with scipy optimization
        @param t: t value
        @param pt: point to get distance from curve at
        @return: distance
        """
        res_pt = self.eval_crv(t)
        return np.linalg.norm(res_pt - pt)

    def project_ctrl_hull(self, pt) -> float:
        """Get t value for projecting point on hull

        :param pt: point to project
        :return: t value
        """
        t, _, min_seg = self.project_on_hull(pt)
        if min_seg == -1:
            raise ValueError(f"Could not project {pt} on hull")

        return t + min_seg

    def project_to_curve(self, pt :Union[list, np.ndarray], t :float = None) -> (float, np.ndarray):
        """Project a point on the current spline
        @param pt: point to project
        @param t: if t is given, use as the starting point for fmin
        @return t value at min, point, and distance
        """
        # Best guess from control hull
        # t = self.project_ctrl_hull(pt)

        if t is None:
            # Sampling of points along curve
            ts = np.linspace(0, self.max_t(), self.n_points() * 4)
            pts = self.eval_crv(ts)
            # Distance calculation
            for d in range(0, self.dim()):
                pts[:, d] = np.pow(pts[:, d] - pt[d], 2)
            dists = np.sum(pts, axis=1)
            indx = np.argmin(dists)
            t_start = ts[indx]
        else:
            t_start = t

        # Standard fmin search for distance from curve
        t_min = fmin(self.get_distance_from_curve, np.array(t_start), args=(pt,), disp=False)  # limit to 10 TODO
        if t_min[0] < 0.0 or t_min[0] > self.max_t():
            t_min[0] = t_start  # Bail to the original t if fit went haywire

        pt_proj = self.eval_crv(t_min[0])
        return t_min[0], pt_proj, np.linalg.norm(pt_proj - pt)


if __name__ == "__main__":
    # np.set_printoptions(precision=3, suppress=True)

    for dim_check in range(1, 3):
        for deg_check in ['linear', 'quadratic', 'cubic']:
            print(f"dimension {dim_check} degree {deg_check}")
            cntrl_hull = []
            for i_pt in range(0, 4):
                cntrl_hull.append(i_pt * np.ones(dim_check))
            crv_check = BSplineCurve(ctrl_pts=cntrl_hull, degree=deg_check)
            pts_check = crv_check.eval_crv(np.array([0.0, 0.25, 0.5, 0.75, 0.999999999]))

            crv_check.eval_crv(np.linspace(0.0, crv_check.max_t(), 20))
            deriv_vecs = crv_check.eval_deriv(np.array([0.0, 0.25, 0.5, 0.75, 0.999999999]))
            pt_mid = crv_check.eval_crv(0.4)
            pt_mid_next = crv_check.eval_crv(0.41)
            vec_check = (pt_mid_next - pt_mid) / 0.01
            mid_vec = crv_check.eval_deriv(0.4)

            assert np.isclose(vec_check, mid_vec).all()

            crv_len = crv_check.curve_length()
            assert crv_check.curve_length() <= crv_check.hull_length()

            res_project_check = crv_check.project_to_curve(pt_mid)

            assert crv_check.degree_name() == deg_check

            print(f" pt mid {pt_mid} res t {res_project_check[0]}, res p {res_project_check[1]} {np.linalg.norm(res_project_check[1] - pt_mid)}")
            print(f"   Deriv {deriv_vecs[0]}")
            assert np.isclose(pt_mid, res_project_check[1], atol=0.01).all()
            assert np.isclose(a=0.4, b=res_project_check[0], atol=0.01)

        print("\n")

