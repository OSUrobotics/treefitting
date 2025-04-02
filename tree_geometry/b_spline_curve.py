#!/usr/bin/env python3
"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""

import numpy as np
from typing import Union, Any
from numpy import floating
from numpy.polynomial.polynomial import polyval
from scipy.optimize import fmin
from scipy.integrate import quad

from tree_geometry.control_hull import ControlHull


# A b-spline curve (linear, quadratic, or cubic) defined as a sequence of control points (ControlHull)
# - Defined as f(t) -> [x,y,z] for t in 0,# pts - dim
# - Always a uniform knot vector
# Supports
#   Tacking on another control point
#   Projecting points onto the curve (for fitting)
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
        0: np.array([0]),                  # Shift the basis up by 1 and multiply by power of t
        1: np.array([[-1, 1],              # -1
                     [ 0, 0]]),            #  1
        2: 1 / 2 * np.array([[-2,   2, 0],    # 2 t - 2
                             [ 2,  -4, 2],    # -4 t + 2
                             [0,    0, 0]]),  # 2 t
        3: 1 / 6 * np.array([[-3,   0,  3, 0],  # -3 t^2 + 6 t - 3
                             [ 6, -12,  6, 0],  #  9 t^2 -12 t
                             [-3,   9, -9, 3],  # -9 t^2 + 6 t + 3
                             [0, 0, 0, 0]]),          # 3 t^2
    }
    _second_derivative_dict = {
        0: np.array([0]),                  # Shift the basis up by 1 and divide by power of t
        1: np.array([[0, 0],               # 0
                     [0, 0]]),             # 0
        2: 1 / 2 * np.array([[2, -4, 2],    # 2
                             [0,  0, 0],    # -4
                             [0,  0, 0]]),  # 2
        3: 1 / 6 * np.array([[ 6, -12,   6, 0],    # -6 t + 6
                             [-6,  18, -18, 6],    # 18 t - 12
                             [ 0,   0,   0, 0],    # -18 t + 6
                             [0, 0, 0, 0]]),       # 6 t
    }

    def __init__(self, ctrl_pts: Union[list[np.ndarray], np.ndarray, list[list]], degree: str = "quadratic") -> None:
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
        self._second_deriv_matrix: np.ndarray = BSplineCurve._second_derivative_dict[self._degree]

    def degree(self):
        return self._degree

    def n_segments(self):
        return self.n_points() - self._degree

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
        # Evaluate with derivative matrix instead of basis matrix
        eval_deriv_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._deriv_matrix, t=t_clamp - idx)
        if isinstance(t_clamp, float):
            return eval_deriv_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]

        res_vecs = np.zeros((len(t_clamp), self.dim()))
        for i_row, cp_id in enumerate(idx):
            res_vecs[i_row, :] = eval_deriv_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
        return res_vecs

    def eval_second_deriv(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Get the value of the second derivative of the spline at parameter t
        @param t - parameter
        @return derivative
        """
        t_clamp = self.clamp_t(t)
        idx = np.floor(t_clamp).astype(int)
        # Evaluate with second derivative matrix instead of basis matrix
        eval_second_deriv_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._second_deriv_matrix, t=t_clamp - idx)
        if isinstance(t_clamp, float):
            return eval_second_deriv_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]

        res_vecs = np.zeros((len(t_clamp), self.dim()))
        for i_row, cp_id in enumerate(idx):
            res_vecs[i_row, :] = eval_second_deriv_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
        return res_vecs

    def _norm(self, vec_deriv: np.ndarray, vec_second_deriv: np.ndarray):
        """ Get the unit-length normal (either 2D or 3D) from the tangent
        @param vec_deriv: Derivative
        @param vec_second_deriv: Second derivative vector (constant for dim 2)
        """
        if self.dim() == 1:
            return np.array([0.0])
        elif self.dim() == 2:
            vec_length = np.linalg.norm(vec_deriv)
            if np.isclose(vec_length, 0.0):
                return np.array([0.0, 0.0])
            return np.array([-vec_deriv[1] / vec_length, vec_deriv[0] / vec_length])
        elif self.dim() == 3:
            vec_binormal = np.cross(vec_deriv, vec_second_deriv)
            if np.isclose(np.linalg.norm(vec_second_deriv), 0.0):
                for i in range(0, 2):
                    if not np.isclose(vec_deriv[i], 0.0):
                        vec_binormal[i] = -vec_deriv[(i + 1) % 3]
                        vec_binormal[(i + 1) % 3] = vec_deriv[i]
                        vec_binormal[(i + 2) % 3] = 0.0
                        break

            return vec_binormal / np.linalg.norm(vec_binormal)
        raise ValueError("Curve not 1, 2, or 3 dimensions")

    def eval_norm(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Get the value of the normal of the derivative of the spline at parameter t
        Note: Not defined for 1 dimensional curves; for 2 dimensional, just rotate the derivative
        @param t - parameter
        :param dir: direction of normal, 0 for left, 1 for right
        @return normal of the curve
        """
        t_clamp = self.clamp_t(t)
        idx = np.floor(t_clamp).astype(int)
        # Evaluate with derivative matrix instead of basis matrix
        eval_deriv_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._deriv_matrix, t=t_clamp - idx)
        eval_second_deriv_basis_matrix = BSplineCurve.eval_basis(basis_to_use=self._second_deriv_matrix, t=t_clamp - idx)
        if isinstance(t_clamp, float):
            vec_deriv = eval_deriv_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]
            vec_second_deriv = eval_second_deriv_basis_matrix @ self.points_as_ndarray()[idx:idx+self.order()]
            return self._norm(vec_deriv, vec_second_deriv)

        res_vecs = np.zeros((len(t_clamp), self.dim()))
        for i_row, cp_id in enumerate(idx):
            vec_deriv = eval_deriv_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
            vec_second_deriv = eval_second_deriv_basis_matrix[i_row, :] @ self.points_as_ndarray()[cp_id:cp_id+self.order(), :]
            res_vecs[i_row, :] = self._norm(vec_deriv, vec_second_deriv)
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
                pts[:, d] = np.power(pts[:, d] - pt[d], 2)
            dists = np.sum(pts, axis=1)
            indx = np.argmin(dists)
            t_start = ts[indx]
        else:
            t_start = t

        # Standard fmin search for distance from curve
        t_min = fmin(self.get_distance_from_curve, np.array(t_start), args=(pt,), disp=False)  # limit to 10 TODO
        t_best = t_min[0]
        if t_best < 0.0 or t_best > self.max_t():
            t_best = t_start # Bail to the original t if fit went haywire

        pt_proj = self.eval_crv(t_best)
        return t_best, pt_proj, np.linalg.norm(pt_proj - pt)


if __name__ == "__main__":
    # np.set_printoptions(precision=3, suppress=True)

    cv = [[1, 3], [1, 2, -1], [1, 2, 2.5, -1]]
    checks = [[cv[0][0], cv[0][1], cv[0][1] - cv[0][0], cv[0][1] - cv[0][0]],
              [0.5*(cv[1][0] + cv[1][1]), 0.5*(cv[1][1] + cv[1][2]), (cv[1][1] - cv[1][0]), (cv[1][2] - cv[1][1])],
              [1.0/6.0*(cv[2][0] + cv[2][2]) + 4.0/6.0 * cv[2][1], 1.0/6.0*(cv[2][1] + cv[2][3]) + 4.0/6.0 * cv[2][2],
                0.5*(cv[2][2] - cv[2][0]), 0.5*(cv[2][3] - cv[2][1])] ]
    for dim_check in range(1, 3):
        for i_deg, deg_check in zip([2, 3, 4], ['linear', 'quadratic', 'cubic']):
            print(f"dimension {dim_check} degree {deg_check}")

            # First check - start point
            #.   Linear - start point is at first point
            #.   Quadratic - start point is half way between first and second point
            #.   Cubic - start point is 1/6 p0 + 4/6 p1 + 1/6 p2
            # Second check - end point
            #.   Linear - end point is last point
            #.   Quadratic - end point is half way between second to last and last point
            #.   Cubic - end point is 1/6 p1 + 4/6 p2 + 1/6 p3

            # Third check - start deriv
            #.   Linear - start deriv is end point - start point
            #.   Quadratic - start derivative is (p1 - p0)
            #.   Cubic - start derivative is 1/2 (p2 - p0)

            # Fourth check - 3nd deriv
            #.   Linear - end deriv is end point - start point
            #.   Quadratic - end derivative is (p2 - p1)
            #.   Cubic - start derivative is 1/2 (p3 - p1)

            # Need n+1 control points
            cntrl_hull = []
            for i_pt in range(0, i_deg):
                pt_add = np.ones(dim_check)
                pt_add[0] *= cv[i_deg-2][i_pt]
                if pt_add.size > 1:
                    pt_add[1] = i_pt + 0.5
                if pt_add.size > 2:
                    pt_add[2] = 3.0 - pt_add[0] * 0.5
                cntrl_hull.append(pt_add)

            crv_check = BSplineCurve(ctrl_pts=cntrl_hull, degree=deg_check)
            t_check = [0.0, 1.0]
            pts_check = crv_check.eval_crv(np.array(t_check))
            deriv_check = crv_check.eval_deriv(np.array(t_check))

            assert np.isclose(pts_check[0][0], checks[i_deg-2][0])
            assert np.isclose(pts_check[-1][0], checks[i_deg-2][1])
            assert np.isclose(deriv_check[0][0], checks[i_deg-2][2])
            assert np.isclose(deriv_check[-1][0], checks[i_deg-2][3])

            # Check derivs for a sequence of t's
            eps = 0.00001
            t_deriv_check = np.array([0.0, 0.25, 0.5, 0.75, 1.0 - 2 * eps])
            deriv_vecs = crv_check.eval_deriv(t_deriv_check)
            for tc in t_deriv_check:
                deriv_vec = crv_check.eval_deriv(tc)
                pt_at_t = crv_check.eval_crv(tc)
                pt_at_t_plus_eps = crv_check.eval_crv(tc + eps)
                deriv_check_vec = (pt_at_t_plus_eps - pt_at_t) / eps
                assert np.allclose(deriv_vec, deriv_check_vec, atol=10.0 * eps)

                deriv_vec_next = crv_check.eval_deriv(tc + eps)
                second_deriv_check_vec = (deriv_vec_next - deriv_vec) / eps
                second_deriv_vec = crv_check.eval_second_deriv(tc)
                assert np.allclose(second_deriv_vec, second_deriv_check_vec, atol=10.0 * eps)

                if dim_check > 1:
                    vec_norm = crv_check.eval_norm(tc)
                    dot_prod = np.dot(vec_norm, deriv_vec)
                    assert np.isclose(dot_prod, 0.0, eps)

            crv_len = crv_check.curve_length()
            crv_len_check = 0.0
            pts_along_crv = crv_check.eval_crv(np.linspace(0, 1.0, 100))
            for ind_pt in range(0, pts_along_crv.shape[0]-1):
                pt = pts_along_crv[ind_pt, :]
                pt_next = pts_along_crv[ind_pt+1, :]
                crv_len_check += np.linalg.norm(pt_next - pt)
            crv_hull_len = crv_check.hull_length()
            assert crv_len <= crv_hull_len + eps
            assert np.isclose(crv_len_check, crv_len, atol=0.0001)

            t_mid = 0.5
            pt_mid = crv_check.eval_crv(t_mid)
            vec_norm = crv_check.eval_norm(t_mid)

            # Go off the curve along the normal and projecting back onto the curve should give you the same point
            if dim_check > 1:
                res_project_check = crv_check.project_to_curve(pt_mid + vec_norm * 0.01)
                assert np.isclose(res_project_check[0], 0.5, atol=0.01)
                assert np.isclose(pt_mid[0], res_project_check[1][0], atol=0.01)
                assert np.isclose(0.01, res_project_check[2], atol=0.01)

            assert crv_check.degree_name() == deg_check

        print("\n")

