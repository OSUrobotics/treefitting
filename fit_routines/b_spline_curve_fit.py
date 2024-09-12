#!/usr/bin/env python3
from __future__ import annotations

from tree_geometry.b_spline_curve import BSplineCurve
from bspline_fit_params import BSplineFitParams
from tree_geometry.point_lists import PointListWithTs, PointList
from eval_routines.bspline_fit_eval import BSplineFitEval


class BSplineCurveFit:
    def __init__(self,
                 pts_to_fit: PointList,
                 params: BSplineFitParams = None,
                 crv_start: BSplineCurve = None) -> None:
        """Initialize the curve fitting object
        @param pts_to_fit - points to fit to
        @param params - BSplineFit parameters (if overriding default)
        @param crv_start - use this curve for the first iteration
        """
        self.pts_to_fit_to = PointListWithTs(pts_to_fit.points())
        if params is None:
            self.params = BSplineFitParams()
        else:
            self.params = deepcopy(params)

        if crv_start is not None:
            self.crv_start = BSplineCurve(crv_start.points())
        else:
            self.crv_start = BSplineCurve(pts_to_fit.points())

        self.crv_fitted = BSplineCurve(self.crv_start.points())

    def __deepcopy__(self, memo):
        """Deep copy constructor for BSplineCurveFit"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @staticmethod
    def _setup_basic_lsq(crv: BSplineCurve, ts: np.ndarray, params: BSplineFitParams) -> np.ndarray:
        """Set up least squares problem for fitting a bspline curve to the parameterized points for
        arbitrary or minimum number of control points
        @param crv - the curve we're fitting to the points (for degree and number of control points)
        @param ts - t values for the least-squares basis
        @param params - add derivative constraints at beginning and end and/or keep control points
        :return: A matrix of Ax = B
        """
        ts_clamp = crv.clamp_t(ts)
        basis_matrix_for_pts = crv.get_banded_matrix(basis_to_use=crv.basis_matrix, t=ts_clamp)
        if params["end derivs"]:
            a_constraints = np.zeros((len(ts) + 2, crv.n_points()))
            a_constraints[0, 0:crv.order()] = crv.eval_basis(basis_to_use=crv.deriv_matrix, t=0.0)
            a_constraints[-1, -crv.order():] = crv.eval_basis(basis_to_use=crv.deriv_matrix, t=1.0)
            a_constraints[1:-1, :] = basis_matrix_for_pts
        else:
            a_constraints = basis_matrix_for_pts

        if params["weight ctrl pts"] > 0.0:
            a_constraints = np.stack(a_constraints,
                                     params["weight ctrl pts"] * np.identity(crv.n_points(), crv.n_points()))

        return a_constraints

    @staticmethod
    def _lsq_fit(crv: BSplineCurve,
                 params: BSplineFitParams,
                 pts_and_ts: PointListWithTs,
                 vecs: Union[None, np.ndarray]) -> (BSplineCurve, np.ndarray):
        """Fit the curve to the points with current params.
        @param crv - current curve
        @param pts_and_ts - points with t values
        @param vecs - start/end derivatives to set to. If none then use current curve derivs
        @return new curve and residuals of fit"""

        # get basic least squares constraints
        a_constraints = BSplineCurveFit._setup_basic_lsq(crv=crv, ts=pts_and_ts.ts, params=params)

        # Fill in the right hand side
        b_constraints = np.zeros((a_constraints.shape[0], crv.dim()))
        b_constraints[0:pts_and_ts.n_points(), :] = pts_and_ts.points_as_ndarray()
        if params["end derivs"]:
            if vecs is None:
                b_constraints[0, :] = crv.eval_deriv(0.0)
                b_constraints[pts_and_ts.n_points(), :] = crv.eval_deriv(crv.max_t())
            else:
                b_constraints[0, :] = vecs[0, :]
                b_constraints[pts_and_ts.n_points(), :] = vecs[1, :]
        if params["weight ctrl pts"] > 0.0:
            b_constraints[b_constraints.shape[0] - crv.n_points(), :] = crv.points_as_ndarray()

        # least squares fit
        ctrl_pts, residuals, _, __ = np.linalg.lstsq(a=a_constraints, b=b_constraints, rcond=None)

        # Make a new curve out of the resulting control points
        crv_fitted = BSplineCurve(ctrl_pts=ctrl_pts, degree=crv.degree_name())

        return crv_fitted, residuals

    @staticmethod
    def initial_fit(crv_start: BSplineCurve, pts: PointList) -> (BSplineCurve, PointListWithTs):
        """Fit the curve to the points with Ax = b, using the chord length parameterization
        @param crv_start - curve with desired number of control points and degree
        @param pts: x,y or x, y, z
        @return a new curve fitted to the points, the points with the ts used for the fit
        """
        # Default params are fine (no end derivatives, no keeping original points)
        params = BSplineFitParams()
        pts_and_ts = PointListWithTs(pts.points())
        pts_and_ts.normalize_ts(start_t=0.0, end_t=crv_start.max_t())

        crv_fit, _ = BSplineCurveFit._lsq_fit(crv=crv_start, params=params, pts_and_ts=pts_and_ts, vecs=None)
        return crv_fit, pts_and_ts

    @staticmethod
    def project_ts_fit(crv_initial_fit: BSplineCurve,
                       pts_and_ts: PointListWithTs,
                       params: BSplineFitParams) -> (BSplineCurve, PointListWithTs):
        """Fit the curve to the points; assumes crv is a good fit, and projects the points onto the curve
        @param crv_initial_fit - curve initially fit to points
        @param pts_and_ts: points with previous fit t values
        @param params - controls fit
        @return a new curve fitted to the points, the points with the ts used for the fit
        """

        ts = np.zeros(pts_and_ts.ts.shape)
        ts[-1] = crv_initial_fit.max_t()
        ts[1:-1] = pts_and_ts.ts[1:-1]
        for _ in range(0, 3):
            b_clipped = False
            for i, p in enumerate(pts_and_ts.points()[1:-1]):
                t, _, _ = crv_initial_fit.project_to_curve(p)
                clip_t_left = ts[i]
                clip_t_right = ts[i + 2]
                clip_delta_t = clip_t_right - clip_t_left
                clip_t_left += 0.1 * clip_delta_t
                clip_t_right -= 0.1 * clip_delta_t
                if t < clip_t_left:
                    t = clip_t_left
                    b_clipped = True
                if t > clip_t_right:
                    t = clip_t_right
                    b_clipped = True
                ts[i] = t
            if b_clipped == False:
                break

        if params["end derivs"] > 0.0:
            vecs = np.zeros((2, crv_initial_fit.dim()))
            pts = pts_and_ts.points_as_ndarray()
            ts = pts_and_ts.ts
            vecs[0] = ((pts[1, :] - pts[0, :]) / (ts[1] - ts[0]))
            vecs[1] = ((pts[-1, :] - pts[-2, :]) / (ts[-1] - ts[-2]))
        else:
            vecs = None

        pts_proj_ts = PointListWithTs(pts=pts_and_ts.points(), ts=ts)
        crv_fit, _ = BSplineCurveFit._lsq_fit(crv=crv_initial_fit, pts_and_ts=pts_proj_ts, vecs=vecs, params=params)
        return crv_fit, pts_proj_ts

    @staticmethod
    def fit_project_fit(crv_start: BSplineCurve,
                        pts: PointList,
                        params: BSplineFitParams = None) -> (BSplineCurve, PointListWithTs, BSplineFitParams):
        """ Fit the curve to the points twice, first using chord-length parameterization, then project ts
        @param crv_start - start curve with desired number of control points and degree,
        @param pts - initial points
        @param params - parameters controlling fit (use derivs y/n, use existing pts y/n)
        @return the fitted curve, the points with projected ts, and an evaluation of the fit"""

        if params is None:
            params = BSplineFitParams()
        crv_fit, pts_with_ts = BSplineCurveFit.initial_fit(crv_start, pts)
        crv_refit, pts_with_ts_refit = BSplineCurveFit.project_ts_fit(crv_fit, pts_with_ts, params)
        params["inlier threshold"] = 0.1 * crv_refit.curve_length() / crv_refit.n_points()
        crv_eval = BSplineFitEval(params)
        crv_eval.calc_values(crv_refit, pts_with_ts_refit)

        # One more re-project t values to curve
        #   Note: Keep first and last t values to prevent shrinking

        return crv_refit, pts_with_ts_refit, crv_eval

    @staticmethod
    def outlier_removal(pts: PointList, crv_degree: str = 'cubic', range_n_pts: float = 0.5) -> PointListWithTs:
        """Look for any points that are outliers/mess up the fit
         @param pts - initial points
         @param crv_degree - desired degree (linear, quadratic, cubic)
         @param range_n_pts - range of control points to try, as a percentage of the number of points
         @return points with outliers removed"""
        ...  # to write
        n_min_pts = int(range_n_pts * pts.n_points())
        p = pts.points()[0]
        pts = [p for _ in range(0, n_min_pts)]
        crv = BSplineCurve(degree=crv_degree, ctrl_pts=pts)
        return PointListWithTs(pts=pts.points())

    @staticmethod
    def fit_adjust_control_pts(crv_initial: BSplineCurve,
                               pts: PointList, params: BSplineFitParams
                               ) -> (BSplineCurve, PointListWithTs, BSplineFitEval):
        """ Use the fewest number of control points that produces a decent fit
        @param crv_initial - curve with desired degree
        @param pts: Points to fit to
        @param params - controls fit; inlier threshold determines when a point is an inlier, perc allowable outliers,
                     average fit
        @return a new curve fitted to the points, points with t values for the fit, evaluation of the fit
        """
        if params is None:
            params = BSplineFitParams()

        # Just keep adding points until we meet the threshold
        pt = pts.points()[0]
        pts_cntrol_hull = PointList([pt for _ in range(0, crv_initial.order())])
        while pts_cntrol_hull.n_points() <= pts.n_points():
            crv_initial = BSplineCurve(ctrl_pts=pts_cntrol_hull.points(), degree=crv_initial.degree_name())
            crv, pts_with_ts, eval_crv = BSplineCurveFit.fit_project_fit(crv_initial, pts, params)
            if eval_crv.is_acceptable():
                return crv, pts_with_ts, eval_crv
            pts_cntrol_hull.add_point(pt)

        crv_initial = BSplineCurve(ctrl_pts=pts_cntrol_hull.points(), degree=crv_initial.degree_name())
        return BSplineCurveFit.fit_project_fit(crv_initial, pts, params)

    @staticmethod
    def fit_ransac(crv_initial: BSplineCurve, pts: PointList, params: BSplineFitParams) -> (
        BSplineCurve, PointListWithTs, BSplineFitEval):

        # First fit the curve to all of the points as a base-line
        crv_start, pts_with_ts_start, eval_crv_start = BSplineCurveFit.fit_project_fit(crv_initial, pts, params)
        #
        n_to_keep = int(eval_crv_start.perc_inliers)

        results = [(crv_start, pts_with_ts_start, eval_crv_start)]
        # sample points
        crv_best = crv_start
        pts_best = pts_with_ts_start
        eval_best  = eval_crv_start
        for i in range(params["ransac iterations"]):
            indices = np.random
            sampled_points = np.random.sample(pts.points(), n_to_keep)
            pts_to_use = PointList(sampled_points)
            crv_try, pts_with_ts_try, eval_crv_try = BSplineCurveFit.fit_project_fit(crv_initial, pts_to_use, params)
            if eval_crv_try.is_better(eval_best):
                crv_best = crv_try
                pts_best = pts_to_use
                eval_best = eval_crv_try

            results.append((crv_try, pts_with_ts_try, eval_crv_try))
        # get best curve from results
        return crv_best, pts_best, eval_best

    @staticmethod
    def extend_curve(crv_initial: BSplineCurve, pts_initial: PointListWithTs,
                     pts_new: PointList, params: BSplineFitParams) -> (BSplineCurve, PointListWithTs, BSplineFitEval):
        """Extend the curve to fit the new data points.
        @param crv_initial: curve originally fit to pts_initial
        @param pts_initial: original points the curve was fit to
        @param pts_new: new points to add
        @param params: parameters controlling fit (use derivs y/n, use existing pts y/n)
        @return curve fit to new points by extending points
        """
        # Note: doing the fit with the *same* number of contol points will be handled outside of this. The assumption
        #   here is that we're adding one (or more) curve segments to handle the new points (i.e., by calling
        #   fit_adjust_control_pts with the extended list of points
        # Also assuming that the params indicating an acceptible fit (inlier fit, etc) are set for the
        #    new points already

        crv_extend = BSplineCurve(ctrl_pts=crv_initial.points(), degree=crv_initial.degree_name())
        pts_new_with_ts = PointListWithTs(pts_new.points())
        pts_all_with_ts = PointListWithTs(pts_initial.points().extend(pts_new.points()))
        pts_all_with_ts.ts()[0:pts_initial.n_points()] = pts_initial.ts()

        n_to_add = pts_new.n_points()
        results = []
        for _ in range(0, n_to_add):
            # Add a new control point and set the new ts to be along that last point
            crv_extend.add_point(pts_new.points()[-1])
            pts_new_with_ts.normalize_ts(crv_initial.max_t(), crv_extend.max_t())
            # Set the new points to have the extended ts
            pts_all_with_ts.ts()[pts_initial.n_points():] = pts_new_with_ts.ts()

            # Now the actual fit
            crv_fit, pts_fit, crv_eval_fit = BSplineCurveFit.project_ts_fit(crv_extend, pts_all_with_ts, params)
            results.append((crv_fit, pts_fit, crv_eval_fit))
            if crv_eval_fit.is_acceptable():
                return crv_fit, pts_fit, crv_eval_fit

        return results[-1]
        """
        new_fit = deepcopy(fit_obj)

        existing_in_t = new_fit.params.get_param("ts")
        if existing_in_t is None:
            raise ValueError("No curve to extend!")
        # logger.debug(f"extending curve of {len(existing_in_t)} points by {len(new_data_pts)}")
        new_points_in_t = np.zeros(len(new_data_pts), dtype=float)
        
        # calculate new points in t
        old_points = np.reshape(new_fit.params.get_param("lsq_points"), (-1, new_fit.dim))
        if len(new_data_pts) == 1:
            new_points_in_t = np.array((
                np.linalg.norm(new_data_pts[0] - old_points[-1])
                * new_fit.params.p_norm
            ) + existing_in_t[-1])
        else:
            points = BSplineCurve.flatten_dim(new_data_pts)
            _, new_points_in_t = new_fit._parameterize_chord(points, new_fit.params)
            new_points_in_t += existing_in_t[-1]
        points_in_t = np.zeros((len(existing_in_t) + len(new_data_pts)), dtype=float)
        points_in_t = np.hstack((existing_in_t, new_points_in_t))
        new_fit.params.update_param("ts", points_in_t)

        # get data points to fit
        b_constraints = np.zeros((len(points_in_t), new_fit.dim), dtype=float)
        b_constraints[: len(existing_in_t)] = new_fit.curve.eval_crv(existing_in_t) # key: do not refit existing points!
        b_constraints[len(existing_in_t) :] = new_data_pts

        # least squares fit
        num_ctrl_pts = int(np.ceil(points_in_t[-1] + new_fit.curve.degree))
        # logger.debug(f"new control point count {num_ctrl_pts}")
        return BSplineCurveFit._lsq_fit(new_fit, b_constraints, num_ctrl_pts)
        """


if __name__ == "__main__":
    import logging
    from typing import Tuple, Union, List, Callable
    from copy import deepcopy

    import numpy as np

    fit_pts = PointList([[0, 0], [1, 1], [2, -1], [3, 0]])

    params_basic = BSplineFitParams()
    params_ends = BSplineFitParams()
    params_ends["end derivs"] = True
    params_keep_pts = BSplineFitParams()
    params_keep_pts["weight ctrl pts"] = 0.1
    params_both = BSplineFitParams()
    params_both["end derivs"] = True
    params_both["weight ctrl pts"] = 0.1
    for param in [params_basic, params_ends, params_keep_pts, params_both]:
        for n in range(3, 5):
            crv_start = BSplineCurve(np.zeros((n, 2)), degree='quadratic')
            res_initial = BSplineCurveFit.initial_fit(crv_start, fit_pts)
            res_full = BSplineCurveFit.fit_project_fit(crv_start, fit_pts, param)
            res_adjust = BSplineCurveFit.fit_adjust_control_pts(crv_start, fit_pts, param)
