from __future__ import annotations
import logging
from typing import Tuple, Union, List, Callable
from copy import deepcopy

import numpy as np

from tree_geometry.b_spline_curve import BSplineCurve
from params import fit_params

logger = logging.getLogger("b_spline_fit")
logger.setLevel(level=logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
np.set_printoptions(precision=3, suppress=True)

class BSplineCurveFit:
    def __init__(
        fit_obj,
        original: BSplineCurve = None,
        params: fit_params = fit_params(),
        points_to_fit: list[np.ndarray] = [],
    ) -> None:
        """Initialize the curve fitting object

        :param original: original curve, defaults to None
        :param fit_params: parameters for fitting data, defaults to fit_params()
        :param points_to_fit: points to fit the curve to, defaults to []
        """
        fit_obj.curve = original
        fit_obj.points_to_fit: list[np.ndarray] = points_to_fit
        fit_obj.params = deepcopy(params)

    @property
    def dim(fit_obj) -> int:
        if fit_obj.curve.is_initialized:
            return fit_obj.curve.dim
        else:
            return len(fit_obj.points_to_fit[0])

    def __deepcopy__(fit_obj, memo):
        """Copy constructor for BSplineCurveFit"""
        new_curve = deepcopy(fit_obj.curve)
        new_params = deepcopy(fit_obj.params)
        new_points = deepcopy(fit_obj.points_to_fit)
        return BSplineCurveFit(
            original=new_curve, params=new_params, points_to_fit=new_points
        )

    @staticmethod
    def _parameterize_chord(
        points: Union[list[np.ndarray], np.ndarray],
        params: fit_params,
        renorm: bool = False,
        start: int = 0,
        stop: int = 1,
    ) -> Tuple[fit_params, np.ndarray]:
        """Get chord length parameterization of euclidean points

        :param points: points to parameterize
        :param params: fitting parameters
        :param renorm: whether to renormalize the parameterization, defaults to False
        :param start: t value to start, defaults to 0
        :param stop: t value to stop, defaults to 1
        :return: tuple(params, array of parameterized points)
        """
        distances = [
            np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))
        ]
        distances.insert(0, 0.0)
        parameterized = np.cumsum(distances)
        if renorm is True or not params.is_param("p_norm"):
            p_norm = (stop - start) / (parameterized[-1])
            params.update_param("p_norm", p_norm)
        parameterized = start + params.get_param("p_norm") * parameterized
        params.update_param("ts", parameterized)
        return params, parameterized

    @staticmethod
    def _setup_basic_lsq(
        fit_obj: BSplineCurveFit, num_ctrl_pts: Union[None, int] = None
    ) -> Tuple[BSplineCurveFit, np.ndarray]:
        """Set up least squares problem for fitting a bspline curve to the parameterized points for
        arbitrary or minimum number of control points

        :param fit_obj: curve fitting object
        :return: tuple of new fit object and points in t
        """
        degree = fit_obj.curve.degree
        ts = fit_obj.params.get_param("ts")
        if ts is None:
            raise ValueError("No parameterization found")
        last_t = int(np.ceil(ts[-1]))
        if num_ctrl_pts is None:  # get minimum number of control points
            num_ctrl_pts = last_t + degree
        elif last_t + degree > num_ctrl_pts:
            raise ValueError("Number of control points too few for the parametrization")
        a_constraints = fit_obj.curve.get_banded_matrix(ts, num_ctrl_pts)
        fit_obj.params.update_param("num_ctrl_pts", num_ctrl_pts)
        return fit_obj, a_constraints

    @staticmethod
    def _lsq_fit(fit_obj, points, num_ctrl_pts: Union[None, int] = None) -> BSplineCurveFit:
        """Fit the curve to the points with current params. Does not preserve original fit!"""
        # get basic least squares constraints
        fit_obj, a_constraints = BSplineCurveFit._setup_basic_lsq(fit_obj, num_ctrl_pts)
    
        logger.debug(
            f" A = \n{a_constraints}\n B = \n{BSplineCurve.unflatten_dim(points, fit_obj.dim)}")

        # least squares fit
        ctrl_pts, residuals, _, __ = np.linalg.lstsq(
            a=a_constraints, b=points, rcond=None
        )
        residuals = np.linalg.norm(points - np.dot(a_constraints, ctrl_pts), axis=1)

        # update the fit object
        fit_obj.curve.ctrl_pts = deepcopy(ctrl_pts)
        fit_obj.points_to_fit = deepcopy(points)
        fit_obj.params.update_param("residuals", residuals)
        fit_obj.params.update_param("lsq_points", points)
        logger.debug(f"lsq fit: {fit_obj.params.__dict__}")
        return fit_obj

    @staticmethod
    def simple_fit(fit_obj: BSplineCurveFit, points: list[np.ndarray]) -> BSplineCurveFit:
        """Fit the curve to the points with current or default parameterization

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        new_fit = deepcopy(fit_obj)
        # parameterization
        if new_fit.params.get_param("ts") is None:
            new_fit.params, _ = BSplineCurveFit._parameterize_chord(points, params=new_fit.params)

        return BSplineCurveFit._lsq_fit(new_fit, points)

    @staticmethod
    def renorm_fit(fit_obj: BSplineCurveFit, points: list[np.ndarray], start: int = 0, stop: int = 1) -> BSplineCurveFit:
        """Fit the curve to the points with renormalized parameterization
        
        :param fit_obj: curve fitting object
        :type fit_obj: BSplineCurveFit
        :param points: x,y or xyz
        :type points: list[np.ndarray]
        :param start: start parameterization at t value, defaults to 0
        :type start: int, optional
        :param stop: stop parameterization at t value, defaults to 1
        :type stop: int, optional
        """
        new_fit = deepcopy(fit_obj)
        new_params, _ = BSplineCurveFit._parameterize_chord(
            points, params=new_fit.params, renorm=True, start=start, stop=stop
        )
        new_fit.params.__dict__ = new_params.__dict__
        logger.debug(f"renorm fit: {new_fit.params.__dict__}")
        return BSplineCurveFit._lsq_fit(new_fit, points)

    @staticmethod
    def one_segment_fit(fit_obj, points: list[np.ndarray]) -> BSplineCurveFit:
        """Fit a simple one_segment curve, ie t = [0, 1) to the points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        logger.debug("one segment fit")
        return BSplineCurveFit.renorm_fit(fit_obj, points, 0, 1)

    @staticmethod
    def iteratively_fit(
        fit_obj,
        points: list[np.ndarray],
        outlier_ratio: float = 0.1,
        inlier_threshold: float = 0.1,
        max_iter: int = 20,
        max_ctrl_ratio = 10,
    ):
        """Iteratively fit the curve to the points using RANSAC type approach

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        results = []
        # sample points
        for i in range(max_iter):
            sampled_points = np.random.sample(points, int(len(points) * outlier_ratio))
            # get fit for subset
            new_fit = BSplineCurveFit.simple_fit(fit_obj, sampled_points)
            new_fit.points_to_fit = points

            # check all
            all_points_in_t = BSplineCurveFit._parameterize_chord(points, new_fit.params)
            new_fit, a_constraints = BSplineCurveFit._setup_basic_lsq(all_points_in_t)
            residuals = np.linalg.norm(points - np.dot(a_constraints, new_fit.curve.ctrl_pts), axis=1)
            new_fit.params.update_param("residuals", residuals)

            results.append((new_fit, residuals))
        # get best curve from results
        min_result, min_residual = min(results, key=lambda x: np.sum(x[-1]))
        min_result.params.add_param("outlier_ratio", outlier_ratio)
        return min_result

    @staticmethod
    def extend_curve(fit_obj: BSplineCurveFit, new_data_pts: list[np.ndarray]):
        """Extend the curve to fit the new data points
        :param new_data_pts: new data points
        :type new_data_pts: list[np.ndarray]
        """
        new_fit = deepcopy(fit_obj)

        existing_in_t = new_fit.params.get_param("ts")
        if existing_in_t is None:
            raise ValueError("No curve to extend!")
        logger.debug(f"extending curve of {len(existing_in_t)} points by {len(new_data_pts)}")
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
        logger.debug(f"new control point count {num_ctrl_pts}")
        return BSplineCurveFit._lsq_fit(new_fit, b_constraints, num_ctrl_pts)

    @staticmethod
    def evaluate(
        new_fit, last_good_fit = None, residual_min=1e-6, residual_max=1e-2, max_control_ratio=10, max_curve_diff = 0.2
    ) -> tuple[bool, Union[BSplineCurve, Tuple[Callable, List]]]:
        """Evaluate the result of the fit and recommend next steps if not satisfactory

        :param fit_obj: tuple of control points, points in t and residuals
        :param last_good_fit: last good fit for reuse in case of extension
        :param residual_min: residual threshold per segment, defaults to 1e-6
        :param residual_max: residual threshold per segment, defaults to 0.01
        :param max_control_ratio: no of points per control pt, defaults to 10
        :param max_curve_diff: ratio of max difference between curve and hull length, defaults to 0.2
        :return: (validity, (next function, [args]))
        """
        ctrl_pts, points_in_t, residuals = (
            new_fit.curve.ctrl_pts,
            new_fit.params.get_param("ts"),
            new_fit.params.get_param("residuals"),
        )
        min_t = int(np.floor(min(points_in_t)))
        max_t = int(np.ceil(max(points_in_t)))
        logger.debug(f"eval minmax: {min_t, max_t}")

        # bucket residuals
        t_buckets = np.array(range(min_t, max_t + 1))
        segmentwise_residual = []
        for i in range(len(t_buckets) - 1):
            cond = np.asarray(
                (points_in_t >= t_buckets[i]) & (points_in_t < t_buckets[i + 1])
            )
            segmentwise_residual.append(np.sum(residuals[cond.nonzero()]))
        logger.info(f"residuals: {residuals}, seg: {segmentwise_residual} buckets: {t_buckets}")
        segmentwise_residual = np.array(segmentwise_residual)
        new_fit.params.add_param("seg_residuals", segmentwise_residual)

        # check segmentwise residuals
        max_residual = segmentwise_residual < residual_max
        min_residual = segmentwise_residual > residual_min
        check_residual = max_residual & min_residual
        if check_residual.all() or (max_t == 1 and max_residual.all()):
            logger.info(f"Valid spline with residuals {segmentwise_residual}\n\n")
            return True, new_fit
        elif not max_residual.all():
            indices = np.asarray(max_residual == False).nonzero()
            logger.warning(f"Residuals too high in segments {t_buckets[indices]}")
            # if residuals high towards the end, extend curve if last fit is good
            if last_good_fit is not None:
                last_good_points = BSplineCurve.flatten_dim(last_good_fit.params.get_param("lsq_points"))
                curr_points = BSplineCurve.flatten_dim(new_fit.params.get_param("lsq_points"))
                new_points = curr_points[len(last_good_points):]
                # remove common points from last_good_points and curr_points
                if len(new_points) > 0 and np.mean(t_buckets[indices]) >= 0.8 * (max_t - 1):
                    return False, (
                        BSplineCurveFit.extend_curve,
                        [last_good_fit,
                        new_points],
                    )
            logger.info(f'Adding new control point to play with')
            return False, (
                BSplineCurveFit.renorm_fit,
                [new_fit, new_fit.points_to_fit, min_t, max_t + 1]
            )

        elif not min_residual.all(): # check overfitting
            if len(new_fit.points_to_fit) / max_control_ratio > len(ctrl_pts):
                logger.warning(
                    f"Overfitting detected by too many control points {len(ctrl_pts)} for data points {len(new_fit.data_pts)}"
                )
                return False, (
                    BSplineCurveFit.renorm_fit,
                    [new_fit, new_fit.points_to_fit, min_t, max_t - 1],
                )
            # else:
            #     ctrl_hull_length = fit_obj.curve.ctrl_hull_length
            #     logger.warning(f"ctrl hull length {ctrl_hull_length}")
            #     diff_of_curves = abs(fit_obj.curve.curve_length - ctrl_hull_length) / ctrl_hull_length
            
            #     # if diff_curves is > ratio, then overfitting
            #     if diff_of_curves > max_curve_diff:
            #         logger.warning(f"Overfitting detected by curve length {diff_of_curves}")
            #         return False, (
            #             BSplineCurveFit.renorm_fit,
            #             [fit_obj, fit_obj.points_to_fit, min_t, max_t - 1],
            #         )
            else:
                logger.info(f"Valid spline with residuals {segmentwise_residual}\n\n")
                return True, new_fit
