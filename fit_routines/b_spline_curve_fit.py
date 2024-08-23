from __future__ import annotations
import logging
from typing import Tuple, Union, List, Callable
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

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
    def _lsq_fit(fit_obj, points):
        """Fit the curve to the points with current params. Does not preserve original fit!"""
        # get basic least squares constraints
        fit_obj, a_constraints = BSplineCurveFit._setup_basic_lsq(fit_obj)

        # least squares fit
        ctrl_pts, residuals, _, __ = np.linalg.lstsq(
            a=a_constraints, b=points, rcond=None
        )
        residuals = np.linalg.norm(points - np.dot(a_constraints, ctrl_pts), axis=1)


        # update the fit object
        fit_obj.curve.ctrl_pts = deepcopy(ctrl_pts)
        fit_obj.points_to_fit = deepcopy(points)
        fit_obj.params.update_param("residuals", residuals)
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
        new_fit.params = new_params
        return BSplineCurveFit._lsq_fit(new_fit, points)

    @staticmethod
    def one_segment_fit(fit_obj, points: list[np.ndarray]) -> BSplineCurveFit:
        """Fit a simple one_segment curve, ie t = [0, 1) to the points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        logger.info("attempt one_segment fit")
        return BSplineCurveFit.renorm_fit(fit_obj, points, 0, 1)

    @staticmethod
    def iteratively_fit(
        fit_obj,
        points: list[np.ndarray],
        outlier_ratio: float = 0.1,
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
            new_fit = BSplineCurveFit.simple_fit(sampled_points)
            new_fit.points_to_fit = points

            # check all 
            all_points_in_t = BSplineCurveFit._parameterize_chord(points, new_fit.params)
            new_fit, a_constraints = BSplineCurveFit._setup_basic_lsq(all_points_in_t)
            residuals = np.linalg.norm(points - np.dot(a_constraints, new_fit.curve.ctrl_pts), axis=1)
            results.append((new_fit.curve.ctrl_pts, all_points_in_t, residuals))
        # get best curve from results
        min_result = min(results, key=lambda x: np.sum(x[2]))
        return min_result

    def extend_curve(fit_obj, new_data_pts: list[np.ndarray]):
        """Extend the curve to fit the new data points
        :param new_data_pts: new data points
        :type new_data_pts: list[np.ndarray]
        """
        logger.debug(f"extending curve of {len(fit_obj.ts)} by {len(new_data_pts)}")
        new_points_in_t = np.zeros(len(new_data_pts), dtype=float)
        points_in_t = np.zeros((len(fit_obj.ts) + len(new_data_pts)), dtype=float)
        old_points = np.reshape(fit_obj.data_pts, (-1, fit_obj.dim))
        b_constraints = np.zeros((len(points_in_t), fit_obj.dim), dtype=float)
        if len(new_data_pts) == 1:
            new_points_in_t = (
                np.linalg.norm(new_data_pts[0] - old_points[-1])
                * fit_obj.parameter_normalization
            ) + fit_obj.ts[-1]
            print(f"last {fit_obj.ts[-1], new_data_pts[0], fit_obj.data_pts}")
        else:
            points = new_data_pts
            new_points_in_t = fit_obj._parameterize_chord(points) + fit_obj.ts[-1]
        points_in_t = np.hstack((fit_obj.ts, new_points_in_t))
        b_constraints[: len(fit_obj.ts)] = fit_obj.eval_crv(fit_obj.ts)
        b_constraints[len(fit_obj.ts) :] = new_data_pts
        a_constraints = fit_obj.setup_basic_lsq(points_in_t)
        logger.debug(
            f" A = \n{a_constraints}\n B = \n{b_constraints} \n calculated using t: \n {points_in_t} \n"
        )
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(
            a=fit_obj.setup_basic_lsq(points_in_t), b=b_constraints, rcond=None
        )
        residuals = np.linalg.norm(
            b_constraints - np.dot(a_constraints, ctrl_pts), axis=1
        )
        diff_of_curves = (
            1 - abs(fit_obj.ctrl_hull_length - fit_obj.curve_length) / fit_obj.ctrl_hull_length
        )
        # if diff_curves is > 0.1 and residuals are low: overfitting
        logger.debug(
            f"Extended residuals {residuals}, rank {rank}, curve length diff {diff_of_curves}"
        )
        return ctrl_pts, points_in_t, residuals

    def plot_points(fit_obj, fig=None, ax=None):
        """plot clicked points with ctrl and data points. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        :return: (ctrl_point_line, spline_line)
        """
        logger.debug("plot_points")
        if fig == None and ax == None and fit_obj.fig == None and fit_obj.ax == None:
            logger.debug("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            fit_obj.fig = fig
            fit_obj.ax = ax

        ctrl_array = np.reshape(fit_obj.curve.ctrl_pts, (-1, fit_obj.curve.dim))
        clicked_array = np.reshape(fit_obj.points_to_fit, (-1, fit_obj.curve.dim))

        fit_obj.ax.plot(
            clicked_array[:, 0], clicked_array[:, 1], "bo", label="points to fit"
        )
        fit_obj.ax.plot(ctrl_array[:, 0], ctrl_array[:, 1], "ro", label="control points")

        # axes
        fit_obj.ax.plot(
            [
                min(-2, min(clicked_array[:, 0]) - 5),
                max(10, max(clicked_array[:, 0]) + 5),
            ],
            [0, 0],
            "-k",
        )  # x axis
        fit_obj.ax.plot(
            [0, 0],
            [
                min(-10, min(clicked_array[:, 1]) - 5),
                max(10, max(clicked_array[:, 1]) + 5),
            ],
            "-k",
        )
        fit_obj.ax.axis("equal")
        fit_obj.ax.grid()
        plt.draw()

    @staticmethod
    def plot_curve(fit_obj, fig = None, ax = None):
        """plot spline curve. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        """
        logger.debug("plot_curve")
        if fig == None and ax == None and fit_obj.fig == None and fit_obj.ax == None:
            logger.error("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            fit_obj.fig = fig
            fit_obj.ax = ax
        tr = np.linspace(0, fit_obj.params.ts[-1], 1000)
        spline = fit_obj.curve.eval_crv(tr)
        # logger.debug(f"{spline}")
        logger.debug(
            f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(fit_obj.curve.ctrl_pts)} points"
        )
        fit_obj.ax.plot(spline[:, 0], spline[:, 1], label="spline")
        fit_obj.ax.axis("equal")
        fit_obj.ax.grid()
        plt.draw()

    def evaluate(
        fit_obj, result, residual_min=1e-6, residual_max=1e-2, max_control_ratio=10
    ) -> tuple[bool, Union[BSplineCurve, Tuple[Callable, List]]]:
        """Evaluate the result of the fit and recommend next steps if not satisfactory

        :param result: tuple of control points, points in t and residuals
        :param residual_threshold: residual threshold per segment, defaults to 0.1
        :param max_control_points: _description_, defaults to 30
        :return: whether to
        """
        ctrl_pts, points_in_t, residuals = result
        min_t = int(np.floor(min(points_in_t)))
        max_t = int(np.ceil(max(points_in_t)))
        logger.debug(f"minmax: {min_t, max_t}")

        # overfitting
        if len(fit_obj.points_to_fit) / max_control_ratio > len(ctrl_pts):
            logger.warning(
                f"Too many control points {len(ctrl_pts)} for data points {len(fit_obj.data_pts)}"
            )
            return False, (fit_obj.renorm_fit, [fit_obj.data_pts, min_t, max_t - 1])

        # bucket residuals
        t_buckets = np.array(range(min_t, max_t + 1))

        segmentwise_residual = []
        for i in range(len(t_buckets) - 1):
            cond = np.asarray(
                (points_in_t >= t_buckets[i]) & (points_in_t < t_buckets[i + 1])
            )
            segmentwise_residual.append(np.sum(residuals[cond.nonzero()]))
        logger.debug(f"{segmentwise_residual, t_buckets}")
        segmentwise_residual = np.array(segmentwise_residual)
        max_residual = segmentwise_residual < residual_max
        min_residual = segmentwise_residual > residual_min
        check_residual = max_residual & min_residual
        if check_residual.all() or (max_t == 1 and max_residual.all()):
            new_spline = deepcopy(fit_obj)
            new_spline.ctrl_pts = [ctrl_pts[i] for i in range(ctrl_pts.shape[0])]
            new_spline.ts = points_in_t
            new_spline.residuals = residuals
            logger.debug(f"Valid spline with residuals {segmentwise_residual}")
            return True, new_spline
        elif not max_residual.all():
            indices = np.asarray(max_residual == False).nonzero()
            logger.info(f"Residuals too high in segments {t_buckets[indices]}")
            # if residuals high towards the end, extend curve
            if np.mean(t_buckets[indices]) > 0.8 * (max_t - 1):
                return False, (fit_obj.extend_curve, t_buckets[indices])
            else:  # add another control point
                return False, (fit_obj.renorm_fit, [fit_obj.data_pts, min_t, max_t + 1])
        # elif not min_residual.all(): # overfitting
        #     return False, (fit_obj.renorm_fit, [fit_obj.data_pts, min_t, max_t - 1])

    def onclick(fit_obj, event):
        """manages matplotlib interactive plotting

        :param event: _description_
        """

        # print(type(event))
        if event.button == 1:  # projection on convex hull LEFT
            ix, iy = np.round(event.xdata, 3), np.round(event.ydata, 3)
            if ix == None or iy == None:
                logger.warning("You didn't actually select a point!")
                return
            fit_obj.points_to_fit.append(np.array((ix, iy)))
            logger.info(f"x {ix} y {iy} added")
            fit_obj.ax.clear()
            fit_obj.plot_points()
            if len(fit_obj.points_to_fit) > fit_obj.degree:
                if not fit_obj.is_initialized:
                    new_control_points, points_in_t, residuals = fit_obj.one_segment_fit(
                        fit_obj.points_to_fit
                    )
                else:
                    new_control_points, points_in_t, residuals = fit_obj.renorm_fit(
                        fit_obj.points_to_fit, 0, fit_obj.max_t
                    )
                good, func = fit_obj.evaluate((new_control_points, points_in_t, residuals))
                if good:
                    fit_obj.__dict__.update(func.__dict__)
                    logger.debug(f"Control points {fit_obj.ctrl_pts}")
                    fit_obj.data_pts = fit_obj.points_to_fit

                else:
                    logger.info("Residuals too high, extending curve")
                    new_control_points, points_in_t, residuals = fit_obj.extend_curve(
                        [fit_obj.points_to_fit[-1]]
                    )

                    if residuals.size == 0 or (residuals < 2.0).all():
                        fit_obj.ctrl_pts = [
                            new_control_points[i]
                            for i in range(new_control_points.shape[0])
                        ]
                        fit_obj.ts = points_in_t
                        logger.debug(f"Control points {new_control_points}")
                        fit_obj.add_data_point(fit_obj.points_to_fit[-1])
                        fit_obj.residuals = residuals
                        fit_obj.residuals = np.zeros(len(fit_obj.data_pts))
                        logger.debug(
                            f"ts: {fit_obj.ts} now eval to \n{fit_obj.eval_crv(fit_obj.ts)}\n"
                            f"for original \n{fit_obj.unflatten_dim(fit_obj.points_to_fit, fit_obj.dim)}"
                        )
                    else:
                        logger.warning("Residuals too high, not adding points")
                    fit_obj.ax.clear()
                fit_obj.plot_points()
                fit_obj.plot_curve()

            fit_obj.plot_points()
        logger.debug("plotted")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.draw()

    def enable_onclick(fit_obj):
        fit_obj.cid = fit_obj.fig.canvas.mpl_connect("button_press_event", fit_obj.onclick)
