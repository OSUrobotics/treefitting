#!/usr/bin/env python3
import json

import numpy as np

from tree_geometry.b_spline_curve import BSplineCurve
from tree_geometry.point_lists import PointListWithTs
from fit_routines.bspline_fit_params import BSplineFitParams

class BSplineFitEval(dict):
    def __init__(self, fit_params: BSplineFitParams):
        """ Keep the results of the fit and/or calculate fit values
        @param fit_params - inlier, outlier etc values"""
        super().__init__()

        self.params = fit_params

        self.perc_inliers = 0.0
        self.avg_fit = -1.0
        self.avg_fit_all = -1.0
        self.per_pt_distances = None
        self.per_pt_ts = None
        self.per_pt_outlier = None

    def calc_values(self, crv : BSplineCurve, pts_with_ts : PointListWithTs):
        """ Calculate the fit statistics; assumes inlier threshold set
        @param crv - the fitted curve
        @param pts_with_ts - the points list, with t values used for fit
        """
        self.per_pt_distances = np.zeros(pts_with_ts.n_points())
        self.per_pt_ts = np.zeros(pts_with_ts.n_points())
        self.per_pt_outlier = np.zeros(pts_with_ts.n_points(), dtype=bool)

        self.perc_inliers = 0.0
        self.avg_fit = 0.0
        for i, p in enumerate(pts_with_ts.points()):
            t_ret, pt_ret, dist_ret = crv.project_to_curve(p, pts_with_ts.ts[i])
            self.per_pt_distances[i] = dist_ret
            self.per_pt_ts[i] = t_ret

            if dist_ret > self.params["inlier threshold"]:
                self.per_pt_outlier[i] = True
            else:
                self.per_pt_outlier[i] = False
                self.avg_fit += dist_ret
                self.perc_inliers += 1.0

        self.perc_inliers /= pts_with_ts.n_points()
        self.avg_fit_all = np.mean(self.per_pt_distances)
        if self.perc_inliers < 1.0:
            self.avg_fit /= np.sum(self.per_pt_outlier)
        else:
            self.avg_fit = self.avg_fit_all * 100.0

    def calc_inlier_value(self):
        """ Determine a good inlier value based on the current fit"""
        pass    # Need tow rite
        return 0.1

    def is_acceptable(self) -> bool:
        """ Is the current fit good enough - average below a threshold, percentage of outliers below a threshold
        @return True/False"""
        if self.avg_fit_all > self.params["average fit"]:
            return False
        if 1.0 - self.perc_inliers > self.params["outlier ratio"]:
            return False
        return True

    def is_better(self, other_fit : object):
        """ Check both number of inliers and average fit
        @param other_fit - the other fit eval params (of type BSplineFitEval)"""

        if self.perc_inliers >= other_fit.perc_inliers and self.avg_fit <= other_fit.avg_fit:
            return True

        if self.perc_inliers <= other_fit.perc_inliers and self.avg_fit >= other_fit.avg_fit:
            return False

        if abs(self.perc_inliers - other_fit.perc_inliers) < 0.1:
            if self.avg_fit >= other_fit.avg_fit:
                return True
            return False

        if self.perc_inliers <= other_fit.perc_inliers:
            return True
        return False

    # @staticmethod
    # def evaluate(
    #         new_fit, last_good_fit=None, residual_min=1e-6, residual_max=1e-2, max_control_ratio=10, max_curve_diff=0.2
    # ) -> tuple[bool, Union[BSplineCurve, Tuple[Callable, List]]]:
        """Evaluate the result of the fit and recommend next steps if not satisfactory

        :param fit_obj: tuple of control points, points in t and residuals
        :param last_good_fit: last good fit for reuse in case of extension
        :param residual_min: residual threshold per segment, defaults to 1e-6
        :param residual_max: residual threshold per segment, defaults to 0.01
        :param max_control_ratio: no of points per control pt, defaults to 10
        :param max_curve_diff: ratio of max difference between curve and hull length, defaults to 0.2
        :return: (validity, (next function, [args]))
        ctrl_pts, points_in_t, residuals = (
            new_fit.curve.ctrl_pts,
            new_fit.params.get_param("ts"),
            new_fit.params.get_param("residuals"),
        )
        min_t = int(np.floor(min(points_in_t)))
        max_t = int(np.ceil(max(points_in_t)))
        # logger.debug(f"eval minmax: {min_t, max_t}")

        # bucket residuals
        t_buckets = np.array(range(min_t, max_t + 1))
        segmentwise_residual = []
        for i in range(len(t_buckets) - 1):
            cond = np.asarray(
                (points_in_t >= t_buckets[i]) & (points_in_t < t_buckets[i + 1])
            )
            segmentwise_residual.append(np.sum(residuals[cond.nonzero()]))
        # logger.info(f"residuals: {residuals}, seg: {segmentwise_residual} buckets: {t_buckets}")
        segmentwise_residual = np.array(segmentwise_residual)
        new_fit.params.add_param("seg_residuals", segmentwise_residual)

        # check segmentwise residuals
        max_residual = segmentwise_residual < residual_max
        min_residual = segmentwise_residual > residual_min
        check_residual = max_residual & min_residual
        if check_residual.all() or (max_t == 1 and max_residual.all()):
            # logger.info(f"Valid spline with residuals {segmentwise_residual}\n\n")
            return True, new_fit
        elif not max_residual.all():
            indices = np.asarray(max_residual == False).nonzero()
            # logger.warning(f"Residuals too high in segments {t_buckets[indices]}")
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
            # logger.info(f'Adding new control point to play with')
            return False, (
                BSplineCurveFit.renorm_fit,
                [new_fit, new_fit.points_to_fit, min_t, max_t + 1]
            )

        elif not min_residual.all():  # check overfitting
            if len(new_fit.points_to_fit) / max_control_ratio > len(ctrl_pts):
                # logger.warning(
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
            # logger.info(f"Valid spline with residuals {segmentwise_residual}\n\n")
            return True, new_fit
        """


if __name__ == '__main__':
    params = BSplineFitParams()
    check_eval = BSplineFitEval(params)
