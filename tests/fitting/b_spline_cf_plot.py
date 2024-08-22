import os
import sys
import logging
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import matplotlib.pyplot as plt
import numpy as np
from fit_routines.params import fit_params
from tree_geometry.b_spline_curve import BSplineCurve
from fit_routines.b_spline_curve_fit import BSplineCurveFit

logger = logging.getLogger("b_spline_cf_plot")
logger.setLevel(level=logging.WARN)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
np.set_printoptions(precision=3, suppress=True)


class BSplineCfPlot(BSplineCurveFit):
    def __init__(self, original, fit_params, points_to_fit, figax):
        super().__init__(original, fit_params, points_to_fit)
        self.fig, self.ax = figax
        self.last_good_fit = None

    @staticmethod
    def from_fit_obj(fit_obj, fig, ax):
        return BSplineCfPlot(
            fit_obj.curve,
            fit_obj.params,
            fit_obj.points_to_fit,
            (fit_obj.fig, fit_obj.ax),
        )

    @staticmethod
    def plot_points(fit_obj, fig, ax):
        """plot clicked points with ctrl and data points. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on
        :param ax: mpl axes to use
        :return: (ctrl_point_line, spline_line)
        """
        logger.debug("plot_points")
        if fig == None and ax == None:
            logger.debug("Atleast pass figure and ax!")
        if fit_obj.curve.is_initialized:
            ctrl_array = np.reshape(fit_obj.curve.ctrl_pts, (-1, fit_obj.dim))
            ax.plot(ctrl_array[:, 0], ctrl_array[:, 1], "ro", label="control points")
            fit_array = np.reshape(fit_obj.params.get_param("lsq_points"), (-1, fit_obj.dim))
            ax.plot(fit_array[:, 0], fit_array[:, 1], "yo", label="lsq points")

        clicked_array = np.reshape(fit_obj.points_to_fit, (-1, fit_obj.dim))
        ax.plot(clicked_array[:, 0], clicked_array[:, 1], "bo", label="points to fit")
        # axes
        ax.plot(
            [
                min(-2, min(clicked_array[:, 0]) - 5),
                max(10, max(clicked_array[:, 0]) + 5),
            ],
            [0, 0],
            "-k",
        )  # x axis
        ax.plot(
            [0, 0],
            [
                min(-10, min(clicked_array[:, 1]) - 5),
                max(10, max(clicked_array[:, 1]) + 5),
            ],
            "-k",
        )


        ax.axis("equal")
        ax.grid()
        plt.draw()

    @staticmethod
    def plot_curve(fit_obj, fig=None, ax=None):
        """plot spline curve. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on
        :param ax: mpl axes to use
        """
        logger.debug("plot_curve")
        if fig == None and ax == None:
            logger.debug("Atleast pass figure and ax!")
        tr = np.linspace(0, fit_obj.params.ts[-1], 1000)
        spline = fit_obj.curve.eval_crv(tr)
        # logger.debug(f"{spline}")
        logger.debug(
            f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(fit_obj.curve.ctrl_pts)} points"
        )
        ax.plot(spline[:, 0], spline[:, 1], label="spline")
        ax.axis("equal")
        ax.grid()
        plt.draw()

    def onclick(self, event):
        """manages matplotlib interactive plotting

        :param event: _description_
        """
       
        # print(type(event))
        if event.button == 1:  # projection on convex hull LEFT
            ix, iy = np.round(event.xdata, 3), np.round(event.ydata, 3)
            if ix == None or iy == None:
                logger.warning("You didn't actually select a point!")
                return
            self.points_to_fit.append(np.array((ix, iy)))
            logger.info(f"x {ix} y {iy} added")
            self.ax.clear()
            BSplineCfPlot.plot_points(self, self.fig, self.ax)
            
            if len(self.points_to_fit) > self.curve.degree:
                if not self.curve.is_initialized:
                    new_fit = BSplineCurveFit.one_segment_fit(self, self.points_to_fit)
                else:
                    new_fit = BSplineCurveFit.renorm_fit(
                        self, self.points_to_fit, 0, self.curve.max_t
                    )
                
                good, next = BSplineCurveFit.evaluate(new_fit, self.last_good_fit)
                if good:
                    self.last_good_fit = deepcopy(new_fit)
                    print(f"\n\ngood fit {self.last_good_fit.params.__dict__}\n\n")
                
                i = 0
                while not good and i < 20:
                    func, args = next
                    logger.warn(f"attempting {func.__name__}")
                    new_fit = func(*args)
                    good, new_next = BSplineCurveFit.evaluate(new_fit, self.last_good_fit)
                    next = new_next
                    i += 1
                if not good:
                    logger.error("Failed to fit curve")
                    exit(1)
                else:
                    self.last_good_fit = deepcopy(new_fit)
                    print(f"\n\ngood fit {self.last_good_fit.params.__dict__}\n\n")
                self.ax.clear()
                BSplineCfPlot.plot_curve(self.last_good_fit, self.fig, self.ax)
                BSplineCfPlot.plot_points(self.last_good_fit, self.fig, self.ax)
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.draw()

    def enable_onclick(self):
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
