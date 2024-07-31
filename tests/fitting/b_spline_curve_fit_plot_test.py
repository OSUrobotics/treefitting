#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import matplotlib.pyplot as plt
from fit_routines.b_spline_curve_fit import BSplineCurveFit

def plot_test():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bs = BSplineCurveFit(dim=2, ctrl_pts=[], degree="cubic", figax=(fig, ax))
    bs.enable_onclick()
    # bs.plot_basis(plt)
    # bs.plot_curve()
    ax.plot([min(-2, -5), max(10, 5)], [0, 0], "-k")  # x axis
    ax.plot([0, 0], [min(-10, -5), max(10, 5)], "-k")
    plt.show()
    fig.canvas.mpl_disconnect(bs.cid)
    return


if __name__ == "__main__":
    plot_test()
