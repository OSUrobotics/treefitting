import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import matplotlib.pyplot as plt
import numpy as np
from tree_geometry.b_spline_curve import BSplineCurve

def plot_test():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bs = BSplineCurve(
        ctrl_pts=[
            np.array([0, 0]),
            np.array([3, 5]),
            np.array([6, -5]),
            np.array([6.5, -3]),
        ],
        degree="cubic",
        figax=(fig, ax),
    )
    bs.enable_onclick()
    # bs.plot_basis(plt)
    bs.plot_curve()
    plt.show()
    fig.canvas.mpl_disconnect(bs.cid)
    return


if __name__ == "__main__":
    plot_test()