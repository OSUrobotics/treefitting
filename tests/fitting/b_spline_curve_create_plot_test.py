import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import matplotlib.pyplot as plt
from tree_geometry.b_spline_curve import plot_test

if __name__ == "__main__":
    plot_test()