#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import matplotlib.pyplot as plt
import unittest
import numpy as np
from fit_routines.bspline_fit_params import BSplineFitParams
from tree_geometry.b_spline_curve import BSplineCurve
from fit_routines.b_spline_curve_fit import BSplineCurveFit
from tests.fitting.b_spline_cf_plot import BSplineCfPlot

class TestBSplineCurveFit(unittest.TestCase):
    def setUp(self):
        self.params = BSplineFitParams()
        self.curve = BSplineCurve(
                degree="cubic",
            )

    def test_parameterize_chord_basic(self):
        points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]
        self.params.update_param("p_norm", 1.0)
        new_params, parameterized = BSplineCurveFit._parameterize_chord(
            points, self.params
        )
        expected_parameterized = np.array([0.0, np.sqrt(2), 2 * np.sqrt(2)])
        self.assertTrue(np.allclose(parameterized, expected_parameterized))
        self.assertIsNotNone(new_params.get_param("p_norm"))
        self.assertIsNotNone(new_params.get_param("ts"))

    def test_parameterize_chord_renorm(self):
        points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]
        params, parameterized = BSplineCurveFit._parameterize_chord(
            points, self.params, renorm=True
        )
        expected_parameterized = np.array([0.0, 0.5, 1.0])
        self.assertTrue(np.allclose(parameterized, expected_parameterized))

    def test_parameterize_chord_custom_start_stop(self):
        points = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])]
        new_params, parameterized = BSplineCurveFit._parameterize_chord(
            points, self.params, start=2, stop=4
        )
        expected_parameterized = np.array([2.0, 3.0, 4.0])
        self.assertTrue(np.allclose(parameterized, expected_parameterized))

    def test_basic_lsq(self):
        curve = BSplineCurve(
                degree="cubic",
            )
        points_to_fit = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0]), np.array([3, 0])]
        fit_obj = BSplineCurveFit(original=curve, params=BSplineFitParams(), points_to_fit=points_to_fit)
        ts = np.array([0., 1., 2., 3.])
        fit_obj.params.update_param("ts", ts)
        new_fit_obj, a_constraints = BSplineCurveFit._setup_basic_lsq(fit_obj)
        self.assertIsNotNone(a_constraints)
        self.assertEqual(a_constraints.shape, (len(ts), ts[-1] + new_fit_obj.curve.degree))
    
    def test_one_segment_fit_and_simple_fit(self):
        curve = BSplineCurve(
                degree="cubic",
            )
        points_to_fit = [np.array([3.0, 2.5]), np.array([5.58,  -2.992]), np.array([7.166, -2.347]),  np.array([10.572,  1.052])]
        fit_obj = BSplineCurveFit(original=curve, params=BSplineFitParams(), points_to_fit=points_to_fit)

        new_fit_obj = BSplineCurveFit.one_segment_fit(fit_obj, points_to_fit)
        expected_ctrl_pts = np.array([[ 25.379,  62.572],
                                      [ -6.095, -12.847],
                                      [ 17.   ,   3.816],
                                      [  1.526,   3.893]])
        
        expected_residuals = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(new_fit_obj.curve.ctrl_pts, expected_ctrl_pts, decimal=3)
        np.testing.assert_array_almost_equal(new_fit_obj.params.residuals, expected_residuals, decimal=3)

        new_fit_obj = BSplineCurveFit.simple_fit(fit_obj, points_to_fit)
        np.testing.assert_array_almost_equal(new_fit_obj.curve.ctrl_pts, expected_ctrl_pts, decimal=3)
        np.testing.assert_array_almost_equal(new_fit_obj.params.residuals, expected_residuals, decimal=3)

    def test_renorm_fit(self):
        # getting ts value
        curve = BSplineCurve(
                degree="cubic",
            )
        points_to_fit = [np.array([3.0, 2.5]), np.array([5.58,  -2.992]), np.array([7.166, -2.347]),  np.array([10.572,  1.052])]
        fit_obj = BSplineCurveFit(original=curve, params=BSplineFitParams(), points_to_fit=points_to_fit)
        fit_obj.params.update_param("p_norm", 1.0) # break normalisation
        
        original_ts = [0., 0.482, 0.618, 1.]
        expected_ctrl_pts = np.array([[ 25.379,  62.572],
                                      [ -6.095, -12.847],
                                      [ 17.   ,   3.816],
                                      [  1.526,   3.893]])
        
        new_fit_obj = BSplineCurveFit.simple_fit(fit_obj, points_to_fit)
        self.assertLess(len(expected_ctrl_pts), len(new_fit_obj.curve.ctrl_pts))
        self.assertLess(original_ts[-1], new_fit_obj.params.ts[-1])
        

        new_fit_obj = BSplineCurveFit.renorm_fit(new_fit_obj, points_to_fit)
        np.testing.assert_array_almost_equal(new_fit_obj.curve.ctrl_pts, expected_ctrl_pts, decimal=3)
    
    def test_extend_fit(self):
        curve = BSplineCurve(
                degree="cubic",
            )
        points_to_fit = [np.array([3.0, 2.5]), np.array([5.58,  -2.992]), np.array([7.166, -2.347]),  np.array([10.572,  1.052])]
        fit_obj = BSplineCurveFit(original=curve, params=BSplineFitParams(), points_to_fit=points_to_fit)
        new_fit_obj = BSplineCurveFit.one_segment_fit(fit_obj, points_to_fit)
        
        original_ts = [0., 0.482, 0.618, 1.]
        expected_ctrl_pts = np.array([[ 25.379,  62.572],
                                      [ -6.095, -12.847],
                                      [ 17.   ,   3.816],
                                      [  1.526,   3.893]])
        
        new_fit_obj = BSplineCurveFit.extend_curve(new_fit_obj, [np.array([12.0, 2.5])])
        self.assertLess(len(expected_ctrl_pts), len(new_fit_obj.curve.ctrl_pts))
        self.assertLess(original_ts[-1], new_fit_obj.params.ts[-1])
        

        new_fit_obj = BSplineCurveFit.renorm_fit(new_fit_obj, points_to_fit)
        np.testing.assert_array_almost_equal(new_fit_obj.curve.ctrl_pts, expected_ctrl_pts, decimal=3)

def plot_test():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    params = BSplineFitParams()
    params.add_param("degree", 3)
    params.add_param("dim", 2)
    bs = BSplineCfPlot(BSplineCurve(degree="cubic"), params, [], (fig, ax))
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
    unittest.main()
