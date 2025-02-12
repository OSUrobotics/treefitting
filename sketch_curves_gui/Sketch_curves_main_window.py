#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./Image_based'))
sys.path.insert(0, os.path.abspath('./Utilities'))
sys.path.insert(0, os.path.abspath('./sketch_curves_gui'))
sys.path.insert(0, os.path.abspath('./fit_routines'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../Image_based'))
sys.path.insert(0, os.path.abspath('../Utilities'))
sys.path.insert(0, os.path.abspath('../sketch_curves_gui'))
sys.path.insert(0, os.path.abspath('../fit_routines'))
from os.path import exists

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QSpacerItem

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QTextEdit, QSizePolicy
import cv2
import numpy as np

from MySliders import SliderIntDisplay, SliderFloatDisplay
from Draw_spline_3D import DrawSpline3D
from FileNames import FileNames

from extract_curves import ExtractCurves
from fit_bezier_cyl_2d_sketch import FitBezierCyl2DSketch
from fit_bezier_cyl_3d_depth import FitBezierCyl3dDepth

from sketch_curves_gui.Sketches_for_curves import SketchesForCurves

from b_spline_curve_fit import BSplineCurveFit

class SketchCurvesMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Fitted Curves Viewer')

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()
        right_side_layout = self._init_right_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)
        self.lower_left = [0, 0]
        self.upper_right = [1, 1]

        # Two side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderFloatDisplay.gui = self
        SliderIntDisplay.gui = self

        self.last_index = ()
        self.handle_filenames = None
        self.crv = None
        self.extract_crv = None
        self.fit_crv_3d = None
        self.in_reset_file_menus = False
        self.in_read_images = False
        self.sketch_curve = SketchesForCurves()
        self.crv_from_sketch = None
        if exists("save_crv.json"):
            self.sketch_curve = SketchesForCurves.read_json("save_crv.json")

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        # self.path_name = QLineEdit("/Users/cindygrimm/PycharmProjects/treefitting/Image_based/data/EnvyTree/")
        # self.file_name = QLineEdit("envy_fnames.json")
        self.path_name = QLineEdit("/Users/cindygrimm/VSCode/BlueBerryData/bush_9_west_2/")
        self.file_name = QLineEdit("bush_9_west_2_fnames.json")
        self.sub_dir_number = SliderIntDisplay("Sub dir", 0, 10, 0)
        self.image_number = SliderIntDisplay("Image", 0, 10, 0)
        self.mask_number = SliderIntDisplay("Mask", 0, 3, 0)
        self.mask_id_number = SliderIntDisplay("Mask id", 0, 3, 0)
        self.image_name = QLabel("image name")
        path_names_layout.addWidget(QLabel("Path dir:"))
        path_names_layout.addWidget(self.path_name)
        path_names_layout.addWidget(QLabel("File data names:"))
        path_names_layout.addWidget(self.file_name)
        path_names_layout.addWidget(QLabel("Subdir:"))
        path_names_layout.addWidget(self.sub_dir_number)
        path_names_layout.addWidget(QLabel("Image:"))
        path_names_layout.addWidget(self.image_number)
        path_names_layout.addWidget(QLabel("Mask:"))
        path_names_layout.addWidget(self.mask_number)
        path_names_layout.addWidget(QLabel("Mask id:"))
        path_names_layout.addWidget(self.mask_id_number)
        path_names_layout.addWidget(self.image_name)
        path_names_layout.setSpacing(5)
        path_names.setLayout(path_names_layout)

        self.sub_dir_number.slider.valueChanged.connect(self.read_images)
        self.image_number.slider.valueChanged.connect(self.read_images)
        self.mask_number.slider.valueChanged.connect(self.read_images)
        self.mask_id_number.slider.valueChanged.connect(self.read_images)

        read_filenames_button = QPushButton('Read file names')
        read_filenames_button.clicked.connect(self.read_file_names)

        file_io = QGroupBox('File io')
        file_io_layout = QVBoxLayout()
        file_io_layout.addWidget(path_names)
        file_io_layout.addWidget(read_filenames_button)
        file_io.setLayout(file_io_layout)

        # Sliders for Camera
        reset_view = QPushButton('Reset view')
        reset_view.clicked.connect(self.reset_view)
        self.turntable = SliderFloatDisplay('Rotate turntable', 0.0, 360, 0, 361)
        self.up_down = SliderFloatDisplay('Up down', 0, 360, 0, 361)
        self.zoom = SliderFloatDisplay('Zoom', 0.6, 2.0, 1.0)
        self.horizontal_angle = SliderFloatDisplay('Angle', 45, 175, 90)

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)
        params_camera_layout.addWidget(self.zoom)
        params_camera_layout.addWidget(reset_view)
        params_camera_layout.addWidget(self.horizontal_angle)
        params_camera.setLayout(params_camera_layout)

        params_crvs = QGroupBox('3D Curve parameters')
        params_crvs_layout = QVBoxLayout()
        self.show_3d_crv_button = QCheckBox('Show 3d crv')
        self.show_3d_crv_button.setCheckState(2)
        self.show_3d_crv_button.clicked.connect(self.redraw_self)
        self.show_3d_crv_axis_button = QCheckBox('Show 3d crv axis')
        self.show_3d_crv_axis_button.clicked.connect(self.redraw_self)
        self.show_3d_crv_axis_button.setCheckState(2)
        self.n_around = SliderIntDisplay("N around", 8, 64, 32)
        self.n_along = SliderIntDisplay("N along", 8, 64, 16)
        params_crvs_layout.addWidget(self.show_3d_crv_button)
        params_crvs_layout.addWidget(self.show_3d_crv_axis_button)
        params_crvs_layout.addWidget(self.n_around)
        params_crvs_layout.addWidget(self.n_along)
        params_crvs.setLayout(params_crvs_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        left_side_layout.addWidget(file_io)
        left_side_layout.addStretch()
        left_side_layout.addWidget(params_camera)
        left_side_layout.addWidget(params_crvs)

        return left_side_layout

    # Drawing screen and quit button
    def _init_middle_layout_(self):
        # The display for the robot drawing
        self.glWidget = DrawSpline3D(self)

        self.up_down.slider.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        self.blank_text = QTextEdit('Space')
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)
        quit_button.setMinimumWidth(640)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget(self.glWidget)
        mid_layout.addWidget(self.blank_text)
        mid_layout.addWidget(quit_button, stretch=20)

        return mid_layout

    # Set up the right set of sliders/buttons (recalc)
    def _init_right_layout_(self):
        # Iterate fits
        restart_fit_button = QPushButton('Restart fit')
        restart_fit_button.clicked.connect(self.refit)

        # Fit mas has:
        #   step_size - number of pixels along the mask
        #   width_mask - percent bigger than mask to search (1.0 to 1.5ish)
        #   width_edge - percent of edge +- to search (0.1 to 0.3)
        #   width_profile - same, but for profile curves
        #   edge_max - pixel value to call a valid edge (0..255)
        #   n_per_seg - number of pixels along the profile to resample, should be less than step_size
        #   width_inside - percentage considered "inside" mask
        self.step_size = SliderIntDisplay('Step size', 10, 100, 40)
        self.width_mask = SliderFloatDisplay('Width mask', 1.0, 2.0, 1.4)
        self.width_edge = SliderFloatDisplay('Width edge', 0.1, 0.6, 0.3)
        self.width_profile = SliderFloatDisplay('Width profile', 0.1, 0.6, 0.3)
        self.edge_max = SliderIntDisplay('Edge max', 0, 255, 128)
        self.n_per_seg = SliderIntDisplay('N per seg', 3, 40, 10)
        self.width_inside = SliderFloatDisplay('Width inside', 0.1, 1.0, 0.6)

        resets = QGroupBox('Resets')
        resets_layout = QVBoxLayout()
        resets_layout.addWidget(restart_fit_button)
        resets_layout.setSpacing(5)
        resets.setLayout(resets_layout)

        curve_drawing = QGroupBox('Curve drawing')
        curve_drawing_layout = QVBoxLayout()

        self.show_backbone_button = QCheckBox('Show backbone')
        self.show_backbone_button.setCheckState(2)
        self.show_backbone_button.clicked.connect(self.redraw_self)

        self.show_interior_rects_button = QCheckBox('Show interior rects')
        self.show_interior_rects_button.clicked.connect(self.redraw_self)

        self.show_edge_rects_button = QCheckBox('Show edge rects')
        self.show_edge_rects_button.clicked.connect(self.redraw_self)

        self.show_profiles_button = QCheckBox('Show profile curves')
        self.show_profiles_button.clicked.connect(self.redraw_self)

        curve_drawing_layout.addWidget(self.show_backbone_button)
        curve_drawing_layout.addWidget(self.show_interior_rects_button)
        curve_drawing_layout.addWidget(self.show_edge_rects_button)
        curve_drawing_layout.addWidget(self.show_profiles_button)
        curve_drawing_layout.addWidget(self.step_size)
        curve_drawing_layout.addWidget(self.width_mask)
        curve_drawing_layout.addWidget(self.width_edge)
        curve_drawing_layout.addWidget(self.width_profile)
        curve_drawing_layout.addWidget(self.edge_max)
        curve_drawing_layout.addWidget(self.n_per_seg)
        curve_drawing_layout.addWidget(self.width_inside)
        curve_drawing_layout.setSpacing(5)
        curve_drawing.setLayout(curve_drawing_layout)

        # For showing images and curves
        shows = QGroupBox('Shows')
        shows_layout = QHBoxLayout()

        self.show_rgb_button = QCheckBox('Show rgb')
        self.show_rgb_button.setCheckState(2)
        self.show_rgb_button.clicked.connect(self.redraw_self)
        self.show_mask_button = QCheckBox('Show mask')
        self.show_mask_button.clicked.connect(self.redraw_self)
        self.show_edge_button = QCheckBox('Show edge')
        self.show_edge_button.clicked.connect(self.redraw_self)
        self.show_opt_flow_button = QCheckBox('Show optical flow')
        self.show_opt_flow_button.clicked.connect(self.redraw_self)
        self.show_depth_button = QCheckBox('Show depth')
        self.show_depth_button.clicked.connect(self.redraw_self)

        self.show_sketch_crv_button = QCheckBox('Show sketch')
        self.show_sketch_crv_button.clicked.connect(self.redraw_self)
        self.show_mask_crv_button = QCheckBox('Show mask')
        self.show_mask_crv_button.clicked.connect(self.redraw_self)
        self.show_edge_crv_button = QCheckBox('Show edge')
        self.show_edge_crv_button.clicked.connect(self.redraw_self)
        self.show_edge_crv_button.setCheckState(2)

        show_images = QGroupBox('Image shows')
        show_images_layout = QVBoxLayout()
        show_images_layout.addWidget(self.show_rgb_button)
        show_images_layout.addWidget(self.show_mask_button)
        show_images_layout.addWidget(self.show_edge_button)
        show_images_layout.addWidget(self.show_opt_flow_button)
        show_images_layout.addWidget(self.show_depth_button)
        show_images_layout.setSpacing(5)
        show_images.setLayout(show_images_layout)

        show_curves = QGroupBox('Curve shows')
        show_curves_layout = QVBoxLayout()
        show_curves_layout.addWidget(self.show_sketch_crv_button)
        show_curves_layout.addWidget(self.show_mask_crv_button)
        show_curves_layout.addWidget(self.show_edge_crv_button)
        show_curves.setLayout(show_curves_layout)

        shows_layout.addWidget(show_images)
        shows_layout.addWidget(show_curves)
        shows.setLayout(shows_layout)
        # Drawing
        drawing_states = QGroupBox('Drawing states         ')
        drawing_states_layout = QVBoxLayout()
        self.draw_backbone_button = QPushButton('New curve')
        self.draw_backbone_button.clicked.connect(self.new_curve)
        clear_drawings_button = QPushButton('Clear drawings')
        clear_drawings_button.clicked.connect(self.clear_drawings)
        self.mask_name = QLabel("None")

        drawing_states_layout.addWidget(self.draw_backbone_button)
        drawing_states_layout.addWidget(clear_drawings_button)
        drawing_states_layout.addWidget(self.mask_name)

        drawing_states.setLayout(drawing_states_layout)

        # Put all the pieces in one box
        right_side_layout = QVBoxLayout()

        right_side_layout.addWidget(resets)
        right_side_layout.addWidget(curve_drawing)
        right_side_layout.addWidget(shows)
        right_side_layout.addStretch()
        right_side_layout.addWidget(drawing_states)

        return right_side_layout

    def reset_file_menus(self):
        if self.in_reset_file_menus:
            return
        self.in_reset_file_menus = True
        indx_sub_dir = self.sub_dir_number.value()
        indx_image = self.image_number.value()
        indx_mask = self.mask_number.value()
        id_mask = self.mask_id_number.value()
        print(f"Begin reset file name {indx_sub_dir} {indx_image} {indx_mask} {id_mask}")
        b_changed = False
        sldr_maxs_orig = (self.sub_dir_number.slider.maximum(),
                     self.image_number.slider.maximum(),
                     self.mask_number.slider.maximum(),
                     self.mask_id_number.slider.maximum())
        print(f"Sliders orig {sldr_maxs_orig}")
        if self.sub_dir_number.slider.maximum() > len(self.handle_filenames.sub_dirs):
            self.sub_dir_number.slider.setMaximum(len(self.handle_filenames.sub_dirs))
            b_changed = True
        if indx_sub_dir >= self.sub_dir_number.slider.maximum():
            indx_sub_dir = 0

        if self.image_number.slider.maximum() != len(self.handle_filenames.image_names[indx_sub_dir]):
            self.image_number.slider.setMaximum(len(self.handle_filenames.image_names[indx_sub_dir]))
            b_changed = True
            print(f" Changing image number {self.image_number.slider.maximum()} {indx_image}")
        if indx_image >= self.image_number.slider.maximum():
            indx_image = 0
            self.image_number.set_value(indx_image)

        if self.mask_number.slider.maximum() != len(self.handle_filenames.mask_names):
            self.mask_number.slider.setMaximum(len(self.handle_filenames.mask_names))
            b_changed = True
            print(f" Changing mask number {self.mask_number.slider.maximum()} {indx_mask}")
        if indx_mask >= self.mask_number.slider.maximum():
            indx_mask = 0
            self.mask_number.set_value(indx_mask)

        if self.mask_id_number.slider.maximum() != len(self.handle_filenames.mask_ids[indx_sub_dir][indx_image][indx_mask]):
            self.mask_id_number.slider.setMaximum(len(self.handle_filenames.mask_ids[indx_sub_dir][indx_image][indx_mask]))
            b_changed = True
            print(f" Changing mask id number {self.mask_id_number.slider.maximum()} {id_mask}")
        if id_mask >= self.mask_id_number.slider.maximum():
            id_mask = 0
            self.mask_id_number.set_value(id_mask)

        indx = (indx_sub_dir, indx_image, indx_mask, id_mask)
        if indx != self.last_index:
            b_changed = True
        print(f" New index {indx}")

        img_name = self.handle_filenames.get_image_name(index=indx)
        img_name_split = img_name.split("/")
        if indx_mask >= 0 and indx_mask < len(self.handle_filenames.mask_names):
            mask_name = self.handle_filenames.mask_names[indx_mask]
        else:
            mask_name = "none"
        self.mask_name.setText(mask_name)
        if len(img_name_split) > 2:
            self.image_name.setAccessibleName(img_name_split[-2])
            self.image_name.setText(img_name_split[-1] + " mask: " + mask_name)
        else:
            self.image_name.setText(img_name + " mask: " + mask_name)
        sldr_maxs = (self.sub_dir_number.slider.maximum(),
                     self.image_number.slider.maximum(),
                     self.mask_number.slider.maximum(),
                     self.mask_id_number.slider.maximum())
        print(f"index {indx} sldrs {sldr_maxs} redo {b_changed}")
        self.in_reset_file_menus = False

        return b_changed, indx

    def reset_params_menus(self):
        """ Set all the sliders based on the stored curve"""
        if not self.extract_crv.params:
            return
        
        self.step_size.set_value(self.extract_crv.params["step_size"])
        self.width_mask.set_value(self.extract_crv.params["width_mask"])
        self.width_edge.set_value(self.extract_crv.params["width_edge"])
        self.width_profile.set_value(self.extract_crv.params["width_profile"])
        self.edge_max.set_value(self.extract_crv.params["edge_max"])
        self.n_per_seg.set_value(self.extract_crv.params["n_per_seg"])

    def get_file_name_tuple(self):
        return (self.sub_dir_number.value(), self.image_number.value(), self.mask_number.value(), self.mask_id_number.value())

    def read_file_names(self):
        fname = self.path_name.text() + self.file_name.text()
        self.handle_filenames = FileNames.read_filenames(fname, path=self.path_name.text())
        self.reset_file_menus()
        self.read_images()
        self.reset_params_menus()

        if self.crv:
            width_rgb_image = self.crv.image_rgb.shape[1]
            height_rgb_image = self.crv.image_rgb.shape[0]
            aspect_ratio = height_rgb_image / width_rgb_image

            w = self.glWidget.width()
            h = int(aspect_ratio * w)


            self.glWidget.resize(w, h)

    def reset_view(self):
        self.turntable.set_value(0.0)
        self.up_down.set_value(0.0)
        self.zoom.set_value(1.0)
        self.redraw_self()

    def sizePolicy(self) -> 'QSizePolicy':
        return QSizePolicy.Fixed

    def refit(self):
        params = {}
        params["step_size"] = self.step_size.value()
        params["width_mask"] = self.width_mask.value()
        params["width_edge"] = self.width_edge.value()
        params["width_profile"] = self.width_profile.value()
        params["edge_max"] = self.edge_max.value()
        params["n_per_seg"] = self.n_per_seg.value()

        self.step_size.set_value(self.extract_crv.params["step_size"])
        self.width_mask.set_value(self.extract_crv.params["width_mask"])
        self.width_edge.set_value(self.extract_crv.params["width_edge"])
        self.width_profile.set_value(self.extract_crv.params["width_profile"])
        self.edge_max.set_value(self.extract_crv.params["edge_max"])
        self.n_per_seg.set_value(self.extract_crv.params["n_per_seg"])
        self.set_crv(params)

    def refit_edges(self):
        pass

    def clear_drawings(self):
        self.sketch_curve.clear()
        self.redraw_self()

    def new_curve(self):
        if self.crv is None:
            return
        
        self.sketch_curve.write_json("save_crv.json")
        ret_index = self.ge
        mask_id = f"{self.mask_id_number.slider.maximum()}"
        self.last_index = self.handle_filenames.add_mask_id(ret_index, mask_id)

        # Actually convert the curve
        width_rgb_image = self.crv.image_rgb.shape[1]
        height_rgb_image = self.crv.image_rgb.shape[0]
        crv_in_image_coords = self.sketch_curve.convert_image(lower_left=self.lower_left, upper_right=self.upper_right, 
                                                              width=width_rgb_image, height=height_rgb_image)
        self.sketch_curve.write_json("save_crv_in_image.json")

        # Will create a mask image
        self.crv_from_sketch = FitBezierCyl2DSketch.create_from_filenames(self.handle_filenames,
                                                                          crv_in_image_coords,
                                                                          self.last_index)
        self.reset_file_menus()
        self.mask_number.set_value(self.last_index[2])
        self.mask_id_number.set_value(self.last_index[3])
        self.refit()
        self.read_images()

    def set_corners(self):
        """ Calculate the lower left and upper right corners of the image in the window frame"""

        if self.crv == None:
            return
        
        width_rgb_image = self.crv.image_rgb.shape[1]
        height_rgb_image = self.crv.image_rgb.shape[0]

        width_window = self.glWidget.width()
        height_window = self.glWidget.height()

        # The rectangle of the image in window coordinates
        self.lower_left = [0, 0]
        self.upper_right = [width_window, height_window]

    def set_crv(self, params):
        """Read in the images etc and recalc (or not)
        @param params - if None, recalculate"""
        print(f"{self.handle_filenames.get_image_name(index=self.last_index, b_add_tag=True)}")

        b_recalc = False
        if params is not None:
            b_recalc = True
        self.extract_crv = ExtractCurves.create_from_filenames(self.handle_filenames,
                                                               index=self.last_index,
                                                               b_do_debug=False,
                                                               b_do_recalc=b_recalc)
        self.crv = self.extract_crv.bezier_edge

        depth_fname = self.handle_filenames.get_depth_image_name(index=self.last_index, b_add_tag=True)
        if exists(depth_fname):
            depth_fname_calculate = self.handle_filenames.get_mask_name(index=self.last_index, b_calculate_path=True, b_add_tag=False)
            depth_fname_debug = self.handle_filenames.get_mask_name(index=self.last_index, b_debug_path=True, b_add_tag=False)
            params = {"camera_width_angle": self.horizontal_angle.value()}
            self.fit_crv_3d = FitBezierCyl3dDepth(depth_fname, self.crv.bezier_crv_fit_to_edge,
                                                  params=params,
                                                  fname_calculated=depth_fname_calculate,
                                                  fname_debug=depth_fname_debug, b_recalc=b_recalc)

    def read_images(self):
        if self.in_read_images:
            return
        self.in_read_images = True
        if self.handle_filenames is not None:
            print("Read images")
            b_get_image, self.last_index = self.reset_file_menus()
            print(f" masks {self.handle_filenames.mask_ids[self.last_index[0]][self.last_index[1]][self.last_index[2]]}")
            if b_get_image:
                self.image_names = {}
                self.image_names["rgb"] = self.handle_filenames.get_image_name(index=self.last_index, b_add_tag=True)
                self.image_names["mask"] = self.handle_filenames.get_mask_name(index=self.last_index, b_add_tag=True)
                self.image_names["edge"] = self.handle_filenames.get_edge_name(index=self.last_index, b_add_tag=True)
                self.image_names["flow"] = self.handle_filenames.get_flow_image_name(index=self.last_index, b_add_tag=True)
                self.image_names["depth"] = self.handle_filenames.get_depth_image_name(index=self.last_index, b_add_tag=True)

                self.images = {}
                for k, v in self.image_names.items():
                    if exists(v):
                        self.images[k] = cv2.imread(v)

                self.set_crv(params=None)

                self.glWidget.bind_texture(self.images)
                self.set_corners()
                self.redraw_self()
        self.in_read_images = False

    def resizeEvent(self, event):
        # Really only need to do this on resize
        self.set_corners()

    def redraw_self(self):
        self.glWidget.update()
        self.repaint()


if __name__ == '__main__':
    app = QApplication([])

    gui = SketchCurvesMainWindow()

    gui.show()

    app.exec_()
