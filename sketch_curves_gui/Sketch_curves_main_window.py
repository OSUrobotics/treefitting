#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit
import cv2

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../Image_based'))
from os.path import exists

from MySliders import SliderIntDisplay, SliderFloatDisplay
from Draw_spline_3D import DrawSpline3D
from HandleFileNames import HandleFileNames

from extract_curves import ExtractCurves


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
        self.in_reset_file_menus = False
        self.in_read_images = False


    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        self.path_name = QLineEdit("/Users/grimmc/PycharmProjects/treefitting/Image_based/data/")
        self.file_name = QLineEdit("forcindy_fnames.json")
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

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)
        params_camera_layout.addWidget(self.zoom)
        params_camera_layout.addWidget(reset_view)
        params_camera.setLayout(params_camera_layout)

        params_crvs = QGroupBox('3D Curve parameters')
        params_crvs_layout = QVBoxLayout()
        self.n_around = SliderIntDisplay("N around", 8, 64, 32)
        self.n_along = SliderIntDisplay("N along", 8, 64, 16)
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

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)
        quit_button.setMinimumWidth(640)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget(self.glWidget)
        mid_layout.addWidget(quit_button)

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
        curve_drawing.setLayout(curve_drawing_layout)

        # For showing images
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

        show_images = QGroupBox('Image shows')
        show_images_layout = QVBoxLayout()
        show_images_layout.addWidget(self.show_rgb_button)
        show_images_layout.addWidget(self.show_mask_button)
        show_images_layout.addWidget(self.show_edge_button)
        show_images_layout.addWidget(self.show_opt_flow_button)
        show_images_layout.addWidget(self.show_depth_button)
        show_images.setLayout(show_images_layout)

        # Drawing
        drawing_states = QGroupBox('Drawing states         ')
        drawing_states_layout = QVBoxLayout()
        self.draw_backbone_button = QCheckBox('Draw backbone')
        self.draw_backbone_button.clicked.connect(self.redraw_self)
        clear_drawings_button = QPushButton('Clear drawings')
        clear_drawings_button.clicked.connect(self.clear_drawings)

        drawing_states_layout.addWidget(self.draw_backbone_button)
        drawing_states_layout.addWidget(clear_drawings_button)

        drawing_states.setLayout(drawing_states_layout)

        # Put all the pieces in one box
        right_side_layout = QVBoxLayout()

        right_side_layout.addWidget(resets)
        right_side_layout.addWidget(curve_drawing)
        right_side_layout.addWidget(show_images)
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
        print(f"Begin rest file name {indx_sub_dir} {indx_image} {indx_mask} {id_mask}")
        b_changed = False
        sldr_maxs_orig = (self.sub_dir_number.slider.maximum(),
                     self.image_number.slider.maximum(),
                     self.mask_number.slider.maximum(),
                     self.mask_id_number.slider.maximum())
        print(f"Sliders orig {sldr_maxs_orig}")
        if self.image_number.slider.maximum() != len(self.handle_filenames.image_names[indx_sub_dir]):
            self.image_number.slider.setMaximum(len(self.handle_filenames.image_names[indx_sub_dir]))
            b_changed = True
            print(f" Changing image number {self.image_number.slider.maximum()} {indx_image}")
        if indx_image >= self.image_number.slider.maximum():
            indx_image = 0
            self.image_number.set_value(indx_image)

        if self.mask_number.slider.maximum() != len(self.handle_filenames.mask_names[indx_sub_dir][indx_image]):
            self.mask_number.slider.setMaximum(len(self.handle_filenames.mask_names[indx_sub_dir][indx_image]))
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

        self.image_name.setText(self.handle_filenames.get_image_name("", index=indx))
        sldr_maxs = (self.sub_dir_number.slider.maximum(),
                     self.image_number.slider.maximum(),
                     self.mask_number.slider.maximum(),
                     self.mask_id_number.slider.maximum())
        print(f"index {indx} sldrs {sldr_maxs} redo {b_changed}")
        self.in_reset_file_menus = False

        return b_changed, indx

    def reset_params_menus(self):
        """ Set all the sliders based on the stored curve"""
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
        self.handle_filenames = HandleFileNames.read_filenames(fname)
        self.sub_dir_number.slider.setMaximum(len(self.handle_filenames.image_names))
        self.sub_dir_number.slider.setValue(0)
        self.reset_file_menus()
        self.read_images()
        self.reset_params_menus()

    def reset_view(self):
        self.turntable.set_value(0.0)
        self.up_down.set_value(0.0)
        self.zoom.set_value(1.0)
        self.redraw_self()

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
        pass

    def set_crv(self, params):
        """Read in the images etc and recalc (or not)
        @param params - if None, recalculate"""
        print(f"{self.handle_filenames.get_image_name(self.handle_filenames.path, index=self.last_index, b_add_tag=True)}")

        rgb_fname = self.handle_filenames.get_image_name(path=self.handle_filenames.path, index=self.last_index, b_add_tag=True)
        edge_fname = self.handle_filenames.get_edge_image_name(path=self.handle_filenames.path_calculated, index=self.last_index, b_add_tag=True)
        mask_fname = self.handle_filenames.get_mask_name(path=self.handle_filenames.path, index=self.last_index, b_add_tag=True)
        edge_fname_debug = self.handle_filenames.get_mask_name(path=self.handle_filenames.path_debug, index=self.last_index, b_add_tag=False)

        edge_fname_calculate = self.handle_filenames.get_mask_name(path=self.handle_filenames.path_calculated, index=self.last_index, b_add_tag=False)

        if not exists(mask_fname):
            print(f"Error, file {mask_fname} does not exist")
        if not exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")

        #self.crv = FitBezierCyl2DEdge(rgb_fname, edge_fname, mask_fname, edge_fname_calculate, edge_fname_debug, b_recalc=False)

        b_recalc = False
        if params is not None:
            b_recalc = True
        self.extract_crv = ExtractCurves(rgb_fname, edge_fname, mask_fname,
                                         fname_calculated=edge_fname_calculate,
                                         params=params,
                                         fname_debug=edge_fname_debug,
                                         b_recalc=b_recalc)
        self.crv = self.extract_crv.bezier_edge

    def read_images(self):
        if self.in_read_images:
            return
        self.in_read_images = True
        if self.handle_filenames is not None:
            print("Read images")
            b_get_image, self.last_index = self.reset_file_menus()
            print(f" masks {self.handle_filenames.mask_ids[self.last_index[0]][self.last_index[1]][self.last_index[2]]}")
            if b_get_image:
                image_flow_name = self.handle_filenames.get_flow_image_name(path=self.handle_filenames.path, index=self.last_index, b_add_tag=True)
                image_depth_name = self.handle_filenames.get_depth_image_name(path=self.handle_filenames.path, index=self.last_index, b_add_tag=True)

                if exists(image_flow_name):
                    image_flow = cv2.imread(image_flow_name)
                else:
                    image_flow = None

                if exists(image_depth_name):
                    image_depth = cv2.imread(image_depth_name)
                else:
                    image_depth = None

                self.set_crv(params=None)
                self.glWidget.bind_texture(self.crv.image_rgb,
                                           self.crv.mask_crv.stats_dict.mask_image,
                                           self.crv.image_edge,
                                           image_flow,
                                           image_depth)
                self.redraw_self()
        self.in_read_images = False


    def redraw_self(self):
        self.glWidget.update()
        self.repaint()


if __name__ == '__main__':

    app = QApplication([])

    gui = SketchCurvesMainWindow()

    gui.show()

    app.exec_()