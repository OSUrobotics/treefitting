#!/usr/bin/env python3

from os.path import exists
import json
import numpy as np

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, \
    QSpacerItem, QRadioButton
from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QTextEdit, QSizePolicy
import cv2

from MySliders import SliderIntDisplay, SliderFloatDisplay
from draw_routines.image_draw_geom_utils import draw_line_seg
from sketch_curves_gui.opengl_draw_window import OopenGLDrawWindow
from tree_geometry.line_segs import LineSeg2D
from utils.video_annotation_data import VideoAnnotationData
from utils.file_names_sub_dirs import FileNamesSubDirs
from utils.camera_projections import CameraProjections

from utils.sketched_curve import SketchedCurve

class DataAnnotationMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Data Annotation Viewer')

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        #middle_layout = self._init_middle_layout_()
        right_side_layout = self._init_right_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        self.last_read_image_index = -1

        # Two side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderFloatDisplay.gui = self
        SliderIntDisplay.gui = self

        self.cam_rgb = CameraProjections(camera_fname=("azure_camera.json", "rgb_half_size"),
                                    camera_calibration_fname=("azure_camera_calibration.json", "color"),
                                    params={})
        self.cam_depth = CameraProjections(camera_fname=("azure_camera.json", "depth_narrow_unbinned"),
                                      camera_calibration_fname=("azure_camera_calibration.json", "depth"),
                                      params={})

        self.glWidget.draw_curve_2d.show_profile_curves = False
        self.glWidget.draw_curve_2d.show_edge_rects = False
        self.glWidget.draw_curve_2d.show_interior_rects = False

        self.video_annot = None
        self.in_reset_file_menus = False
        self.in_read_images = False
        self.sketch_curve = SketchedCurve()
        self.x_rgb_to_depth = 1.0
        self.y_rgb_to_depth = 1.0
        # Pairs of points for calculating the rgb to depth matrix
        self.pts_rgb_to_depth = {"rgb": [], "depth": []}

        if exists("save_rgb_to_depth_pts.json"):
            with open ("save_rgb_to_depth_pts.json", "r") as f:
                self.pts_rgb_to_depth = json.load(f)

        if exists("save_crv.json"):
            with open("save_crv.json", "r") as f:
                my_dict = json.load(f)
                self.sketch_curve = SketchedCurve.read_json(my_dict)

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        # path_names_layout.setSpacing(5)
        path_names.setLayout(path_names_layout)
        #path_names_layout.setColumnMinimumWidth(0, 40)
        # path_names_layout.setColumnMinimumWidth(1, 200)
        src_drive = FileNamesSubDirs.get_path() + "/PycharmProjects/data/"
        tree_name = "bush_1_east"
        self.path_name = QLineEdit(src_drive + tree_name + "/")
        self.file_name = QLineEdit("video_annot_final.json")
        self.image_number = SliderIntDisplay("Image", 0, 10, 0)
        self.mask_number = SliderIntDisplay("Type", 0, 3, 0)
        self.mask_id_number = SliderIntDisplay("Type id", 0, 3, 0)

        path_names_layout.addWidget(QLabel("Path dir:"), 0, 0)
        path_names_layout.addWidget(self.path_name, 0, 1)
        path_names_layout.addWidget(QLabel("File data names:"), 1, 0)
        path_names_layout.addWidget(self.file_name, 1, 1)
        path_names_layout.addWidget(self.image_number, 2, 0, 1, 2)
        path_names_layout.addWidget(self.mask_number, 3, 0, 1, 2)
        path_names_layout.addWidget(self.mask_id_number, 4, 0, 1, 2)

        self.image_number.slider.valueChanged.connect(self.read_images)
        self.mask_number.slider.valueChanged.connect(self.read_images)
        self.mask_id_number.slider.valueChanged.connect(self.read_images)

        self.image_name = QLabel("image name")
        self.mask_name = QLabel("None")
        path_names_layout.addWidget(self.image_name, 5, 0)
        path_names_layout.addWidget(self.mask_name, 5, 1)

        read_filenames_button = QPushButton('Read file')
        read_filenames_button.clicked.connect(self.read_file_names)
        write_filenames_button = QPushButton('Write file')
        write_filenames_button.clicked.connect(self.write_file_names)
        path_names_layout.addWidget(read_filenames_button, 6, 0)
        path_names_layout.addWidget(write_filenames_button, 6, 1)


        # Sliders for Camera
        reset_view = QPushButton('Reset view')
        reset_view.clicked.connect(self.reset_view)
        self.turntable = SliderFloatDisplay('Rotate turntable', 0.0, 360, 0, 361)
        self.up_down = SliderFloatDisplay('Up down', 0, 360, 0, 361)
        self.zoom = SliderFloatDisplay('Zoom', 0.6, 2.0, 1.0)
        self.horizontal_angle = SliderFloatDisplay('Angle', 45, 175, 90)

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera.setLayout(params_camera_layout)
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)
        params_camera_layout.addWidget(self.zoom)
        params_camera_layout.addWidget(self.horizontal_angle)
        params_camera_layout.addWidget(reset_view)

        params_crvs = QGroupBox('3D Data parameters')
        params_crvs_layout = QGridLayout()
        params_crvs.setLayout(params_crvs_layout)
        self.show_3d_crv_button = QCheckBox('Show 3d crv')
        self.show_3d_crv_button.clicked.connect(self.redraw_self)
        self.show_3d_crv_axis_button = QCheckBox('Show 3d crv axis')
        self.show_3d_crv_axis_button.clicked.connect(self.redraw_self)
        self.show_3d_pts_button = QCheckBox('Show 3d pts')
        self.show_3d_pts_button.clicked.connect(self.redraw_self)
        self.n_around = SliderIntDisplay("N around", 8, 64, 32)
        self.n_along = SliderIntDisplay("N along", 8, 64, 16)
        params_crvs_layout.addWidget(self.show_3d_crv_button, 0, 0)
        params_crvs_layout.addWidget(self.show_3d_crv_axis_button, 1, 0)
        params_crvs_layout.addWidget(self.show_3d_pts_button, 2, 0)
        params_crvs_layout.addWidget(self.n_around, 0, 1)
        params_crvs_layout.addWidget(self.n_along, 1, 1)

        # For showing images and curves
        shows = QGroupBox('Shows')
        shows_layout = QGridLayout()
        shows.setLayout(shows_layout)

        self.show_rgb_button = QCheckBox('Show rgb')
        self.show_rgb_button.setCheckState(2)
        self.show_rgb_button.clicked.connect(self.redraw_self)
        self.show_overlay_button = QCheckBox('Overlay next')
        self.show_overlay_button.clicked.connect(self.redraw_self)
        self.show_depth_button = QCheckBox('Show depth')
        self.show_depth_button.clicked.connect(self.redraw_self)
        self.show_edge_button = QCheckBox('Show edge')
        self.show_edge_button.clicked.connect(self.redraw_self)

        self.show_backbone_button = QCheckBox('Show backbone')
        self.show_backbone_button.setCheckState(2)
        self.show_backbone_button.clicked.connect(self.redraw_self)
        self.show_sketch_crv_button = QCheckBox('Show sketch')
        self.show_sketch_crv_button.clicked.connect(self.redraw_self)
        self.show_sketch_crv_button.setCheckState(2)

        shows_layout.addWidget(self.show_rgb_button, 0, 0)
        shows_layout.addWidget(self.show_edge_button, 1, 0)
        shows_layout.addWidget(self.show_depth_button, 2, 0)
        shows_layout.addWidget(self.show_overlay_button, 3, 0)
        shows_layout.addWidget(self.show_backbone_button, 0, 1)
        shows_layout.addWidget(self.show_sketch_crv_button, 1, 1)
        shows_layout.setSpacing(5)

        # Drawing
        drawing_states = QGroupBox('Drawing states         ')
        drawing_states_layout = QGridLayout()
        drawing_states.setLayout(drawing_states_layout)
        new_sketch_button = QPushButton('New curve')
        new_sketch_button.clicked.connect(self.new_curve)
        done_kp_button = QPushButton('Done kp')
        done_kp_button.clicked.connect(self.done_keypoints)
        clear_drawings_button = QPushButton('Clear drawings')
        clear_drawings_button.clicked.connect(self.clear_drawings)

        self.do_sketch_curve_draw = QRadioButton('Draw backbone')
        self.do_sketch_curve_draw.clicked.connect(self.on_draw_toggled)
        self.do_keypoints_background = QRadioButton('Draw background')
        self.do_keypoints_background.clicked.connect(self.on_draw_toggled)
        self.do_keypoints_depth_draw = QRadioButton('Draw depth keypoints')
        self.do_keypoints_depth_draw.clicked.connect(self.on_draw_toggled)
        self.do_sketch_curve_draw.setChecked(1)

        drawing_states_layout.addWidget(self.do_sketch_curve_draw, 0, 0)
        drawing_states_layout.addWidget(self.do_keypoints_background, 1, 0)
        drawing_states_layout.addWidget(self.do_keypoints_depth_draw, 2, 0)

        drawing_states_layout.addWidget(new_sketch_button, 0, 1)
        drawing_states_layout.addWidget(done_kp_button, 1, 1)
        drawing_states_layout.addWidget(clear_drawings_button, 2, 1)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()
        left_side_layout.addWidget(path_names)
        # left_side_layout.addStretch()
        left_side_layout.addWidget(params_crvs)
        left_side_layout.addWidget(shows)
        left_side_layout.addWidget(drawing_states)
        left_side_layout.addWidget(params_camera)

        return left_side_layout

    # Drawing screen and quit button
    def _init_right_layout_(self):
        # The display for the robot drawing
        self.glWidget = OopenGLDrawWindow(gui=self, parent=self, size_start=(3*640, 3*480))

        self.up_down.slider.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        self.blank_text = QTextEdit('Space')
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)
        #quit_button.setMinimumWidth(1280)
        quit_button.setMinimumWidth(1200)

        # Put them together, quit button on the bottom
        right_layout = QVBoxLayout()

        right_layout.addWidget(self.glWidget)
        right_layout.addWidget(self.blank_text)
        right_layout.addWidget(quit_button, stretch=10)

        return right_layout

    def reset_file_menus(self):
        if self.in_reset_file_menus:
            return

        # Poor person's lock
        self.in_reset_file_menus = True

        indx_image = self.image_number.value()
        indx_mask = self.mask_number.value()
        id_mask = self.mask_id_number.value()
        print(f"Begin reset file name {indx_image} {indx_mask} {id_mask}")
        b_changed = False
        sldr_maxs_orig = (self.image_number.slider.maximum(),
                        self.mask_number.slider.maximum(),
                        self.mask_id_number.slider.maximum())
        print(f"Sliders orig {sldr_maxs_orig}")

        if self.image_number.slider.maximum() != len(self.video_annot.image_names[0]):
            self.image_number.slider.setMaximum(len(self.video_annot.image_names[0]))
            b_changed = True
            print(f" Changing image number {self.image_number.slider.maximum()} {indx_image}")
        if indx_image >= self.image_number.slider.maximum():
            indx_image = 0
            self.image_number.set_value(indx_image)

        if self.mask_number.slider.maximum() != len(self.video_annot.mask_names):
            self.mask_number.slider.setMaximum(len(self.video_annot.mask_names))
            b_changed = True
            print(f" Changing mask number {self.mask_number.slider.maximum()} {indx_mask}")
        if indx_mask >= self.mask_number.slider.maximum():
            indx_mask = 0
            self.mask_number.set_value(indx_mask)

        if self.mask_id_number.slider.maximum() != len(self.video_annot.mask_ids[0][indx_image][indx_mask]):
            self.mask_id_number.slider.setMaximum(len(self.video_annot.mask_ids[0][indx_image][indx_mask]))
            b_changed = True
            print(f" Changing mask id number {self.mask_id_number.slider.maximum()} {id_mask}")
        if id_mask >= self.mask_id_number.slider.maximum():
            id_mask = 0
            self.mask_id_number.set_value(id_mask)

        sldr_maxs = (self.image_number.slider.maximum(),
                     self.mask_number.slider.maximum(),
                     self.mask_id_number.slider.maximum())

        print(f"index {indx_image} {indx_mask} {id_mask} sldrs {sldr_maxs} redo {b_changed}")
        self.in_reset_file_menus = False

        return (indx_image, indx_mask, id_mask)

    def get_file_name_tuple(self):
        return (self.sub_dir_number.value(), self.image_number.value(), self.mask_number.value(), self.mask_id_number.value())

    def read_file_names(self):
        fname = self.path_name.text() + self.file_name.text()
        with open(fname, 'r') as f:
            my_data = json.load(f)
            self.video_annot = VideoAnnotationData.read_json(my_data)
        self.reset_file_menus()
        self.last_image_index = -1
        self.read_images()
        self.set_draw_params_from_sliders()

    def write_file_names(self):
        fname = self.path_name.text() + self.file_name.text()
        indx = 0
        fname_next = fname[0:-5] + "_" + str(indx) + ".json"
        while exists(fname_next):
            indx += 1
            fname_next = fname[0:-5] + "_" + str(indx) + ".json"
        with open(fname_next, 'w') as f:
            json.dump(self.video_annot.write_json(), f, indent=4)

    def on_draw_toggled(self):
        self.redraw_self()

    def set_draw_params_from_sliders(self):
        """ Set all the draw drawing parameters from the sliders"""
        # Image
        self.glWidget.draw_images.draw_tex = "None"
        if self.show_rgb_button.isChecked():
            if self.show_overlay_button.isChecked():
                self.glWidget.draw_images.draw_tex = "rgb_edge_rgb_next"
            elif self.show_edge_button.isChecked():
                self.glWidget.draw_images.draw_tex = "rgb_edge"
            elif self.show_depth_button.isChecked():
                self.glWidget.draw_images.draw_tex = "rgb_depth"
            else:
                self.glWidget.draw_images.draw_tex = "rgb"
        elif self.show_edge_button.isChecked():
            self.glWidget.draw_images.draw_tex = "edge"
        elif self.show_depth_button.isChecked():
            self.glWidget.draw_images.draw_tex = "depth"

        # 2D Drawing
        self.glWidget.draw_curve_2d.show_backbone = self.show_backbone_button.isChecked()

        self.glWidget.draw_curve_2d.show_sketched_curve = self.show_sketch_crv_button.isChecked()

        # 3d Drawing
        self.glWidget.draw_curve_3d.show_axis = self.show_3d_crv_button.isChecked()
        self.glWidget.draw_curve_3d.show_mesh = self.show_3d_crv_axis_button.isChecked()

        self.glWidget.draw_curve_3d.n_around = self.n_around.value()
        self.glWidget.draw_curve_3d.n_along = self.n_along.value()

        if self.video_annot is not None:
            # Pull out the image name
            img_name = self.video_annot.get_image_name(index=(0, self.image_number.value(), 0, 0))
            img_name_split = img_name.split("/")
            indx_mask = self.mask_number.value()
            if indx_mask >= 0 and indx_mask < len(self.video_annot.mask_names):
                mask_name = self.video_annot.mask_names[indx_mask]
            else:
                mask_name = "none"

            self.mask_name.setText(mask_name)
            if len(img_name_split) > 2:
                self.image_name.setAccessibleName(img_name_split[-2])
                self.image_name.setText(img_name_split[-1])
            else:
                self.image_name.setText(img_name)

        self.glWidget.draw_sketch_curve = self.do_sketch_curve_draw.isChecked()

    def reset_view(self):
        self.turntable.set_value(0.0)
        self.up_down.set_value(0.0)
        self.zoom.set_value(1.0)
        self.redraw_self()

    def sizePolicy(self) -> 'QSizePolicy':
        return QSizePolicy.Fixed

    def clear_drawings(self):
        if self.do_sketch_curve_draw.isChecked():
            self.sketch_curve.clear()
        else:
            if self.video_annot is None:
                return

            kf_indx = self.image_number.value()
            if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
                return

            kf = self.video_annot.keyframes[kf_indx]
            if self.do_keypoints_background.isChecked():
                kf.pts_2d_background = []
            if self.do_keypoints_depth_draw.isChecked():
                if self.show_rgb_button.isChecked():
                    kf.pts_2d_rgb_depth = []
                elif self.show_depth_button.isChecked():
                    kf.pts_2d_depth = []
        self.redraw_self()

    def new_curve(self):
        if self.video_annot is None:
            return
        
        fname = "save_crv.json"
        with open(fname, "w") as f:
            json.dump(self.sketch_curve.write_json(), f, indent=4)

        # Actually convert the curve
        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right
        crv_in_image_coords = self.sketch_curve.convert_image(lower_left=ll, upper_right=ur,
                                                              width=width_rgb_image, height=height_rgb_image)
        fname = "save_crv_in_image.json"
        with open(fname, "w") as f:
            json.dump(self.sketch_curve.write_json(), f, indent=4)

        # Will create bspline curve
        indx = (self.image_number.value(), self.mask_number.value(), 0)
        self.video_annot.add_sketch(image_index=indx, sketch=crv_in_image_coords)

        fname = "save_video_annot.json"
        with open(fname, "w") as f:
            json.dump(self.video_annot.write_json(), f, indent=2)

        self.reset_file_menus()
        self.sketch_curve.clear()
        self.redraw_self()

    def _convert_pt_to_image_coords(self, pt_in, b_do_depth=False):
        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right

        # Convert pt from the QT window to the RGB image size
        x, y = self.sketch_curve.convert_pt(pt_in, lower_left=ll, upper_right=ur,
                                            width=width_rgb_image, height=height_rgb_image)
        pt_out = [x, y]
        if b_do_depth:
            # Converting from rgb image to depth - need to scale from rgb to depth
            pt_out[0] *= self.x_rgb_to_depth
            pt_out[1] *= self.y_rgb_to_depth

        return pt_out

    def _convert_pt_from_image_coords(self, pt_in, b_do_depth=False):
        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right

        pt_out = [0, 0]
        if b_do_depth:
            # Go from depth to rgb image size
            pt_out[0] = pt_in[0] / self.x_rgb_to_depth
            pt_out[1] = pt_in[1] / self.y_rgb_to_depth
        else:
            pt_out[0] = pt_in[0]
            pt_out[1] = pt_in[1]

        # Convert from rgb to qt wincow
        pt_out = self.sketch_curve.convert_pt_back(pt_in, lower_left=ll, upper_right=ur,
                                                   width=width_rgb_image, height=height_rgb_image)
        # Converting from rgb image to depth - need to scale to the rgb window size
        return pt_out

    def _find_transform(self, pts1, pts2, b_do_depth=False):
        """ Find the translate, scale, rotate that takes pts1 to pts2
        Kabash algorithm
         Note: Convert to image coordinates first
        :param pts1 - from points
        :param pts2 - to points
        :param b_do_depth - doing depth, so re-scale points to depth as well
        :return translate, rotate, scale"""
        from scipy.spatial.transform import Rotation as R
        import utils.matrix_routines_2d as mt
        from PIL import Image
        from draw_routines.image_draw_geom_utils import draw_cross

        n_pts = len(pts1)
        # Convert all of the points from the QT window to the RGB image size
        pts_in_image1 = np.ones((3, n_pts))
        pts_in_image2 = np.ones((3, n_pts))
        for ps_out, ps_in in zip((pts_in_image1, pts_in_image2), (pts1, pts2)):
            for pi, pt_in in enumerate(ps_in):
                ps_out[:2, pi] = pt_in

        # RGB image for the current frame
        kf_indx = self.image_number.value()
        rgb_image = np.array(Image.open(self.video_annot.get_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
        if b_do_depth:
            # Depth image for the current frame
            to_image = np.array(Image.open(self.video_annot.get_depth_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
        else:
            to_image = np.array(Image.open(self.video_annot.get_image_name((0, kf_indx + 1, 0, 0)))).astype(np.uint8)

        # Draw the points in the image for debug purposes
        for indx in range(0, n_pts):
            draw_cross(im=rgb_image, p=[pts_in_image1[0:2, indx]], color=[255, 255, 255], thickness=2, length=6)
            draw_cross(im=to_image, p=[pts_in_image2[0:2, indx]], color=[155, 155, 255], thickness=2, length=6)

        rgb_write = Image.fromarray(rgb_image)
        rgb_name = self.video_annot.get_image_name((0, kf_indx, 0, 0), b_debug_path=self.video_annot.path_debug)
        rgb_write.save(rgb_name)

        img_to_write = Image.fromarray(to_image)
        if b_do_depth:
            to_write_name = self.video_annot.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=self.video_annot.path_debug)
        else:
            to_write_name = self.video_annot.get_image_name((0, kf_indx + 1, 0, 0), b_debug_path=self.video_annot.path_debug)
        img_to_write.save(to_write_name)

        pt_center1 = np.mean(pts_in_image1, axis=1)
        pt_center2 = np.mean(pts_in_image2, axis=1)
        # Move center from center1 to center 2
        vec_t = pt_center2 - pt_center1
        vec_t = [vec_t[0], vec_t[1]]
        mat_t2 = mt.make_translation_matrix(pt_center2[0], pt_center2[1])
        mat_tinv2 = mt.make_translation_matrix(-pt_center2[0], -pt_center2[1])
        mat_tinv1 = mt.make_translation_matrix(-pt_center1[0], -pt_center1[1])

        # points centered around 0,0
        pts_in_image1_centered = mat_tinv1 @ pts_in_image1
        pts_in_image2_centered = mat_tinv2 @ pts_in_image2

        # Rotate the points from 1 to 2
        rot, _ = R.align_vectors(pts_in_image1_centered.T, pts_in_image2_centered.T)
        angs = rot.as_euler('XYZ')
        mat_rot = mt.make_rotation_matrix(angs[2])
        if b_do_depth:
            # No rotation in the depth image
            mat_rot = np.identity(3)

        pts_in_image1_rotated = mat_rot @ pts_in_image1_centered
        scl = [1, 1]
        save_scls = np.zeros((n_pts, 1))
        for indxs in range(0, 2):
            for indxp in range(0, n_pts):
                save_scls[indxp] = pts_in_image2_centered[indxs, indxp] / pts_in_image1_rotated[indxs, indxp]
            scl[indxs] = save_scls.mean()

        mat_scl = mt.make_scale_matrix(scl[0], scl[1])

        # mat to take points in image 1 to image 2 = mat_t1 @ mat_scl @ mat_rot @ mat_tinv2
        mat = mat_t2 @ mat_scl @ mat_rot @ mat_tinv1

        pts_aligned = mat @ pts_in_image1
        pts_aligned_rev = np.linalg.inv(mat) @ pts_in_image2
        for indx in range(0, n_pts):
            p_from = pts_aligned[0:2, indx]
            p_to = pts_in_image2[0:2, indx]
            line = LineSeg2D(p_from, p_to)
            draw_cross(im=to_image, p=[pts_aligned[0:2, indx]], color=[25, 150, 25], thickness=2, length=4)
            draw_line_seg(to_image, line, color=(100, 100, 255), thickness=2)

            line2 = LineSeg2D(pts_aligned_rev[0:2, indx], pts_in_image1[0:2, indx])
            draw_cross(im=rgb_image, p=[pts_aligned_rev[0:2, indx]], color=[25, 150, 25], thickness=2, length=4)
            draw_line_seg(rgb_image, line2, color=(100, 100, 255), thickness=2)

        draw_cross(im=to_image, p=[to_image.shape[1] // 2, to_image.shape[0] // 2], color=[255, 10, 20], thickness=2, length=8)
        Image.fromarray(to_image).save(to_write_name)
        Image.fromarray(rgb_image).save(rgb_name)

        return vec_t, angs[2], scl, mat

    def done_keypoints(self):
        if self.video_annot is None:
            return

        fname = "save_video_annot.json"
        with open(fname, "w") as f:
            json.dump(self.video_annot.write_json(), f, indent=2)

        kf_indx = self.image_number.value()
        if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            return

        kf = self.video_annot.keyframes[kf_indx]
        if self.do_keypoints_background.isChecked():
            if kf_indx < self.video_annot.n_keyframes()-1:
                print(f"Optical flow keypoints")
                kf_next = self.video_annot.keyframes[kf_indx + 1]

                kp1 = kf.pts_2d_background
                kp2 = kf_next.pts_2d_background
                if len(kp1) != len(kp2) or len(kp1) < 1:
                    print(f"Number of key points differs or no key points {len(kp1)} {len(kp2)}")
                else:
                    vec_t, ang, scl, _ = self._find_transform(kp1, kp2)
                    kf.pan_vec = vec_t
                    kf.scale_amount = scl
                    kf.rot_amount = ang

        elif self.do_keypoints_depth_draw.isChecked():
            print(f"Depth keypoints")
            kp1 = kf.pts_2d_rgb_depth
            kp2 = kf.pts_2d_depth
            if len(kp1) != len(kp2) or len(kp1) < 1:
                print(f"Number of key points differs or no key points {len(kp1)} {len(kp2)}")
            else:
                _, _, _, mat = self._find_transform(kp1, kp2, b_do_depth=True)
                kf.rgb_to_depth_matrix = mat
        self.video_annot.crvs_in_depth_image()

        with open(fname, "w") as f:
            json.dump(self.video_annot.write_json(), f, indent=2)

    def read_images(self):
        if self.in_read_images:
            return

        if self.video_annot is None:
            self.in_read_images = False
            return

        kf_indx = self.image_number.value()
        if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            self.in_read_images = False
            return

        kf = self.video_annot.keyframes[kf_indx]
        print(f"Read images from {self.video_annot.path}")

        if self.last_image_index == self.image_number.value():
            self.set_draw_params_from_sliders()
            self.in_read_images = False
            return
        else:
            self.last_image_index = self.image_number.value()

        self.image_names = {}
        indx = (0, self.image_number.value(), 0, 0)
        self.image_names["rgb"] = self.video_annot.get_image_name(index=indx, b_add_tag=True)
        self.image_names["edge"] = self.video_annot.get_edge_name(index=indx, b_add_tag=True)
        self.image_names["depth"] = self.video_annot.get_depth_image_name(index=indx, b_add_tag=True)
        if self.image_number.value() < len(self.video_annot.image_names[0]) - 1:
            index_next = (0, self.image_number.value()+1, 0, 0)
            self.image_names["rgb_edge_rgb_next"] = self.video_annot.get_edge_name(index=index_next, b_add_tag=True)
        else:
            # Get a copy of this image
            self.image_names["rgb_edge_rgb_next"] = self.video_annot.get_image_name(index=indx, b_add_tag=True)

        self.images = {}
        for k, v in self.image_names.items():
            if exists(v):
                self.images[k] = cv2.imread(v)
                if k == "rgb":
                    alpha = 1.1  # Controls contrast (alpha > 1 increases contrast)
                    beta = 1.5    # Controls brightness (positive beta increases brightness)

                    self.images[k] = cv2.convertScaleAbs(self.images[k], alpha, beta)
            else:
                self.images[k] = None

        if self.images["rgb"] is not None:
            width_rgb_image = self.images["rgb"].shape[1]
            height_rgb_image = self.images["rgb"].shape[0]

            width_window = self.glWidget.width()
            height_window = self.glWidget.height()

            # The rectangle of the image in window coordinates
            self.glWidget.draw_curve_2d.im_size = (width_rgb_image, height_rgb_image)
            self.glWidget.draw_curve_2d.lower_left = [0, 0]
            self.glWidget.draw_curve_2d.upper_right = [width_window, height_window]
            self.glWidget.draw_curve_2d.aspect_ratio = height_rgb_image / width_rgb_image

            w = self.glWidget.width()
            h = int(self.glWidget.draw_curve_2d.aspect_ratio * w)

            self.glWidget.resize(w, h)

            if self.images["depth"] is not None:
                width_depth_image = self.images["depth"].shape[1]
                height_depth_image = self.images["depth"].shape[0]
                self.x_rgb_to_depth = width_depth_image / width_rgb_image
                self.y_rgb_to_depth = height_depth_image / height_rgb_image

        #mat_depth_to_rgb = kf.rgb_to_depth_matrix
        mat_depth_to_rgb = np.linalg.inv(self.cam_depth.rgb_to_depth_matrix)

        self.glWidget.draw_images.bind_texture(rgb_image=self.images["rgb"],
                                               edge_image=self.images["edge"],
                                               depth_image=self.images["depth"],
                                               next_rgb_image=self.images["rgb_edge_rgb_next"],
                                               mat_depth_to_rgb=mat_depth_to_rgb)

        self.redraw_self()
        self.in_read_images = False

    def sketched_curves(self):
        kf_indx = self.image_number.value()
        if not self.video_annot or kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            return [self.sketch_curve]

        kf = self.video_annot.keyframes[kf_indx]
        sk_crvs = [self.sketch_curve]
        # Actually convert the curve
        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right
        for mask_index, _ in enumerate(self.video_annot.mask_names):
            for mask_id in range(0, len(kf.sketch_curves[mask_index])):
                crv_screen_coords = kf.get_sketch(mask_index, mask_id).convert_back_to_screen(lower_left=ll, upper_right=ur,
                                                   width=width_rgb_image, height=height_rgb_image)
                sk_crvs.append(crv_screen_coords)
        return sk_crvs

    def spline_curves(self):
        if self.video_annot:
            crv_list = []
            image_indx = self.image_number.value()
            if image_indx < len(self.video_annot.keyframes):
                kf = self.video_annot.keyframes[image_indx]
                for mask_index, cyl_type in enumerate(kf.bspline_cyls):
                    for mask_id, crv in enumerate(cyl_type):
                        if self.show_rgb_button.isChecked():
                            crv_list.append(crv)
                        else:
                            depth_crv = kf.get_bsplinecyl_in_depth_image(mask_index=mask_index,
                                                                         mask_id_index=mask_id,
                                                                         mat_transform=self.cam_depth.rgb_to_depth_matrix)
                            new_pts = []
                            for indx in range(0, depth_crv.n_points()):
                                pt = depth_crv.point(indx)
                                pt[0] /= self.x_rgb_to_depth
                                pt[1] /= self.y_rgb_to_depth
                                new_pts.append(pt)
                            depth_crv.set_points(new_pts)
                            crv_list.append(depth_crv)
            return crv_list
        return []

    def sketch_vector(self, dx, dy):
        """ User drew a long vector in the window - pass the vector to the keyframe
        @param dx - change in x
        @param dy - change in y"""
        if self.video_annot is not None:
            kf_indx = self.image_number.value()
            self.video_annot.keyframes[kf_indx].pan_vec = [dx, dy]

    def _delete_key_point(self, pts, x, y):
        """ Delete (in place) the keypoint"""
        closest_i = -1
        best_d = 50
        for indx, pt in enumerate(pts):
            dist = (pt[0] - x) ** 2 + (pt[1] - y) ** 2
            if dist < best_d:
                closest_i = indx
                best_d = dist
        if closest_i != -1:
            pts[closest_i] = None

    def key_point(self, x, y, b_del=True):
        """ User clicks in window - keep click point
        @param x - x
        @param y - y"""
        if self.video_annot is None:
            return

        kf_indx = self.image_number.value()
        if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            return

        kf = self.video_annot.keyframes[kf_indx]

        if self.do_keypoints_end_draw.isChecked():
            if b_del:
                self._delete_key_point(kf.pts_2d_of_end, x, y)
            else:
                kf.pts_2d_of_end.append(self._convert_pt_to_image_coords([x, y]))

        if self.do_keypoints_background.isChecked():
            if b_del:
                self._delete_key_point(kf.pts_2d_background, x, y)
            else:
                kf.pts_2d_background.append(self._convert_pt_to_image_coords([x, y]))

        if self.do_keypoints_depth_draw.isChecked():
            pts_list = self.pts_rgb_to_depth["rgb"]
            b_do_depth = False
            if not self.show_rgb_button.isChecked():
                pts_list = self.pts_rgb_to_depth["depth"]
                b_do_depth = True

            if b_del:
                self._delete_key_point(pts_list, x, y)
            else:
                pts_list.append(self._convert_pt_to_image_coords([x, y], b_do_depth=b_do_depth))

    def get_key_points(self):
        if self.video_annot is None:
            return []

        kf_indx = self.image_number.value()
        if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            return []

        kf = self.video_annot.keyframes[kf_indx]

        pts_and_colors = []
        if self.do_keypoints_background.isChecked():
            pts_and_colors.append((kf.pts_2d_background, Qt.white))
            if kf_indx+1 < self.video_annot.n_keyframes():
                kf_next = self.video_annot.keyframes[kf_indx + 1]
                pts_and_colors.append((kf_next.pts_2d_background, Qt.yellow))
        if self.do_keypoints_depth_draw.isChecked():
            if self.show_rgb_button.isChecked():
                # These can just go in to list
                pts_and_colors.append((self.pts_rgb_to_depth["rgb"], Qt.cyan))
                # TODO Use rgb matrix to convert
                # pts_and_colors.append((kf.depth_pts_in_rgb(), Qt.green))
            else:
                # for all of these points, convert to rgb first
                pts_rgb = []
                pts_depth = []
                for pt in self.pts_rgb_to_depth["depth"]:
                    pts_rgb.append([pt[0] / self.x_rgb_to_depth, pt[1] / self.y_rgb_to_depth])
                for pt in self.pts_rgb_to_depth["depth"]:
                    pts_depth.append([pt[0] / self.x_rgb_to_depth, pt[1] / self.y_rgb_to_depth])
                pts_and_colors.append((pts_rgb, Qt.cyan))
                pts_and_colors.append((pts_depth, Qt.blue))

        pts_and_colors_in_qt_window = []
        for pts, cols in pts_and_colors:
            pts_back = []
            for p in pts:
                pts_back.append(self._convert_pt_from_image_coords(p))
            pts_and_colors_in_qt_window.append((pts_back, cols))
        return pts_and_colors_in_qt_window

    def get_vector(self):
        n_frame = int(self.image_number.value())
        if n_frame < self.video_annot.n_keyframes():
            return self.video_annot.keyframes[n_frame].pan_vec
        else:
            return [0, 0]

    def resizeEvent(self, event):
        # Really only need to do this on resize
        if self.glWidget:
            width_window = self.glWidget.width()
            height_window = self.glWidget.height()

            self.glWidget.draw_curve_2d.upper_right = [width_window, height_window]

    def redraw_self(self):
        self.glWidget.update()
        self.set_draw_params_from_sliders()
        self.repaint()


if __name__ == '__main__':
    app = QApplication([])

    gui = DataAnnotationMainWindow()
    # gui.showMaximized()
    gui.show()

    app.exec_()
