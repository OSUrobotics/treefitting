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
from sketch_curves_gui.opengl_draw_window import OopenGLDrawWindow
from utils.video_annotation_data import VideoAnnotationData
from utils.file_names_sub_dirs import FileNamesSubDirs

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

        self.glWidget.draw_curve_2d.show_profile_curves = False
        self.glWidget.draw_curve_2d.show_edge_rects = False
        self.glWidget.draw_curve_2d.show_interior_rects = False

        self.video_annot = None
        self.in_reset_file_menus = False
        self.in_read_images = False
        self.sketch_curve = SketchedCurve()
        self.scale_x_depth = 1.0
        self.scale_y_depth = 1.0
        if exists("save_crv.json"):
            with open("save_crv.json", "r") as f:
                my_dict = json.load(f)
                self.sketch_curve = SketchedCurve.read_json(my_dict)

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        src_drive = FileNamesSubDirs.get_path() + "/PycharmProjects/data/"
        tree_name = "bush_8_west"
        self.path_name = QLineEdit(src_drive + tree_name + "/")
        self.file_name = QLineEdit("video_annot.json")
        self.image_number = SliderIntDisplay("Image", 0, 10, 0)
        self.mask_number = SliderIntDisplay("Type", 0, 3, 0)
        self.mask_id_number = SliderIntDisplay("Type id", 0, 3, 0)

        path_names_layout.addWidget(QLabel("Path dir:"))
        path_names_layout.addWidget(self.path_name)
        path_names_layout.addWidget(QLabel("File data names:"))
        path_names_layout.addWidget(self.file_name)
        path_names_layout.addWidget(QLabel("Image:"))
        path_names_layout.addWidget(self.image_number)
        path_names_layout.addWidget(QLabel("Type:"))
        path_names_layout.addWidget(self.mask_number)
        path_names_layout.addWidget(QLabel("Type id:"))
        path_names_layout.addWidget(self.mask_id_number)

        names_layout = QHBoxLayout()
        self.image_name = QLabel("image name")
        self.mask_name = QLabel("None")
        names_layout.addWidget(self.image_name)
        names_layout.addWidget(self.mask_name)

        # path_names_layout.setSpacing(5)
        path_names.setLayout(path_names_layout)

        self.image_number.slider.valueChanged.connect(self.read_images)
        self.mask_number.slider.valueChanged.connect(self.read_images)
        self.mask_id_number.slider.valueChanged.connect(self.read_images)

        read_filenames_button = QPushButton('Read file names')
        read_filenames_button.clicked.connect(self.read_file_names)

        file_io = QGroupBox('File io')
        file_io_layout = QVBoxLayout()
        file_io_layout.addWidget(path_names)
        file_io_layout.addWidget(read_filenames_button)
        file_io_layout.addLayout(names_layout)
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

        params_crvs = QGroupBox('3D Data parameters')
        params_crvs_layout = QVBoxLayout()
        self.show_3d_crv_button = QCheckBox('Show 3d crv')
        self.show_3d_crv_button.clicked.connect(self.redraw_self)
        self.show_3d_crv_axis_button = QCheckBox('Show 3d crv axis')
        self.show_3d_crv_axis_button.clicked.connect(self.redraw_self)
        self.show_3d_pts_button = QCheckBox('Show 3d pts')
        self.show_3d_pts_button.clicked.connect(self.redraw_self)
        self.n_around = SliderIntDisplay("N around", 8, 64, 32)
        self.n_along = SliderIntDisplay("N along", 8, 64, 16)
        params_crvs_layout.addWidget(self.show_3d_crv_button)
        params_crvs_layout.addWidget(self.show_3d_crv_axis_button)
        params_crvs_layout.addWidget(self.show_3d_pts_button)
        params_crvs_layout.addWidget(self.n_around)
        params_crvs_layout.addWidget(self.n_along)
        params_crvs.setLayout(params_crvs_layout)

        # For showing images and curves
        shows = QGroupBox('Shows')
        shows_layout = QVBoxLayout()

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

        shows_layout.addWidget(self.show_rgb_button)
        shows_layout.addWidget(self.show_edge_button)
        shows_layout.addWidget(self.show_depth_button)
        shows_layout.addWidget(self.show_overlay_button)
        shows_layout.addWidget(self.show_backbone_button)
        shows_layout.addWidget(self.show_sketch_crv_button)
        shows_layout.setSpacing(5)
        shows.setLayout(shows_layout)

        # Drawing
        drawing_states = QGroupBox('Drawing states         ')
        drawing_states_layout = QGridLayout()
        new_sketch_button = QPushButton('New curve')
        new_sketch_button.clicked.connect(self.new_curve)
        done_kp_button = QPushButton('Done kp')
        done_kp_button.clicked.connect(self.done_keypoints)
        clear_drawings_button = QPushButton('Clear drawings')
        clear_drawings_button.clicked.connect(self.clear_drawings)

        self.do_sketch_curve_draw = QRadioButton('Draw backbone')
        self.do_sketch_curve_draw.clicked.connect(self.on_draw_toggled)
        self.do_keypoints_start_draw = QRadioButton('Draw start keypoints')
        self.do_keypoints_start_draw.clicked.connect(self.on_draw_toggled)
        self.do_keypoints_end_draw = QRadioButton('Draw end keypoints')
        self.do_keypoints_end_draw.clicked.connect(self.on_draw_toggled)
        self.do_keypoints_depth_draw = QRadioButton('Draw depth keypoints')
        self.do_keypoints_depth_draw.clicked.connect(self.on_draw_toggled)
        self.do_sketch_curve_draw.setChecked(1)

        drawing_states_layout.addWidget(self.do_sketch_curve_draw)
        drawing_states_layout.addWidget(self.do_keypoints_start_draw)
        drawing_states_layout.addWidget(self.do_keypoints_end_draw)
        drawing_states_layout.addWidget(self.do_keypoints_depth_draw)

        drawing_states_layout.addWidget(new_sketch_button)
        drawing_states_layout.addWidget(done_kp_button)
        drawing_states_layout.addWidget(clear_drawings_button)

        drawing_states.setLayout(drawing_states_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        left_side_layout.addWidget(file_io)
        left_side_layout.addStretch()
        left_side_layout.addWidget(params_camera)
        left_side_layout.addWidget(params_crvs)
        left_side_layout.addWidget(shows)
        left_side_layout.addWidget(drawing_states)

        return left_side_layout

    # Drawing screen and quit button
    def _init_right_layout_(self):
        # The display for the robot drawing
        self.glWidget = OopenGLDrawWindow(gui=self, parent=self, size_start=(2*640, 2*480))

        self.up_down.slider.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        self.blank_text = QTextEdit('Space')
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)
        quit_button.setMinimumWidth(1280)

        # Put them together, quit button on the bottom
        right_layout = QVBoxLayout()

        right_layout.addWidget(self.glWidget)
        right_layout.addWidget(self.blank_text)
        right_layout.addWidget(quit_button, stretch=20)

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
            if self.do_keypoints_start_draw.isChecked():
                kf.pts_2d_of_start = []
            if self.do_keypoints_end_draw.isChecked():
                kf.pts_2d_of_end = []
            if self.do_keypoints_depth_draw.isChecked():
                if self.show_rgb_button.isChecked():
                    self.kf.pts_2d_rgb_depth = []
                elif self.show_depth_button.isChecked():
                    self.pts_2d_depth = []
        self.redraw_self()

    def new_curve(self):
        if self.video_annot is None:
            return
        
        fname = "save_crv.json"
        with open(fname, "w") as f:
            json.dump(self.sketch_curve.write_json(), f)

        # Actually convert the curve
        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right
        crv_in_image_coords = self.sketch_curve.convert_image(lower_left=ll, upper_right=ur,
                                                              width=width_rgb_image, height=height_rgb_image)
        fname = "save_crv_in_image.json"
        with open(fname, "w") as f:
            json.dump(self.sketch_curve.write_json(), f)

        # Will create bspline curve
        indx = (self.image_number.value(), self.mask_number.value(), 0)
        self.video_annot.add_sketch(image_index=indx, sketch=crv_in_image_coords)

        fname = "save_video_annot.json"
        with open(fname, "w") as f:
            json.dump(self.video_annot.write_json(), f, indent=2)

        self.reset_file_menus()
        self.sketch_curve.clear()
        self.redraw_self()

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

        width_rgb_image = self.glWidget.draw_curve_2d.im_size[0]
        height_rgb_image = self.glWidget.draw_curve_2d.im_size[1]
        ll = self.glWidget.draw_curve_2d.lower_left
        ur = self.glWidget.draw_curve_2d.upper_right

        n_pts = len(pts1)
        pts_in_image1 = np.ones((3, n_pts))
        pts_in_image2 = np.ones((3, n_pts))
        for ps_out, ps_in in zip((pts_in_image1, pts_in_image2), (pts1, pts2)):
            for pi, p_in in enumerate(ps_in):
                pt_in_image = self.sketch_curve.convert_pt(p_in, lower_left=ll, upper_right=ur,
                                                           width=width_rgb_image, height=height_rgb_image)
                ps_out[:2, pi] = pt_in_image

        rgb_image = np.array(Image.open(self.video_annot.get_image_name((0, 0, 0, 0)))).astype(np.uint8)
        depth_image = np.array(Image.open(self.video_annot.get_depth_image_name((0, 0, 0, 0)))).astype(np.uint8)
        if b_do_depth:
            pts_in_image2[0, :] = pts_in_image2[0, :] * self.scale_x_depth
            #pts_in_image2[1, :] = (rgb_image.shape[1] - pts_in_image2[1, :] - 1) * self.scale_y_depth
            pts_in_image2[1, :] = pts_in_image2[1, :] * self.scale_y_depth

        for indx in range(0, n_pts):
            draw_cross(im=rgb_image, p=[pts_in_image1[0:2, indx]], color=[255, 255, 255], thickness=2, length=4)
            draw_cross(im=depth_image, p=[pts_in_image2[0:2, indx]], color=[255, 255, 255], thickness=2, length=4)

        rgb_write = Image.fromarray(rgb_image)
        rgb_name = self.video_annot.get_image_name((0, 0, 0, 0), b_debug_path=self.video_annot.path_debug)
        rgb_write.save(rgb_name)

        depth_write = Image.fromarray(depth_image)
        depth_name = self.video_annot.get_depth_image_name((0, 0, 0, 0), b_debug_path=self.video_annot.path_debug)
        depth_write.save(depth_name)

        pt_center1 = np.mean(pts_in_image1, axis=1)
        pt_center2 = np.mean(pts_in_image2, axis=1)
        # Move center from center1 to center 2
        vec_t = pt_center2 - pt_center1
        vec_t = [vec_t[0], vec_t[1]]
        mat_t2 = mt.make_translation_matrix(pt_center2[0], pt_center2[1])
        mat_tinv2 = mt.make_translation_matrix(-pt_center2[0], -pt_center2[1])
        mat_tinv1 = mt.make_translation_matrix(-pt_center1[0], -pt_center1[1])

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

        # mat = mat_t1 @ mat_scl @ mat_rot @ mat_tinv2
        mat = mat_t2 @ mat_scl @ mat_rot @ mat_tinv1

        pts_aligned = mat @ pts_in_image2
        print(f" {pts_in_image1}")
        print(f" {pts_aligned}")

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
        if self.do_keypoints_start_draw.isChecked():
            if kf_indx < self.video_annot.n_keyframes()-1:
                print(f"Optical flow keypoints")
                kf_next = self.video_annot.keyframes[kf_indx + 1]

                kp1 = kf.pts_2d_of_start
                kp2 = kf_next.pts_2d_of_end
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
                kf.image_matrix = mat

    def read_images(self):
        if self.in_read_images:
            return

        if self.video_annot is not None:
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
            if indx[0] + 1 < len(self.video_annot.image_names[0]):
                index_next = (0, self.image_number.value()+1, 0, 0)
                self.image_names["rgb_edge_rgb_next"] = self.video_annot.get_edge_name(index=index_next, b_add_tag=True)
            else:
                # Get a copy of this image
                self.image_names["rgb_edge_rgb_next"] = self.video_annot.get_image_name(index=indx, b_add_tag=True)

            self.images = {}
            for k, v in self.image_names.items():
                if exists(v):
                    self.images[k] = cv2.imread(v)
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
                    self.scale_x_depth = width_depth_image / width_rgb_image
                    self.scale_y_depth = height_depth_image / height_rgb_image

            self.glWidget.draw_images.bind_texture(rgb_image=self.images["rgb"],
                                                   edge_image=self.images["edge"],
                                                   depth_image=self.images["depth"],
                                                   next_rgb_image=self.images["rgb_edge_rgb_next"])

            self.redraw_self()
        self.in_read_images = False

    def sketched_curves(self):
        return [self.sketch_curve]

    def spline_curves(self):
        if self.video_annot:
            crv_list = []
            image_indx = self.image_number.value()
            if image_indx < len(self.video_annot.keyframes):
                kf = self.video_annot.keyframes[image_indx]
                for cyl_type in kf.bspline_cyls:
                    for crv in cyl_type:
                        if not self.show_depth_button.isChecked():
                            crv_list.append(crv)
                        else:
                            depth_crv = crv.transform(kf.image_matrix)
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

    def key_point(self, x, y):
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
            kf.pts_2d_of_end.append([x, y])

        if self.do_keypoints_start_draw.isChecked():
            kf.pts_2d_of_start.append([x, y])

        if self.do_keypoints_depth_draw.isChecked():
            if self.show_rgb_button.isChecked():
                kf.pts_2d_rgb_depth.append([x, y])
            else:
                kf.pts_2d_depth.append([x, y])

    def get_key_points(self):
        if self.video_annot is None:
            return []

        kf_indx = self.image_number.value()
        if kf_indx < 0 or kf_indx >= self.video_annot.n_keyframes():
            return []

        kf = self.video_annot.keyframes[kf_indx]

        pts_and_colors = []
        if self.do_keypoints_start_draw.isChecked():
            pts_and_colors.append((kf.pts_2d_of_start, Qt.white))
            if kf_indx+1 < self.video_annot.n_keyframes():
                kf_next = self.video_annot.keyframes[kf_indx + 1]
                pts_and_colors.append((kf_next.pts_2d_of_end, Qt.yellow))
        if self.do_keypoints_end_draw.isChecked():
            pts_and_colors.append((kf.pts_2d_of_end, Qt.yellow))
            if 0 <= kf_indx-1 < self.video_annot.n_keyframes():
                kf_prev = self.video_annot.keyframes[kf_indx - 1]
                pts_and_colors.append((kf_prev.pts_2d_of_start, Qt.white))
        if self.do_keypoints_depth_draw.isChecked():
            if self.show_rgb_button.isChecked():
                pts_and_colors.append((kf.pts_2d_rgb_depth, Qt.blue))
                pts_and_colors.append((kf.depth_pts_in_rgb(), Qt.green))
            else:
                pts_and_colors.append((kf.pts_2d_depth, Qt.green))
                pts_and_colors.append((kf.rgb_pts_in_depth(), Qt.blue))
        return pts_and_colors

    def get_vector(self):
        n_frame = int(self.image_number.value())
        return self.video_annot.keyframes[n_frame].pan_vec

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

    gui.show()

    app.exec_()
