#!/usr/bin/env python3

from os.path import exists

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QSpacerItem

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QTextEdit, QSizePolicy
import cv2

from MySliders import SliderIntDisplay, SliderFloatDisplay
from sketch_curves_gui.opengl_draw_window import OopenGLDrawWindow
from utils.video_annotation_data import VideoAnnotationData

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
        if exists("save_crv.json"):
            with open("save_crv.json", "r") as f:
                import json
                my_dict = json.load(f)
                self.sketch_curve = SketchedCurve.read_json(my_dict)

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        src_drive = "/Users/grimmc/PycharmProjects/data/"
        #src_drive = "/Users/cindygrimm/PycharmProjects/data/"
        self.path_name = QLineEdit(src_drive + "EnvyTree/BP_R1_East_tree2/")
        self.path_name = QLineEdit(src_drive + "/")
        self.file_name = QLineEdit("CindyEnvyPhone_video_annot.json")
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

        self.show_backbone_button = QCheckBox('Show backbone')
        self.show_backbone_button.setCheckState(2)
        self.show_backbone_button.clicked.connect(self.redraw_self)
        self.show_sketch_crv_button = QCheckBox('Show sketch')
        self.show_sketch_crv_button.clicked.connect(self.redraw_self)
        self.show_sketch_crv_button.setCheckState(2)

        shows_layout.addWidget(self.show_rgb_button)
        shows_layout.addWidget(self.show_overlay_button)
        shows_layout.addWidget(self.show_backbone_button)
        shows_layout.addWidget(self.show_sketch_crv_button)
        shows_layout.setSpacing(5)
        shows.setLayout(shows_layout)

        # Drawing
        drawing_states = QGroupBox('Drawing states         ')
        drawing_states_layout = QVBoxLayout()
        self.draw_backbone_button = QPushButton('New curve')
        self.draw_backbone_button.clicked.connect(self.new_curve)
        clear_drawings_button = QPushButton('Clear drawings')
        clear_drawings_button.clicked.connect(self.clear_drawings)

        drawing_states_layout.addWidget(self.draw_backbone_button)
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
        import json
        fname = self.path_name.text() + self.file_name.text()
        with open(fname, 'r') as f:
            my_data = json.load(f)
            self.video_annot = VideoAnnotationData.read_json(my_data)
        self.reset_file_menus()
        self.last_image_index = -1
        self.read_images()
        self.set_draw_params_from_sliders()

    def set_draw_params_from_sliders(self):
        """ Set all the draw drawing parameters from the sliders"""
        # Image
        self.glWidget.draw_images.draw_tex = "None"
        if self.show_rgb_button.isChecked():
            if self.show_overlay_button.isChecked():
                self.glWidget.draw_images.draw_tex = "rgb_edge_rgb_next"
            else:
                self.glWidget.draw_images.draw_tex = "rgb"

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
            img_name = self.video_annot.get_image_name(index=(self.image_number.value(), 0, 0))
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

    def reset_view(self):
        self.turntable.set_value(0.0)
        self.up_down.set_value(0.0)
        self.zoom.set_value(1.0)
        self.redraw_self()

    def sizePolicy(self) -> 'QSizePolicy':
        return QSizePolicy.Fixed

    def clear_drawings(self):
        self.sketch_curve.clear()
        self.redraw_self()

    def new_curve(self):
        import json
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

    def read_images(self):
        if self.in_read_images:
            return

        if self.video_annot is not None:
            print("Read images frame {self.}")

            if self.last_image_index == self.image_number.value():
                self.set_draw_params_from_sliders()
                self.in_read_images = False
                return
            else:
                self.last_image_index = self.image_number.value()

            self.image_names = {}
            indx = (self.image_number.value(), 0, 0)
            self.image_names["rgb"] = self.video_annot.get_image_name(index=indx, b_add_tag=True)
            self.image_names["edge"] = self.video_annot.get_edge_name(index=indx, b_add_tag=True)
            if indx[0] + 1 < len(self.video_annot.image_names[0]):
                index_next = (self.image_number.value()+1, 0, 0)
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

            self.glWidget.draw_images.bind_texture(rgb_image=self.images["rgb"],
                                                   edge_image=self.images["edge"],
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
                        crv_list.append(crv)
            return crv_list
        return []

    def sketch_vector(self, dx, dy):
        """ User drew a long vector in the window - pass the vector to the keyframe
        @param dx - change in x
        @param dy - change in y"""
        if self.video_annot is not None:
            kf_indx = self.image_number.value()
            self.video_annot.keyframes[kf_indx].pan_vec = [dx, dy]

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
