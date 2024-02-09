#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit

from fit_bezier_cyl_2d import bezier_cyl_3d_with_detail

import numpy as np

from MySliders import SliderIntDisplay, SliderFloatDisplay

class FittedCurvesViewerGUI(QMainWindow):
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

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        self.path_name = QLineEdit("Image_based/data/forcindy/")
        self.file_name = QLineEdit("forcindy_fnames.json")
        self.image_number = SliderIntDisplay("Image", 1, 10, 1)
        self.mask_number = SliderIntDisplay("Mask", 1, 3, 1)
        path_names_layout.addWidget(QLabel("Path dir:"))
        path_names_layout.addWidget(self.path_name)
        path_names_layout.addWidget(QLabel("File data names:"))
        path_names_layout.addWidget(self.file_name)
        path_names_layout.addWidget(QLabel("Image:"))
        path_names_layout.addWidget(self.image_number)
        path_names_layout.addWidget(QLabel("Mask:"))
        path_names_layout.addWidget(self.mask_number)
        path_names.setLayout(path_names_layout)

        read_filenames = QPushButton('Read file names')
        read_filenames.clicked.connect(self.read_file_names)

        file_io = QGroupBox('File io')
        file_io_layout = QVBoxLayout()
        file_io_layout.addWidget(path_names)
        file_io_layout.addWidget(read_filenames)
        file_io.setLayout(file_io_layout)

        # Sliders for Camera
        reset_view = QPushButton('Reset view')
        reset_view.clicked.connect(self.reset_view)
        self.turntable = SliderFloatDisplay('Rotate turntable', 0.0, 360, 0, 361)
        self.up_down = SliderFloatDisplay('Up down', 0, 360, 0, 361)
        self.zoom = SliderFloatDisplay('Zoom', 0.6, 2.0, 1.0)

        show_buttons = QGroupBox('Show buttons')
        show_buttons_layout = QGridLayout()

        show_backbone_button = QCheckBox('Show backbone')
        show_backbone_button.clicked.connect(self.show_backbone)

        show_interior_rects_button = QCheckBox('Show interior rects')
        show_interior_rects_button.clicked.connect(self.show_interior_rects)

        show_edge_rects_button = QCheckBox('Show edge rects')
        show_edge_rects_button.clicked.connect(self.show_edge_rets)

        show_buttons_layout.addWidget(show_backbone_button)
        show_buttons_layout.addWidget(show_interior_rects_button)
        show_buttons_layout.addWidget(show_edge_rects_button)
        show_buttons.setLayout(show_buttons_layout)

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera_layout.addWidget(show_buttons)
        params_camera_layout.addWidget(self.show_closeup_slider)
        params_camera_layout.addWidget(self.show_min_val_slider)
        params_camera_layout.addWidget(self.show_max_val_slider)
        params_camera_layout.addWidget(reset_view)
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)
        params_camera_layout.addWidget(self.zoom)
        params_camera.setLayout(params_camera_layout)

        params_crvs = QGroupBox('Curve parameters')
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
        self.glWidget = DrawPointCloud(self)

        self.up_down.slider.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget( self.glWidget )
        mid_layout.addWidget(quit_button)

        return mid_layout

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_right_layout_(self):
        # Recalculate the bins (MyPointCloud) then recalculae
        # cylinders using PCA criteria then fit those
        recalc_neighbors_button = QPushButton('Recalculate bins')
        recalc_neighbors_button.clicked.connect(self.recalc_bins)

        recalc_cylinder_button = QPushButton('Recalculate one cylinder')
        recalc_cylinder_button.clicked.connect(self.recalc_one_cylinder)

        recalc_pca_cylinder_button = QPushButton('Recalculate pca cylinder')
        recalc_pca_cylinder_button.clicked.connect(self.recalc_pca_cylinder)

        recalc_fit_cylinder_button = QPushButton('Recalculate fit cylinder')
        recalc_fit_cylinder_button.clicked.connect(self.recalc_fit_cylinder)

        new_id_button = QPushButton('Random new id')
        new_id_button.clicked.connect(self.new_random_id)

        resets = QGroupBox('Resets')
        resets_layout = QVBoxLayout()
        resets_layout.addWidget(recalc_neighbors_button)
        resets_layout.addWidget(recalc_cylinder_button)
        resets_layout.addWidget(recalc_pca_cylinder_button)
        resets_layout.addWidget(recalc_fit_cylinder_button)
        resets_layout.addWidget(new_id_button)
        resets.setLayout(resets_layout)

        # For setting the bin size, based on width of narrowest branch
        self.smallest_branch_width = SliderFloatDisplay('Width small branch', 0.01, 0.1, 0.015)
        self.largest_branch_width = SliderFloatDisplay('Width big branch', 0.05, 0.2, 0.1)
        self.branch_height = SliderFloatDisplay('Height cyl fit', 0.1, 0.3, 0.15)

        params_neighbors = QGroupBox('Neighbor parameters')
        params_neighbors_layout = QVBoxLayout()
        params_neighbors_layout.addWidget(self.smallest_branch_width)
        params_neighbors_layout.addWidget(self.largest_branch_width)
        params_neighbors_layout.addWidget(self.branch_height)
        params_neighbors.setLayout(params_neighbors_layout)

        # The parameters of the cylinder fit
        params_labels = QGroupBox('Connected parameters                  ')
        params_labels_layout = QVBoxLayout()
        self.mark_neighbor_pca = SliderFloatDisplay('Mark Neighbor PCA', 0.1, 0.9, 0.75)
        self.mark_neighbor_fit = SliderFloatDisplay('Mark Neighbor Fit', 0.1, 0.9, 0.5)

        params_labels_layout.addWidget(self.mark_neighbor_pca)
        params_labels_layout.addWidget(self.mark_neighbor_fit)

        params_labels.setLayout(params_labels_layout)

        # Put all the pieces in one box
        right_side_layout = QVBoxLayout()

        right_side_layout.addWidget(resets)
        right_side_layout.addStretch()
        right_side_layout.addWidget(params_neighbors)
        right_side_layout.addWidget(params_labels)

        return right_side_layout

    # Recalcualte the bins based on the current smallest branch value
    def recalc_bins(self):
        self.glWidget.my_pcd.create_bins(self.smallest_branch_width.value())
        self.glWidget.make_bin_gl_list()
        self.repaint()

    def recalc_one_cylinder(self):
        if not hasattr(self.glWidget, "selected_point"):
            self.new_random_id()
        pt_ids = self.glWidget.my_pcd.find_connected( self.glWidget.selected_point, self.connected_neighborhood_radius.value() )
        self.glWidget.cyl = self.glWidget.my_pcd.fit_cylinder(self.glWidget.selected_point, [ret[0] for ret in pt_ids])
        print( self.glWidget.cyl )
        self.glWidget.update()
        self.repaint()

    def recalc_pca_cylinder(self):
        self.glWidget.cyl_cover.find_good_pca(0.5, self.height(), self.smallest_branch_width.value(), self.largest_branch_width.value())
        fname = self.path_name + self.pcd_name + self.version_name + "_cyl_pca.txt"
        with open(fname, "w") as fid:
            self.glWidget.cyl_cover.write(fid)
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def recalc_fit_cylinder(self):
        self.glWidget.cyl_cover.optimize_cyl()
        fname = self.path_name + self.pcd_name + self.version_name + "_cyl_fit.txt"
        with open(fname, "w") as fid:
            self.glWidget.cyl_cover.write(fid)
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def new_random_id(self):
        id = np.random.uniform(0, len(self.glWidget.my_pcd.pc_data) )
        self.glWidget.selected_point = int( np.floor( id ) )
        self.glWidget.update()
        self.repaint()

    def set_closeup_slider(self):
        if self.glWidget.show_bins:
            self.show_closeup_slider.setMaximum(len(self.glWidget.bin_mapping))
            self.show_min_val_slider.set_range(1, 45)
            self.show_max_val_slider.set_range(10, 65)
        if self.glWidget.show_pca_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_pca))
            self.show_min_val_slider.set_range(0.0, 30.0)
            self.show_max_val_slider.set_range(4.0, 50.0)
        if self.glWidget.show_fitted_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_fitted))
            self.show_min_val_slider.set_range(0.0, 10.0)
            self.show_max_val_slider.set_range(1.0, 10.0)

    def show_closeup(self):
        self.glWidget.show_closeup = not self.glWidget.show_closeup
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_one(self):
        self.glWidget.show_one = not self.glWidget.show_one
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_pca_cylinders(self):
        self.glWidget.show_pca_cylinders = not self.glWidget.show_pca_cylinders
        self.glWidget.show_fitted_cylinders = False
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_fitted_cylinders(self):
        self.glWidget.show_fitted_cylinders = not self.glWidget.show_fitted_cylinders
        self.glWidget.show_pca_cylinders = False
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_bins(self):
        self.glWidget.show_bins = not self.glWidget.show_bins
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_isolated(self):
        self.glWidget.show_isolated = not self.glWidget.show_isolated
        self.glWidget.update()
        self.repaint()

    def read_point_cloud(self):
        fname_pcd = self.path_name.text() + self.pcd_name.text() + ".ply"
        fname_my_pcd = self.path_name.text() + self.pcd_name.text() + self.version_name.text() + "_pcd.txt"

        try:
            with open(fname_my_pcd, "r") as fid:
                self.glWidget.my_pcd.read(fid)
        except FileNotFoundError:
            self.glWidget.my_pcd.load_point_cloud(fname_pcd)
            self.glWidget.my_pcd.create_bins(self.smallest_branch_width.value())
            with open(fname_my_pcd, "w") as fid:
                self.glWidget.my_pcd.write(fid)

        self.glWidget.make_pcd_gl_list()
        self.glWidget.cyl_cover = CylinderCover()

    def read_pca_cylinders(self):
        fname = self.path_name.text() + self.pcd_name.text() + self.version_name.text() + "_cyl_pca.txt"
        try:
            with open(fname, "r") as fid:
                self.glWidget.cyl_cover.read(fid)
        except FileNotFoundError:
            print("File not found {0}".format(fname))

    def read_fit_cylinders(self):
        fname = self.path_name.text() + self.pcd_name.text() + self.version_name.text() + "_cyl_fit.txt"
        try:
            with open(fname, "r") as fid:
                self.glWidget.cyl_cover.read(fid)
        except FileNotFoundError:
            print("File not found {0}".format(fname))

    def redraw_self(self):
        self.glWidget.update()
        self.repaint()


if __name__ == '__main__':
    app = QApplication([])

    gui = PointCloudViewerGUI()

    gui.show()

    app.exec_()
