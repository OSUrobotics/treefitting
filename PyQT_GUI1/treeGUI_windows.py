from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCharts import *
from PyQt6.QtGui import QPixmap

def windows(parent):
    """Set up the main layout and widgets, passing parent as context."""
    
    # Create the layout and sub-windows
    hbox = QHBoxLayout()
    sub_vbox = QVBoxLayout()
    main_vbox = QVBoxLayout()

    # For the subwindows (RGB, Depth, Optical Flow)
    RGB = QChart()
    Depth = QChart()
    OptFlow = QChart()

    RGB_view = QChartView(RGB)
    Depth_view = QChartView(Depth)
    OptFlow_view = QChartView(OptFlow)

    # Add borders to the QChartView widgets
    RGB_view.setStyleSheet("border: 1px solid black; background-color: white;")
    Depth_view.setStyleSheet("border: 1px solid black; background-color: white;")
    OptFlow_view.setStyleSheet("border: 1px solid black; background-color: white;")

    # Create layouts for each section
    rgb_layout = QVBoxLayout()
    rgb_layout.addWidget(QLabel("RGB"))
    rgb_layout.addWidget(RGB_view)

    depth_layout = QVBoxLayout()
    depth_layout.addWidget(QLabel("Depth"))
    depth_layout.addWidget(Depth_view)

    optflow_layout = QVBoxLayout()
    optflow_layout.addWidget(QLabel("Optical Flow"))
    optflow_layout.addWidget(OptFlow_view)

    # Add the three subwindow sections to the left-side vbox layout
    sub_vbox.addLayout(rgb_layout)
    sub_vbox.addLayout(depth_layout)
    sub_vbox.addLayout(optflow_layout)

    # For the Big window (Main Image Display)
    main_image_label = QLabel("No Image Selected")
    #main_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_image_label.setStyleSheet("border: 1px solid black; background-color: white;")

    main_layout = QVBoxLayout()
    main_layout.addWidget(QLabel("Main Image"))
    main_layout.addWidget(main_image_label)

    # Add the vbox layouts to the horizontal layout
    hbox.addLayout(sub_vbox)
    hbox.addLayout(main_layout)

    # Create navigation buttons
    left_button = QPushButton("Left")
    right_button = QPushButton("Right")
    file_button = QPushButton("Select File")

    # Connect buttons to parent functions
    left_button.clicked.connect(lambda: parent.on_left_button_clicked())
    right_button.clicked.connect(lambda: parent.on_right_button_clicked())
    file_button.clicked.connect(lambda: parent.on_file_selected())

    # Create button layout and add to the main layout
    button_layout = QHBoxLayout()
    button_layout.addWidget(left_button)
    button_layout.addWidget(file_button)
    button_layout.addWidget(right_button)

    hbox.addLayout(button_layout)

    # Return the layout, main image label, and buttons to the parent
    return hbox, main_image_label, left_button, right_button, file_button
