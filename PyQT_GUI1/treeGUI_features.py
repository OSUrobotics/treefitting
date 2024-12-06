from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap

def on_left_button_clicked(parent):
    """Handle left button click to show the previous image."""
    if parent.images and parent.image_index > 0:
        parent.image_index -= 1
        update_main_image(parent)

def on_file_selected(parent):
    """Handle file selection and load images."""
    options = QFileDialog.Options()
    file_paths, _ = QFileDialog.getOpenFileNames(
        None, "Select Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options
    )
    if file_paths:
        parent.images = file_paths
        parent.image_index = 0
        update_main_image(parent)

def on_right_button_clicked(parent):
    """Handle right button click to show the next image."""
    if parent.images and parent.image_index < len(parent.images) - 1:
        parent.image_index += 1
        update_main_image(parent)

def update_main_image(parent):
    """Update the main image label with the current image."""
    if parent.images:
        image_path = parent.images[parent.image_index]
        pixmap = QPixmap(image_path).scaled(parent.main_image_label.size(), aspectRatioMode=1)
        parent.main_image_label.setPixmap(pixmap)
    else:
        parent.main_image_label.setText("No Image Selected")
