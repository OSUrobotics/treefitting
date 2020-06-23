from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor
import sys
import pickle
import os
import networkx as nx
from MachineLearningPanel import overlay_pixmap, LabelAndText
from autoencoder_experiment import sample_xys_from_image
import imageio
import numpy as np
import random
from scipy.spatial import KDTree

def get_open_cover(pc, radius):
    kd_tree = KDTree(pc, leafsize=100)

    to_assign = np.arange(0, pc.shape[0])
    np.random.shuffle(to_assign)
    output = []
    while len(to_assign):
        idx = to_assign[0]
        pt = pc[idx]
        all_pts_idx = kd_tree.query_ball_point(pt, radius)
        to_assign = np.setdiff1d(to_assign, all_pts_idx, assume_unique=True)
        # Recenter
        mean = pc[all_pts_idx].mean(axis=0)
        closest_idx = np.argmin(np.linalg.norm(pc - mean, axis=1))

        output.append(closest_idx)

    return output

def create_graph_from_cover(pc, cover, connection_radius):
    graph = nx.Graph()
    cover = np.array(cover)
    cover_pc = pc[cover]

    graph.add_nodes_from([(idx, dict(point=pc[idx])) for idx in cover])

    for idx in cover:
        pt = pc[idx]
        close = np.linalg.norm(cover_pc - pt, axis=1) < connection_radius
        for n_idx in cover[close]:
            graph.add_edge(idx, n_idx, weight=np.linalg.norm(pc[idx] - pc[n_idx]))

    return graph






class TreeLabelerCanvas(QWidget):
    def __init__(self, callbacks=None):
        """
        Initializes a canvas which can display a base image underneath it.
        :param image_size: Either an integer or a 2-tuple of integers
        :param hold_to_draw: If True, drawing will be done by holding down the mouse button.
                             Otherwise it draws every time you click
        :param pen_size: If not specified, either a pe
        """
        super(TreeLabelerCanvas, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Callbacks
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks

        # Image label setup
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        self.base_pixmap = QPixmap(512, 512)
        self.base_pixmap.fill(QtCore.Qt.black)
        self.overlay_pixmap = QPixmap(512, 512)
        self.overlay_pixmap.fill(QtCore.Qt.transparent)
        self.image_label.setPixmap(self.base_pixmap)
        self.image_label.mousePressEvent = self.handle_mouse_press

        # Diagnostic label setup
        diagnostic_layout = QHBoxLayout()
        self.dialog = QLabel('Please load a tree to get started.')
        diagnostic_layout.addWidget(self.dialog)
        layout.addLayout(diagnostic_layout)

        # Data and state information
        self.graph = None
        self.targets = None
        self.node_list = None
        self.selected_edges = []
        self.origin = None
        self.scaling = None
        self.anchor = None

        self.all_sequence_labels = []
        self.current_sequence = []
        self.current_edge_labels = []

    def load_pointcloud(self, pc, graph, hist_size=128, zoom=5):

        mins = pc.min(axis=0)
        maxs = pc.max(axis=0)
        self.origin = mins
        self.scaling = np.max(maxs - mins)

        normalized_pc = (pc - self.origin) / self.scaling   # Normalized to [0, 1] on each axis
        bounds = np.linspace(0, 1, hist_size+1)
        hist = np.histogram2d(normalized_pc[:,0], normalized_pc[:,1], bounds)[0]
        hist = hist / np.max(hist)
        # Hack to avoid annoyingness of loading pixmap
        imageio.imwrite('last_pc.png', hist)
        display_size = hist_size * zoom
        self.base_pixmap = QPixmap('last_pc.png').scaled(display_size, display_size)

        # Load the targets onto the base image
        all_pts = []
        node_list = []
        for node in graph:
            pt = graph.nodes[node]['point']
            pix = self.coord_to_pix(*pt)
            node_list.append(node)
            all_pts.append(pix)

            # Draw target
            self.draw_circle(*pix, rad=10, color=(255, 0, 0, 200))

        self.targets = np.array(all_pts)
        self.node_list = np.array(node_list)
        self.graph = graph
        self.anchor = None
        self.reset_sequence(clear_all=True)
        self.set_select_trunk_message()
        self.update()


    def update(self, overlay=None):
        pixmap = self.base_pixmap
        if overlay is not None:
            pixmap = overlay_pixmap(self.base_pixmap, overlay)
        self.image_label.setPixmap(pixmap)

    def update_all(self):
        overlay = self.blank_pixmap()
        self.draw_current_selection(overlay)
        if len(self.current_sequence) and not len(self.current_edge_labels):
            self.draw_node_neighbors(self.current_sequence[-1], overlay)
        self.update(overlay)

    def blank_pixmap(self):
        pixmap = QPixmap(self.base_pixmap.size())
        pixmap.fill(QtCore.Qt.transparent)
        return pixmap

    def pix_to_coord(self, x, y):
        im_size = self.base_pixmap.size().width()
        coord = np.array([x, y])[::-1]
        return (coord / im_size) * self.scaling + self.origin[::-1]

    def coord_to_pix(self, x, y):
        im_size = self.base_pixmap.size().width()
        coord = np.array([x, y])[::-1]
        return ((coord - self.origin[::-1]) / self.scaling * im_size).astype(np.int)

    def draw_circle(self, x, y, pixmap=None, rad=5, color=(255, 0, 0, 255)):

        if pixmap is None:
            pixmap = self.base_pixmap

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = painter.pen()
        pen.setWidth(5)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setColor(QColor(*color))
        painter.setPen(pen)
        painter.drawEllipse(int(x-rad/2), int(y-rad/2), rad, rad)
        painter.end()

    def draw_line(self, x1, y1, x2, y2, pixmap=None, color=(255, 0, 0, 255), style=QtCore.Qt.SolidLine):
        # Lots of copy-pasted code...
        if pixmap is None:
            pixmap = self.base_pixmap

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = painter.pen()
        pen.setWidth(5)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setStyle(style)
        pen.setColor(QColor(*color))
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)
        painter.end()

    def draw_current_selection(self, pixmap=None):
        if pixmap is None:
            pixmap = self.blank_pixmap()

        for i, (node_s, node_e) in enumerate(zip(self.current_sequence[:-1], self.current_sequence[1:])):
            pix_s = self.coord_to_pix(*self.graph.nodes[node_s]['point'])
            pix_e = self.coord_to_pix(*self.graph.nodes[node_e]['point'])

            try:
                tag, solid = self.current_edge_labels[i]
            except IndexError:
                tag, solid = None, True
            colors = {
                'b': (212, 49, 209, 200),
                'l': (44, 121, 209, 255),
                's': (62, 179, 88, 255),
                't': (148, 118, 35, 255),
                'o': (83, 174, 201, 255),
                'f': (100, 100, 100, 255),
            }
            color = colors.get(tag, (255, 255, 255, 255))
            style = QtCore.Qt.DotLine if not solid else QtCore.Qt.SolidLine

            self.draw_line(pix_s[0], pix_s[1], pix_e[0], pix_e[1], pixmap, color=color, style=style)

        return pixmap

    def draw_node_neighbors(self, node, pixmap=None):
        if pixmap is None:
            pixmap = self.blank_pixmap()
        graph_pt = self.graph.nodes[node]['point']
        pix = self.coord_to_pix(*graph_pt)
        for neighbor_node in self.graph[node]:
            try:
                if neighbor_node == self.current_sequence[-2]:
                    continue
            except IndexError:
                pass
            neighbor_pix = self.coord_to_pix(*self.graph.nodes[neighbor_node]['point'])
            self.draw_line(pix[0], pix[1], neighbor_pix[0], neighbor_pix[1], pixmap, color=(0, 0, 255, 128))
        self.draw_circle(*pix, pixmap, color=(0, 0, 255, 255))
        return pixmap

    def handle_mouse_press(self, e):
        pix = np.array([e.x(), e.y()])
        dists = np.linalg.norm(self.targets - pix, axis=1)
        min_dist_idx = np.argmin(dists)
        if dists[min_dist_idx] < 12.5:
            selected_node = self.node_list[min_dist_idx]
            if self.anchor is None:
                self.anchor = selected_node
                anchor_pix = self.coord_to_pix(*self.graph.nodes[self.anchor]['point'])
                self.draw_circle(*anchor_pix, self.base_pixmap, color=(54, 209, 40, 255))
                self.set_default_message()

            elif not len(self.current_sequence):
                self.current_sequence.append(selected_node)


            elif selected_node == self.current_sequence[-1]:
                if len(self.current_sequence) == 1:
                    self.generate_sequence_to_anchor()
                else:
                    callback = self.callbacks.get('label')
                    if callback is not None:
                        callback()
            else:
                if (self.current_sequence[-1], selected_node) in self.graph.edges:
                    self.current_sequence.append(selected_node)
                    if selected_node == self.anchor:
                        callback = self.callbacks.get('label')
                        if callback is not None:
                            callback()

        self.update_all()

    def generate_sequence_to_anchor(self):
        if not len(self.current_sequence):
            print("Cannot run anchor sequence without source node!")
            return
        start = self.current_sequence[-1]
        goal = self.anchor
        try:
            seq = nx.dijkstra_path(self.graph, start, goal, weight='weight')
            self.current_sequence = self.current_sequence + seq[1:]
            callback = self.callbacks.get('label')
            if callback is not None:
                callback()
        except nx.NetworkXNoPath:
            print('No path found to anchor!')

    def generate_random_sequence(self, n=10):

        if not len(self.current_sequence):
            self.current_sequence = [random.choice(list(self.graph.nodes))]

        while len(self.current_sequence) < n:
            neighbors = list(self.graph[self.current_sequence[-1]])
            random.shuffle(neighbors)
            for next_node in neighbors:
                if next_node in self.current_sequence:
                    continue
                self.current_sequence.append(next_node)
                break
            else:
                # No non-visited neighbors, terminate sequence
                break
        if len(self.current_sequence) > 1:
            callback = self.callbacks.get('label')
            if callback is not None:
                callback()
            self.update_all()
        else:
            self.generate_random_sequence(n)

    def reset_sequence(self, clear_all=False):
        self.current_sequence = []
        self.current_edge_labels = []
        if clear_all:
            self.all_sequence_labels = []
        self.update_all()

    @property
    def can_commit(self):
        return len(self.current_sequence) - 1 == len(self.current_edge_labels)

    def label_sequence(self, tag):

        if tag in '[]':
            val = False if tag == '[' else True
            if self.current_edge_labels:
                self.current_edge_labels[-1] = (self.current_edge_labels[-1][0], val)

        elif tag in '123450':
            label_map = {
                '1': 'b', '2': 'l', '3': 's', '4': 't', '5': 'o', '0': 'f'
            }
            val = label_map[tag]
            truth = val != 'f'
            if not self.can_commit:
                to_append = (val, truth)
                self.current_edge_labels.append(to_append)
        else:
            raise ValueError('Invalid tag value {}'.format(tag))

        self.update_all()

    def commit(self):
        if self.can_commit:
            self.all_sequence_labels.append([self.current_sequence, self.current_edge_labels])
            self.reset_sequence()
            self.dialog.setText('Sequence committed!\nCurrent num: {}'.format(len(self.all_sequence_labels)))
            print(self.all_sequence_labels[-1])
        else:
            print('Not ready to commit yet!')

    def set_select_trunk_message(self):
        self.dialog.setText('Please select a trunk node as an anchor.')

    def set_default_message(self):
        msg = 'Please select nodes to define a sequence.\n'
        msg += 'Hit D or double-click on a terminal node to start the labeling process.\n'
        msg += 'Hit E to erase the current process.\n'
        msg += 'Hit Backspace to undo one of your selections.\n'
        msg += 'Hit A to save all current selections to a file.'
        self.dialog.setText(msg)

    def set_labeling_message(self):
        msg = 'Please label the ground truths.\n'
        msg += '1 = side branch, 2 = leader, 3 = support, 4 = trunk, 5 = other, 0 = false connection.\n'
        msg += 'Hit [ to label the previous branch as a truth, and ] for a false.\n'
        msg += "When done, hit C to commit."
        self.dialog.setText(msg)





class TreeLabeler(QMainWindow):
    def __init__(self, template_root=None, data_root=None):
        QMainWindow.__init__(self)
        self.setWindowTitle('Tree labeller')

        self.template_root = template_root or '/home/main/data/fake_2d_trees/templates'
        self.data_root = data_root or '/home/main/data/fake_2d_trees/data'

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout()
        widget.setLayout(main_layout)

        # Add interface
        self.canvas = TreeLabelerCanvas(callbacks={'label': self.enable_labeling_mode})
        main_layout.addWidget(self.canvas)

        # Add interface defining parameters
        param_layout = QGridLayout()
        self.min_points = LabelAndText('Min Points', '3000')
        self.max_points = LabelAndText('Max Points', '20000')
        self.min_height = LabelAndText('Min Height', '0.8')
        self.max_height = LabelAndText('Max Height', '1.2')
        self.noise = LabelAndText('Noise', '0.02')
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load)
        self.load_text = QLineEdit()
        random_button = QPushButton("Random")
        random_button.clicked.connect(lambda: self.canvas.generate_random_sequence(10))
        resample_button = QPushButton("Resample")
        resample_button.clicked.connect(self.resample)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save)

        layout_list = [
            [self.min_points, self.min_height, random_button, save_button],
            [self.max_points, self.max_height, self.noise, resample_button],
            [None, None, self.load_text, load_button]
        ]
        for row, row_items in enumerate(layout_list):
            for col, widget_to_add in enumerate(row_items):
                if widget_to_add is None:
                    continue
                param_layout.addWidget(widget_to_add, row, col)

        main_layout.addLayout(param_layout)

        # State stuff
        self.pc = None
        self.current_img = None
        self.labeling_mode = False
        self.count = len(os.listdir(self.data_root))

    def resample(self):
        num_points = np.random.randint(self.min_points.value(), self.max_points.value())
        height = np.random.uniform(self.min_height.value(), self.max_height.value())

        xys = sample_xys_from_image(self.current_img, num_points) / np.max(self.current_img.shape)
        # Center X, Y value starts at 0
        x_max = xys[:,0].max()
        x_min = xys[:,0].min()
        xys = xys - np.array([(x_max + x_min) / 2, xys[:,1].min()])
        xys = xys * height

        xys += np.random.normal(0, self.noise.value(), xys.shape)

        self.pc = xys

        cover = get_open_cover(self.pc, 0.10)
        graph = create_graph_from_cover(self.pc, cover, 0.25)

        self.canvas.load_pointcloud(xys, graph)



    def load(self):
        to_load = self.load_text.text().strip()
        if not to_load:
            files = os.listdir(self.template_root)
            to_load = random.choice(files)
        img = np.array(imageio.imread(os.path.join(self.template_root, to_load)))
        img = ((img[:,:,3] > 0) * (img[:,:,0:3].mean(axis=2))).astype(np.int)
        self.current_img = img
        self.resample()

    def save(self):
        # Save: Graph, pointcloud, all labels
        output = {
            'points': self.pc,
            'graph': self.canvas.graph,
            'all_labels': self.canvas.all_sequence_labels
        }
        while True:
            new_file = os.path.join(self.data_root, '{}.tree'.format(self.count))
            if os.path.exists(new_file):
                self.count += 1
            else:
                with open(new_file, 'wb') as fh:
                    pickle.dump(output, fh)
                print('Output saved to: {}'.format(new_file))
                print(output['all_labels'])
                break

    def enable_labeling_mode(self):
        self.labeling_mode = True
        self.canvas.set_labeling_message()

    def keyPressEvent(self, event):
        pressed = event.key()
        if pressed == QtCore.Qt.Key_D:
            if len(self.canvas.current_sequence) > 1:
                self.enable_labeling_mode()
        elif pressed == QtCore.Qt.Key_Backspace:
            if self.labeling_mode:
                self.canvas.current_edge_labels = self.canvas.current_edge_labels[:-1]
            else:
                self.canvas.current_sequence = self.canvas.current_sequence[:-1]
            self.canvas.update_all()
        elif pressed == QtCore.Qt.Key_E:
            self.labeling_mode = False
            self.canvas.reset_sequence()
            self.canvas.set_default_message()
        elif pressed == QtCore.Qt.Key_C:
            self.canvas.commit()
            self.labeling_mode = False
        elif pressed == QtCore.Qt.Key_R and not self.labeling_mode:
            self.canvas.generate_random_sequence(10)

        elif self.labeling_mode:
            val = chr(pressed).lower()
            if val in '123450[]':
                self.canvas.label_sequence(val)


if __name__ == '__main__':
    app = QApplication([])

    gui = TreeLabeler()

    gui.show()

    app.exec_()
