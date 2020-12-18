import numpy as np
from scipy.linalg import svd
import matplotlib.path as mpath
import random
try:
    import pymesh
    pymesh.load_mesh
except (ImportError, AttributeError):
    pymesh = None

class PriorityQueue:

    def __init__(self, minimize=True):
        self.items = []
        self.levels = []
        if minimize:
            self.select_func = min
        else:
            self.select_func = max
            
    def add(self, item, level):
        self.items.append(item)
        self.levels.append(level)

    def pop(self):
        item_idx = self.levels.index(self.select_func(self.levels))
        item = self.items[item_idx]
        level = self.levels[item_idx]
        del self.items[item_idx]
        del self.levels[item_idx]

        return item, level

    def __bool__(self):
        return len(self.items) > 0

def points_to_grid_svd(pts, start, end, normalize=True, output_extra=False):
    main_axis = start - end
    main_axis = main_axis / np.linalg.norm(main_axis)
    projected = project_points_onto_normal(start, main_axis, pts)
    secondary_axis = svd(projected - projected.mean())[2][0]

    all_pts = project_point_onto_plane((start + end) / 2, main_axis, secondary_axis, pts)
    all_pts = all_pts / (np.linalg.norm(start - end) / 2)  # Makes endpoints at (-1, 0), (1, 0)
    bounds_x = np.linspace(-1.5, 1.5, 32 + 1)
    bounds_y = np.linspace(-0.75, 0.75, 16 + 1)

    grid = np.histogram2d(all_pts[:, 0], all_pts[:, 1], bins=[bounds_x, bounds_y])[0]
    if normalize:
        grid = grid / grid.max()
    if not output_extra:
        return grid
    else:

        plane_normal = np.cross(main_axis, secondary_axis)
        proj_3d = project_points_onto_normal((start + end) / 2, plane_normal, pts)

        return {
            'grid': grid,
            'start': start,
            'end': end,
            'first_projection': projected,
            'y_axis': secondary_axis,
            'second_projection': proj_3d,
            'points': pts,
        }


def project_points_onto_normal(plane_origin, plane_normal, points):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    dist = np.dot(points - plane_origin, plane_normal)
    proj_3d = (points - np.reshape(dist, (-1, 1)).dot(np.reshape(plane_normal, (1, -1)))) - plane_origin

    return proj_3d

def project_point_onto_plane(plane_origin, x_axis, y_axis, points):

    plane_normal = np.cross(x_axis, y_axis)
    proj_3d = project_points_onto_normal(plane_origin, plane_normal, points)

    proj_2d = np.zeros((points.shape[0], 2))
    proj_2d[:, 0] = np.dot(proj_3d, x_axis)
    proj_2d[:, 1] = np.dot(proj_3d, y_axis)
    return proj_2d

def rasterize_3d_points(pts, bounds=None, size=128):
    zplane_pts = pts[:, :2]
    if bounds is None:
        x_max, y_max = zplane_pts.max(axis=0)
        x_min, y_min = zplane_pts.min(axis=0)
        scale = max(x_max - x_min, y_max - y_min)
        x_cen, y_cen = (x_max + x_min) / 2, (y_max + y_min) / 2

        bounds_x = np.linspace(x_cen - scale / 2, x_cen + scale / 2, size+1)
        bounds_y = np.linspace(y_cen - scale / 2, y_cen + scale / 2, size+1)
        bounds = [bounds_x, bounds_y]

    raster = np.histogram2d(zplane_pts[:, 0], zplane_pts[:, 1], bins=bounds)[0]
    raster = raster / np.max(raster)
    return raster, bounds

def expand_node_subset(nodes, graph):
    final = set()
    for node in nodes:
        final.update(graph[node])
    return final

def edges(points):
    return zip(points[:-1], points[1:])

def convert_points_to_ndc_space(pts, modelview_matrix, proj_matrix):
    all_pts_homog = np.ones((len(pts), 4))
    all_pts_homog[:, :3] = pts

    clip_space_pts = proj_matrix @ modelview_matrix @ all_pts_homog.T
    ndc = (clip_space_pts / clip_space_pts[3]).T[:, :3]

    return ndc

def convert_ndc_to_gl_viewport(ndc, viewport_info):

    x, y, w, h = viewport_info

    viewport_xy = ndc[:, :2] * np.array([w / 2, h / 2]) + np.array([x + w / 2, y + h / 2])
    # Need to flip y coordinate due to pixel coordinates being defined from the top-left
    viewport_xy[:, 1] = h - viewport_xy[:, 1]

    return viewport_xy

def convert_points_to_gl_viewport_space(pts, modelview_matrix, proj_matrix, viewport_info):
    ndc = convert_points_to_ndc_space(pts, modelview_matrix, proj_matrix)
    return convert_ndc_to_gl_viewport(ndc, viewport_info)


def convert_gl_viewport_space_to_ndc_2d(viewport_xy, viewport_info):
    x, y, w, h = viewport_info
    viewport_xy = viewport_xy.copy()
    viewport_xy[:, 1] = h - viewport_xy[:, 1]
    ndc = (viewport_xy - np.array([x + w / 2, y + h / 2])) / np.array([w / 2, h / 2])
    return ndc


def convert_pyqt_to_gl_viewport_space(pixel, pyqt_wh, viewport_info):
    click_x, click_y = pixel
    pyqt_w, pyqt_h = pyqt_wh
    x, y, w, h = viewport_info

    click_x_vp = click_x * w / pyqt_w
    click_y_vp = click_y * h / pyqt_h

    click_vp = np.array([click_x_vp, click_y_vp])
    return click_vp


def compute_filter(pc, bounds, polygon_filters):
    x_low = pc[:, 0] >= bounds['x'][0]
    x_hi = pc[:, 0] <= bounds['x'][1]
    y_low = pc[:, 1] >= bounds['y'][0]
    y_hi = pc[:, 1] <= bounds['y'][1]
    z_low = pc[:, 2] >= bounds['z'][0]
    z_hi = pc[:, 2] <= bounds['z'][1]

    axis_filter = x_low & x_hi & y_low & y_hi & z_low & z_hi

    for polygon_ndc, modelview_matrix, proj_matrix in polygon_filters:
        polygon_path = mpath.Path(polygon_ndc)
        pts_ndc = convert_points_to_ndc_space(pc, modelview_matrix, proj_matrix)[:, :2]

        polygon_filter = ~polygon_path.contains_points(pts_ndc)
        axis_filter = axis_filter & polygon_filter

    return axis_filter

def random_looper(elements):
    assert len(elements)
    elements = elements[:]
    random.shuffle(elements)
    pos = 0
    while True:
        yield elements[pos]
        pos = (pos + 1) % len(elements)


# All ply reading utils courtesy of:
# https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/io/ply.py

import sys
import pandas as pd
from collections import defaultdict

sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


def read_ply(filename):

    """ Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path to the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

    if pymesh is not None:
        return pymesh.load_mesh(filename).vertices

    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start whith the word ply')
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        has_texture = False
        comments = []
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:

                    if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
                        mesh_names = ["n_points", "v1", "v2", "v3"]
                    else:
                        has_texture = True
                        mesh_names = ["n_coords"] + ["v1_u", "v1_v", "v2_u", "v2_v", "v3_u", "v3_v"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, len(mesh_names)):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append(
                            (line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]]))

            elif b'comment' in line:
                line = line.split(b" ", 1)
                comment = line[1].decode().rstrip()
                comments.append(comment)

            count += 1

        # for bin
        end_header = ply.tell()

    if fmt == 'ascii':
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]

        pts = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)

        for n, col in enumerate(pts.columns):
            pts[col] = pts[col].astype(
                dtypes["vertex"][n][1])
        pts = pts.values
        if not np.abs(pts[-1]).sum():
            pts = pts[:-1]

        return pts

    else:
        raise NotImplementedError("Non-ASCII PLY file!")


def write_ply(filename, points=None, mesh=None, as_text=False, comments=None):
    """

    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    comments: list of string

    Returns
    -------
    boolean
        True if no problems

    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if comments:
            for comment in comments:
                header.append('comment ' + comment)

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element