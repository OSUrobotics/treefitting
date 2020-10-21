import numpy as np
from scipy.linalg import svd
import matplotlib.path as mpath

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
