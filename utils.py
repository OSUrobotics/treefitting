import numpy as np
from scipy.linalg import svd

def points_to_grid_svd(pts, start, end):
    main_axis = start - end
    main_axis = main_axis / np.linalg.norm(main_axis)
    projected = project_points_onto_normal(start, main_axis, pts)
    secondary_axis = svd(projected - projected.mean())[2][0]

    all_pts = project_point_onto_plane((start + end) / 2, main_axis, secondary_axis, pts)
    all_pts = all_pts / (np.linalg.norm(start - end) / 2)  # Makes endpoints at (-1, 0), (1, 0)
    bounds_x = np.linspace(-1.5, 1.5, 32 + 1)
    bounds_y = np.linspace(-0.75, 0.75, 16 + 1)

    grid = np.histogram2d(all_pts[:, 0], all_pts[:, 1], bins=[bounds_x, bounds_y])[0]
    return grid


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