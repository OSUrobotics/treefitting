#!/usr/bin/env python3

# Read in the depth image and the bspline curve in 2D.
# Assumes one fitted 2d curve
#   Extract depth values
#     - Average values along spline cross section
#     - Returns 3D curve
# Also need camera parameters (field of view)

import numpy as np
import cv2
import json
from os.path import exists
from utils.file_names import FileNames
from tree_geometry.b_spline_cyl import BSplineCyl
from tree_geometry.b_spline_cyl_3d import BSplineCyl3d
from utils.camera_projections import frustrum_matrix, from_image_to_box, set_default_params, from_box_to_image
from fit_routines.bspline_fit_params import BSplineFitParams
from draw_routines.b_spline_image import BSplineCylImage
from fit_routines.b_spline_curve_fit import BSplineCurveFit
from tree_geometry.point_lists import PointList


class FitBSplineCyl3dDepth:
    def __init__(self,
                 crv_2d_image: BSplineCyl,
                 crv_2d_depth: BSplineCyl,
                 depth_data: np.ndarray,
                 camera_params: dict,
                 fit_params: BSplineFitParams = None) -> None:
        """ Read in the depth data (data preferred), grab the depth data under the 2d curve, then promote to 3d
        @param crv_2d_image: curve in image
        @param crv_2d_depth: curve in depth image (presumes depth image has a different size than image)
        @param depth_data: Depth data as a .csv file (assumes depth image and csv file same size/aspect ratio)
        @param camera_params: Camera params; should have aspect ratio/field of view filled in
        @param fit_params: Parameters for filtering the depth image - how finely to sample along the edge and how much to believe edge
           perc_width_depth - percent of width to use, should be 0.1 to 0.85
           perc_along_depth - take median of pixels from a perc of curve, should be 0.1 to 0.3
           camera_width_angle - angle in degrees, 45 for intel d45, etc"""

        # save the data
        self.crv_2d = crv_2d_image
        self.crv_2d_depth = crv_2d_depth
        self.depth_data_orig = depth_data
        self.camera_params = camera_params
        self.fit_params = fit_params

        if self.fit_params is None:
            self.fit_params = BSplineFitParams()

        if self.camera_params is None:
            set_default_params(self.camera_params)
        if "depth image width" in self.camera_params:
            assert self.camera_params["depth_image_width"] == self.depth_data_orig.shape[0]
            assert self.camera_params["depth_image_height"] == self.depth_data_orig.shape[1]
        else:
            self.camera_params["depth_image_width"] = self.depth_data_orig.shape[0]
            self.camera_params["depth_image_height"] = self.depth_data_orig.shape[1]

        self.d_min = np.min(np.min(self.depth_data_orig))
        self.d_max = np.max(np.max(self.depth_data_orig))
        self.zero_value = self.depth_data_orig[0, 0]
        n_count = np.count_nonzero(self.depth_data_orig == self.zero_value)
        print(f"zero value {self.depth_data_orig[0, 0]}, perc {n_count/self.depth_data_orig.size} d min {self.d_min}, d max {self.d_max}")
        self.depth_image = 255 - self.depth_data_orig
        for clip in [(2, 20), (20, 100), (100, 225), (225, 240), (240, 254), (1, 254)]:
            im_mask_orig_rgb = np.zeros(self.depth_image.shape, dtype=np.uint8)
            im_mask_orig_rgb[np.logical_and(self.depth_image > clip[0], self.depth_image < clip[1])] = 250
            cv2.imwrite(f"fit3d_depth_orig_{clip[0]}_{clip[1]}.png", im_mask_orig_rgb)
        #self.depth_image = np.uint8(255 * (self.depth_data - self.d_min) / (self.d_max - self.d_min))

    def _full_depth_stats(self):
        """ Get the best pixel offset (if any) for each point/pixel along the edge
        @param depth_data - the depth data as a numpy array
        @param crv_2d - the 2d curve
        @param params - parameters for conversion
        @return t, stats for depth, spaced n apart"""

        ret_stats = {"step_size": 5,
                     "perc_fuzzy": 0.2}

        # First going to render the depth curve into an image
        im_cyl = BSplineCylImage(self.crv_2d_depth.points(), self.crv_2d_depth.degree_name(), self.crv_2d_depth.radii_crv.points())
        # Make a mask image that is 255 where the curve is
        im_mask = np.zeros(self.depth_image.shape)
        im_cyl.make_mask_image(im_mask, step_size=ret_stats["step_size"], perc_fuzzy=ret_stats["perc_fuzzy"])

        cv2.imwrite("fit3d_mask.png", im_mask)

        # TODO clip to rectangle that contains curve
        # All depth values under mask, in mask form
        # Get just the depth values under the mask
        depth_unsorted = self.depth_image[im_mask > 0]
        n_total_pix = im_mask.size
        depth_sort = np.sort(depth_unsorted)
        depth_sort = depth_sort[depth_sort < 254]
        # Get rid of background pixels
        ret_stats["depth_sort"] = depth_sort
        ret_stats["total_non_zero"] = depth_sort.size
        ret_stats["min_value"] = np.min(depth_sort)
        ret_stats["max_value"] = np.max(depth_sort)
        ret_stats["median_value"] = np.median(depth_sort)

        for clip in [(1, 10), (10, 100), (100, 225), (225, 240), (240, 254), (0, 254)]:
            im_mask_orig_rgb = np.zeros(self.depth_image.shape, dtype=np.uint8)
            im_mask_orig_rgb[np.logical_and(self.depth_image > clip[0], self.depth_image < clip[1])] = 250
            im_mask_orig_rgb = im_mask_orig_rgb * im_mask
            cv2.imwrite(f"fit3d_depth_mask_{clip[0]}_{clip[1]}.png", im_mask_orig_rgb)

        # Assuming some pixels are in the background, should have a sharp "jump" to pixels in the foreground
        # Assume most of the foreground pixels (close) are correct
        """
        ret_stats["best_split"] = []
        depth_clip = 0.8 * np.median(depth_sort) + 0.2 * np.max(depth_sort)
        print(f"Median {np.median(depth_sort)} min {np.min(depth_sort)} max {np.max(depth_sort)}")
        print(f" 1/3 {depth_sort[int(depth_sort.size / 3.0)]}  2/3 {depth_sort[int(2.0 * depth_sort.size / 3.0)]}")
        for w in [50, 70, 100, 120]:
            score = (depth_sort.size // 2, 0, w)
            for k in range(w+1, depth_sort.size - (w+1)):
                if depth_sort[k+w] - depth_sort[k-w] > score[1]:
                    score = (k, depth_sort[k+w] - depth_sort[k-w], w)
            ret_stats["best_split"].append(score)
            depth_clip += depth_sort[int(score[0] + w)]
            print(f"Score {score} {depth_sort[int(score[0] + w)]}")
        depth_clip /= 5.0
        ret_stats["depth_clip"] = depth_clip
        """
        ts = np.linspace(0, self.crv_2d_depth.max_t(), self.crv_2d_depth.n_points() * 2)
        rects, ts_rects = self.crv_2d_depth.interior_rects(ts, 1.0)

        n_perc_min = int(0.1 * depth_sort.size)
        n_perc_max = int(0.9 * depth_sort.size)
        ret_stats["clip_min"] = depth_sort[n_perc_min]
        ret_stats["clip_max"] = depth_sort[n_perc_max]
        im_mask_clip = np.zeros(im_mask.shape, dtype=np.uint8)
        b_sel = np.logical_and(self.depth_image >= ret_stats["clip_min"], self.depth_image <= ret_stats["clip_max"])
        im_mask_clip[b_sel] = 255
        cv2.imwrite("fit3d_depth_mask_clip_bw.png", im_mask_clip)
        im_mask_clip[b_sel] = self.depth_image[b_sel]
        cv2.imwrite("fit3d_depth_mask_clip_grey.png", im_mask_clip)
        ret_stats["ts"] = []
        ret_stats["depth_at_center"] = []
        ret_stats["z_at_center"] = []
        ret_stats["r_at_depth"] = []
        ret_stats["radius_3d"] = []
        for t, r in zip(ts_rects, rects):
            min_x = int(np.min(r[:, 0]))
            max_x = int(np.max(r[:, 0]))
            min_y = int(np.min(r[:, 1]))
            max_y = int(np.max(r[:, 1]))
            if min_x < 0 or min_y < 0 or max_x > im_mask_clip.shape[1] or max_y > im_mask_clip.shape[0]:
                continue

            seg_depth = im_mask_clip[min_y:max_y, min_x:max_x]

            max_depth = np.max(seg_depth)
            depth_at_center = 0.0
            # We have at least one pixel > 0
            if max_depth > 0.0:
                depth_at_center_uint = np.median(seg_depth[seg_depth > 0])
                depth_at_center = camera_params["min_depth"] + (255.0 - depth_at_center_uint) / (camera_params["max_depth"] - camera_params["min_depth"])
                print(f"{depth_at_center_uint} ", end="")

            if depth_at_center > 0.0:
                rad_2d = self.crv_2d.radius(t)
                ang_subtend_degrees = self.camera_params["camera_width_angle"] * (2 * rad_2d) / self.camera_params["depth_image_width"]
                ang_subtend_radians = np.pi * ang_subtend_degrees / 180.0
                radius_3d = 0.5 * depth_at_center * np.tan(ang_subtend_radians)
                z_at_center = depth_at_center - radius_3d

                ret_stats["ts"].append(t)
                ret_stats["depth_at_center"].append(depth_at_center)
                ret_stats["r_at_depth"].append(rad_2d)
                ret_stats["z_at_center"].append(z_at_center)
                ret_stats["radius_3d"].append(radius_3d)

        print("\n")
        return ret_stats

    def _curve_from_stats(self, stats_depth, rgb_image:np.array=None):
        """
        From the raw stats, create a set of evenly-spaced t values
        @param stats_depth: The stats from full_depth_stats
        @return: 3d curve
        """

        mat = frustrum_matrix(self.camera_params)
        mat_inv = np.linalg.inv(mat)

        image_width = self.camera_params["image_size"][0]
        image_height = self.camera_params["image_size"][1]

        cam_width_ang_half = 0.5 * self.camera_params['camera_width_angle']
        cam_height_ang_half = (0.5 * self.camera_params['camera_width_angle'] *
                               self.camera_params['depth_image_height'] / self.camera_params['depth_image_width'])
        print(f"cam x ang {cam_width_ang_half * 2} cam y ang {cam_height_ang_half * 2} {image_width}, {image_height}")

        pt_z_origin = np.ones(shape=(4,))
        pt_post_proj = np.ones(shape=(4,))
        pt_z_origin[0] = 0.0
        pt_z_origin[1] = 0.0

        ts_pts = np.linspace(0, self.crv_2d.max_t(), self.crv_2d.n_points() * 4)
        pts = []
        radii = []
        for t in ts_pts:
            # Search for best-match t
            z_at_center = 0.0
            radius = 1.0
            if t < stats_depth["ts"][0]:
                z_at_center = stats_depth["z_at_center"][0]
                radius = stats_depth["radius_3d"][0]
            elif t > stats_depth["ts"][-1]:
                z_at_center = stats_depth["z_at_center"][-1]
                radius = stats_depth["radius_3d"][-1]
            else:
                for indx in range(1, len(stats_depth["ts"])):
                    if stats_depth["ts"][indx-1] <= t <= stats_depth["ts"][indx]:
                        z_at_center = stats_depth["z_at_center"][indx]
                        radius = stats_depth["radius_3d"][indx]

            # Project the point 0, 0, d into the frustum box to get w, d'
            pt_z_origin[2] = -z_at_center
            pt_z_proj = mat @ pt_z_origin

            # Convert point in image coordinates to frustum box post project
            pt_crv_2d = self.crv_2d.eval_crv(t)
            pt_proj_box = from_image_to_box(self.camera_params, pt_crv_2d)

            # Now use the w to get the point pre-divide
            pt_post_proj[0] = pt_proj_box[0] * pt_z_proj[3]
            pt_post_proj[1] = pt_proj_box[1] * pt_z_proj[3]
            pt_post_proj[2] = pt_z_proj[2]
            pt_post_proj[3] = pt_z_proj[3]

            # Now undo the projection
            pt_in_space = mat_inv @ pt_post_proj

            # Check result
            pts.append(pt_in_space[0:3])
            radii.append(radius)

            if rgb_image:
                pt_back_image = from_box_to_image(pt_in_space)

        return ts_pts, pts, radii

    def fit_curve(self):
        """ Fit the curve to the depth image. Returns the curve"""
        stats_depth = self._full_depth_stats()
        ts, pts, radii = self._curve_from_stats(stats_depth)

        # Now do the actual fit
        pt_list = PointList(pts)
        pts_start_3d = []

        for indx in range(0, self.crv_2d.n_points()):
            indx_3d = int((pt_list.n_points()-1) * indx / (self.crv_2d.n_points() - 1.0))
            pt = pt_list.points()[indx_3d]
            pts_start_3d.append(pt)

        crv_fit = BSplineCurveFit(pt_list, self.fit_params)
        crv_3d = BSplineCyl3d(pts_start_3d, "cubic", radii)
        curve_fit, _ = crv_fit.initial_fit(crv_3d, pt_list)

        #curve_fit, _, _ = crv_fit.fit_project_fit(crv_fit.crv_start, pt_list, self.fit_params)

        # TODO do this as an actual fit
        return BSplineCyl3d(ctrl_pts=curve_fit.points(), degree=self.fit_params["degree"], radii=radii)


if __name__ == '__main__':
    import argparse
    from utils.file_names_sub_dirs import FileNamesSubDirs
    from utils.video_annotation_data import VideoAnnotationData

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="PycharmProjects/data/bush_1_east/", type=str, help="where to grab images from")
    parser.add_argument('--annot', default="video_annot.json", type=str, help="which video annotation to use")
    parser.add_argument('--key_frame', default=-1, type=int, help="which key frame, set to -1 for all")
    parser.add_argument('--mask', default=-1, type=int, help="which mask, set to -1 for all")
    parser.add_argument('--mask_id', default=-1, type=int, help="which curve, set to -1 for all")
    parser.add_argument('--camera', default="azure", type=str, help="Camera, one of azure, intel TODO")


    args = parser.parse_args()

    path_start = FileNamesSubDirs.get_path()
    path_full = path_start + args.path + args.annot

    with open(path_full, "r") as f:
        my_dict = json.load(f)
        va = VideoAnnotationData.read_json(my_dict)

    start_kf = args.key_frame
    end_kf = start_kf + 1
    if args.key_frame == -1:
        start_kf = 0
        end_kf = va.n_keyframes()

    start_mask = args.mask
    end_mask = start_mask + 1
    if args.mask == -1:
        start_mask = 0
        end_mask = va.n_masks()

    fit_params = BSplineFitParams()
    camera_params = {}
    set_default_params(camera_params)
    if args.camera == "azure":
        camera_params["image_size"] = (1920, 1080)
        camera_params["depth_image_width"] = 640
        camera_params["depth_image_height"] = 576
        camera_params["camera_width_angle"] = 90
        camera_params["camera_height_angle"] = 59
        camera_params["min_depth"] = 0.5   # meters
        camera_params["max_depth"] = 5.46  # meters

    for kf_indx in range(start_kf, end_kf):
        kf = va.keyframes[kf_indx]

        # The gray scale image
        #  254 is the "no data" value, all data is in the z channel
        depth_image = cv2.imread(va.get_depth_data_name((0, kf_indx, 0, 0)))
        depth_data = depth_image[:, :, 2]

        # depth_data = np.fromfile(va.get_depth_data_name((0, kf_indx, 0, 0)))
        # depth_data.reshape((camera_params["depth_image_width"], camera_params["depth_image_height"]))
        print(f"min {np.min(depth_image)}, {np.min(depth_data)} max {np.max(depth_image)}, {np.max(depth_data)}")
        for m_indx in range(start_mask, end_mask):
            start_id = args.mask_id
            end_id = start_id + 1
            if start_id == -1:
                start_id = 0
                end_id = va.n_mask_ids(0, kf_indx, m_indx)
            for m_id_indx in range(start_id, end_id):
                crv_2d = kf.get_bsplinecyl(m_indx, m_id_indx)
                crv_2d_depth = kf.get_bsplinecyl_in_depth_image(m_indx, m_id_indx)

                # Actual fit
                crv_fit_3d = FitBSplineCyl3dDepth(crv_2d, crv_2d_depth, depth_data=depth_data, camera_params=camera_params, fit_params=fit_params)
                crv_fit = crv_fit_3d.fit_curve()

                crv_name = va.get_mask_name((0, kf_indx, m_indx, m_id_indx), b_calculate_path=True, b_add_tag=False) + "_crv.json"
                mesh_name = va.get_mask_name((0, kf_indx, m_indx, m_id_indx), b_calculate_path=True, b_add_tag=False) + "_crv.obj"
                with open(crv_name, "w") as f:
                    json.dump(crv_fit.write_json(), f, indent=2)
                crv_fit.write_mesh(mesh_name)
