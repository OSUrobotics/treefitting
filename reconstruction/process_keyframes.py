#!/usr/bin/env python3

# From manually marked sketch curves on several key frames (plus some background points) do the following
#   Produce a consistant set of 3D points (from the curves) for each cane/branch
#   Calculate a 3D camera for each key frame
#
# Input is VideoAnnotationData file with 2 (or more) keyframes with sketched 2D points for canes/branches, plus
#   some background points that are the same across all the images
# Assumptions:
#  Version 1: Just use clicked points
#   Each marked keyframe has the same number of curves/canes with the same labels
#   Each marked keyframe has the same background points marked in the same order (can do some cleanup)
#   If the keyframe has curve points marked then it also has background, and vice-versa
#   Each cane/branch has points clicked in the same order, with the same number of points
#  Version 2: fit 2D curves and then calculate the transforms
#   Same as above, but can have more or less number of points clicked on the branch, just need to start/stop end at same place
# Debug images
#  Outputs (for all keyframe images) an image with
#    The original 2D points that were clicked
#    The 3D points projected into that image
#    If camera intrinsics exist, use that to project the 3D points to the image


import os.path
from utils.video_annotation_data import VideoAnnotationData
from utils.camera_projections import CameraProjections
from utils.file_names_sub_dirs import FileNamesSubDirs
from utils.keyframe_data import KeyFrameData
from draw_routines.image_draw_geom_utils import draw_cross
import numpy as np
import cv2


class PointTrackerKeyFrames():
    def __init__(self, va_data: VideoAnnotationData, rgb_camera: CameraProjections):

        # keep the input data
        self.va_data = va_data
        self.rgb_camera = rgb_camera

        # State variables
        #   Using N as the number of camera frames/images, and T as number of points
        self.image_size = rgb_camera.image_size
        self.pose_matrices = []
        self.pt_locations_2d = []
        self.crv_pt_locations_2d = []
        self.pt_locations_3d = []
        self.crv_pt_locations_3d = []

    def _check_consistant(self, vec, pts2d_a, pts2d_b):
        """ Count the number of points where the vec is correct (within a few pixels)
        @param vec: vector from a to b
        @param pts2d_a: a list of points
        @param pts2d_b: a list of points
        @return: number of correct points, best match for each"""
        count = 0
        match = []
        est_error = 100 + 0.2 * abs(vec[0] + abs(vec[1]))
        for pa in pts2d_a:
            best_d = 10000
            best_i = -1
            for indx, pb in enumerate(pts2d_b):
                vec_check = [pb[0] - pa[0], pb[1] - pa[1]]
                pixs_wrong = abs(vec_check[0] - vec[0]) + abs(vec_check[1] - vec[1])
                if pixs_wrong < best_d:
                    best_d = pixs_wrong
                    best_i = indx
            if best_d < est_error:
                match.append(best_i)
                count += 1
            else:
                match.append(-1)

        if count != len(pts2d_a):
            print(f"Warning, no match found {vec} {count}, {match}")
        return count, match

    def _consistent_vec(self, pts2d_a, pts2d_b, b_is_horizontal=False, b_is_vertical=False):
        """ Stupid version of ransac - get a vec between two points in a, b that is correct for most points
        @param pts2d_a: a list of points
        @param pts2d_b: a list of points
        @param b_is_horizontal: Force the vec to be horizontal
        @param b_is_vertical: Force the vec to be vertical
        @return: vec between pts2d_a and pts2d_b, best match for each point in pts2d_a"""
        best_vec = [0, 0]
        best_match = -1
        best_count = -1
        for pa in pts2d_a:
            for pb in pts2d_b:
                vec = [pb[0] - pa[0], pb[1] - pa[1]]
                if b_is_horizontal:
                    vec[1] = 0
                if b_is_vertical:
                    vec[0] = 0

                count, match = self._check_consistant(vec, pts2d_a, pts2d_b)
                if count > 0.7 * len(pts2d_a):
                    return vec, match, True

                if count > best_count:
                    best_vec = vec.copy()
                    best_match = match.copy()
                    best_count = count
        return best_vec, best_match, False

    def collect_background_points(self, b_is_horizontal=False, b_is_vertical=False):
        """ This is just the background points - fill in any missing points by interpolating locations
            Also tries to match points if they're in the wrong order
            Assumes first frame is correct
            @param b_is_horizontal - set True if you know the motion is left-right
            @param b_is_vertical - set True if you know the motion is up-down
            @return pt_locations_2d N x T list of points, vec from frame to frame, valid_keyframes"""

        # The 2D points we're collecting; an N-sized list by a T by 2d point
        self.pt_locations_2d = []
        # keep track of which key frames had background points
        valid_keyframes = []
        # The 2D transform from the first marked frame to each valid keyframe
        vecs = []
        # For each keyframe...
        for kf in self.va_data.keyframes:
            if len(kf.pts_2d_of_start) == 0:
                continue

            # If the keyframe has background points marked...
            self.pt_locations_2d.append([])
            if len(self.pt_locations_2d) == 1:
                # First keyframe with background points marked -make this the default ordering
                for pt in kf.pts_2d_of_start:
                    self.pt_locations_2d[-1].append([pt[0], pt[1]])
                valid_keyframes.append(kf)
                vecs.append([0, 0])
            else:
                # Poor person's ransac
                vec, match, valid = self._consistent_vec(self.pt_locations_2d[-2], kf.pts_2d_of_start, b_is_horizontal=b_is_horizontal, b_is_vertical=b_is_vertical)
                print(f" Match {kf.image_name} {match}")
                if not valid:
                    continue
                for indx, best_match in enumerate(match):
                    if best_match == -1:
                        # No match, so use vec
                        pt_new = [self.pt_locations_2d[0][indx][0] + vec[0],
                                   self.pt_locations_2d[0][indx][1] + vec[1]]
                        self.pt_locations_2d[-1].append(pt_new)
                    else:
                        pt_match = kf.pts_2d_of_start[best_match]
                        self.pt_locations_2d[-1].append([pt_match[0], pt_match[1]])
                valid_keyframes.append(kf)
                vecs.append(vec)
        return self.pt_locations_2d, vecs, valid_keyframes

    def collect_crv_points(self, vecs, valid_keyframes):
        """ This is the cane/curve points - fill in any missing points by propagating by vecs
            Also tries to match points if they're in the wrong order
            @param vecs - the vector to use if no match is found
            @param valid_keyframes - The keyframes to use
            @return pt_locations_2d N x T list of points"""
        # List that is size N containing one list for each curve, with each curve having some number of points
        self.crv_pt_locations_2d = []

        # How many mask ids to loop over
        n_mask_ids = len(valid_keyframes[0].sketch_curves)
        n_crvs = []
        last_valid_kf_with_curves = -1
        for kf_indx, (kf, vec_bkgrnd) in enumerate(zip(valid_keyframes, vecs)):
            if len(kf.sketch_curves) is not n_mask_ids:
                print(f"Warning, key frame {kf.image_name} has wrong number of mask ids")
                continue

            self.crv_pt_locations_2d.append([])
            kf_crv_pt_list = self.crv_pt_locations_2d[kf_indx]

            # Flattening out all mask id curve lists
            count = 0
            for mask_id_crv_list in kf.sketch_curves:
                for crv in mask_id_crv_list:
                    if last_valid_kf_with_curves == -1:
                        # First valid keyframe - just put crv points in
                        kf_crv_pt_list.append([])
                        for pt in crv.backbone_pts:
                            kf_crv_pt_list[-1].append([pt[0], pt[1]])
                    else:
                        # Match as best as possible
                        last_kf_crv =  self.crv_pt_locations_2d[last_valid_kf_with_curves][count]
                        vec, match, valid = self._consistent_vec(last_kf_crv, crv.backbone_pts)
                        print(f" Match {kf.image_name} {match}")
                        if len(last_kf_crv) != len(crv.backbone_pts):
                            print(f"Warning: Keyframe {kf.image_name}, number of curve points does not match {len(last_kf_crv)}, {len(crv.backbone_pts)}")
                        if not valid:
                            print(f"Warning: Key frame {kf.image_name} backbone curves {mask_id_crv_list} {count} not valid")
                        diff = abs(vec[0] - vec_bkgrnd[0]) + abs(vec[1] - vec_bkgrnd[1])
                        if diff > 20:
                            print(f"Warning: Key frame {kf.image_name} backbone curves have different vec {vec}, {vec_bkgrnd}")

                        kf_crv_pt_list.append([])
                        for indx, best_match in enumerate(match):
                            if best_match == -1:
                                # No match, so use vec
                                pt_new = [last_kf_crv[indx][0] + vec[0], last_kf_crv[1] + vec[1]]
                                kf_crv_pt_list[-1].append(pt_new)
                            else:
                                pt_match = crv.backbone_pts[best_match]
                                kf_crv_pt_list[-1].append([pt_match[0], pt_match[1]])
                    count += 1

            # Done with all curves for this keyframe
            if count > 0:
                if last_valid_kf_with_curves != -1:
                    if count is not len(self.crv_pt_locations_2d[last_valid_kf_with_curves]):
                        print(f"Warning, kf {kf.image_name} has wrong number of curves, got {count} expected {len(self.crv_pt_locations_2d[last_valid_kf_with_curves])}")
                last_valid_kf_with_curves = kf_indx

        return self.crv_pt_locations_2d

    def _combine_2d_pts(self):
        """ Flatten all the backbone and curve points into one list
        @return an N x (num background + num curves * num points per curves) list of 2d points"""
        ret_pts_2d = [[] * len(self.crv_pt_locations_2d)]
        for indx, pt_list in enumerate(self.crv_pt_locations_2d):
            ret_pts_2d[indx].append(pt_list)
            for crv_list in self.crv_pt_locations_2d[indx]:
                ret_pts_2d[indx].append(crv_list)
        return ret_pts_2d

    def full_triangulation(self):
        """"""
        """pts_3d = None
        if self.do_3d_point_estimation:
            ref_pose = np.linalg.inv(image_info[ref_idx]["pose"])
            camera_frame_tf_matrices = [(ref_pose @ info["pose"]) for info in image_info]
            triangulator = PointTriangulator(self.camera, min_points=self.min_points.value)
            pts_3d = triangulator.compute_3d_points(camera_frame_tf_matrices, trajs)
            reprojs = triangulator.get_reprojs(pts_3d, camera_frame_tf_matrices, trajs)
            error = np.linalg.norm(trajs - reprojs, axis=2)
            avg_error = error.mean(axis=1)
            max_error = error.max(axis=1)
            # print('Average pix error:\n')
            # print(', '.join('{:.3f}'.format(x) for x in avg_error))
            # print('Max pix error:\n')
            # print(', '.join('{:.3f}'.format(x) for x in max_error))

        trajs = np.transpose(trajs, (1, 0, 2))

        frame_id = image_info[ref_idx]["frame_id"]
        stamp = image_info[ref_idx]["stamp"].to_msg()

        response = Tracked3DPointResponse(header=Header(frame_id=frame_id, stamp=stamp))
        if pts_3d is not None:
            pc = create_cloud_xyz32(Header(frame_id=frame_id, stamp=stamp), points=pts_3d)
            self.pc_pub.publish(pc)
            for group, pts_and_errs in self.unflatten_tracked_points(zip(pts_3d, max_error), groups).items():
                points, errors = zip(*pts_and_errs)
                response.groups.append(
                    Tracked3DPointGroup(name=group, points=[Point(x=x, y=y, z=z) for x, y, z in points], errors=errors)
                )

        for group, points_2d in self.unflatten_tracked_points(trajs[ref_idx].astype(np.float), groups).items():
            response.groups_2d.append(TrackedPointGroup(name=group, points=[Point2D(x=x, y=y) for x, y in points_2d]))

        response.image = bridge.cv2_to_imgmsg(image_info[ref_idx]["image"])
        response.image.header.frame_id = image_info[ref_idx]["frame_id"]
        response.image.header.stamp = image_info[ref_idx]["stamp"].to_msg()

        return response, trajs, groups
        """
        pass

    def debug_tracking(self, valid_keyframes, pts_3d):
        import cv2
        from PIL import Image

        """
        final_imgs = []
        # trajs is point x frame x dim
        for i, img in enumerate(images):
            img = img.copy()
            pts = trajs[:, i]
            for j, pt in enumerate(pts):
                img = cv2.circle(img, pt.astype(int), 4, (0, 0, 255), -1)
                if reprojs is not None and not np.any(np.isnan(reprojs[j, i])):
                    img = cv2.circle(img, reprojs[j, i].astype(int), 4, (0, 255, 255), -1)

            final_imgs.append(img)
            if output is not None:
                stamp = self.get_clock().now().seconds_nanoseconds()[0]
                Image.fromarray(img).save(os.path.join(output, f"{stamp}_{i+1}.png"))
                print("output image")
        """
        pass

    def run_triangulation(self, pose_matrices, point_traj):
        """ Calculate the 3D location of a point given pose matrices for each camera
            and the 2D point location for each of the images (and an intrinsic matrix)
            Basically find x,y,z st proj * pose_n * [u,v]_n minimizes the distance between
            the projected point and the clicked location in the image
        pose_matrices: List of N 4x4 matrices
        trajs: N x 2 array of point trajectory for a single point in typical image XY format
        """

        world_to_camera = self.rgb_camera.world_to_image

        D = np.zeros((len(pose_matrices) * 2, 4))
        # loops N times, one for each image
        for i, (pose_mat, point) in enumerate(zip(pose_matrices, point_traj)):
            # Project 3D point to image
            proj_mat = world_to_camera @ np.linalg.inv(pose_mat)[:3]
            # Difference in the x value when projected
            D[2 * i] = proj_mat[2] * point[0] - proj_mat[0]
            # Difference in the y value when projected
            D[2 * i + 1] = proj_mat[2] * point[1] - proj_mat[1]

        _, _, v = np.linalg.svd(D, full_matrices=True)
        # 3D point
        pt_3d = v[-1, :3] / v[-1, 3]
        return pt_3d

    def compute_3d_points(self, pose_matrices, point_trajs):
        """ Calculate the 3D locations of the points given pose matrices for each camera
            and 2D point trajectories (and an intrinsic matrix)
            Basically find x,y,z st proj * pose_n * [u,v]_n minimizes the distance between
            the projected point and the clicked location in the image
            Computes the 3D location for T points
        pose_matrices: List of N 4x4 matrices
        trajs: T x N x 2 array of point trajectories in typical image XY format
        """
        all_rez = []
        for traj in point_trajs:
            interior = (
                (traj[:, 0] >= 0)
                & (traj[:, 0] <= self.rgb_camera.image_size[0])
                & (traj[:, 1] >= 0)
                & (traj[:, 1] <= self.rgb_camera.image_size[1])
            )
            traj = traj[interior]
            all_rez.append(self.run_triangulation(pose_matrices, traj))

        return np.array(all_rez)

    def get_reprojs(self, points_3d, pose_matrices, point_trajs):
        """
        points_3d: K x 3 matrix of 3D points
        pose_matrices: N x 2 array of point trajectories in typical XY format
        point_traj: K x N x 2 array
        """

        rez = np.zeros(point_trajs.shape)

        for j, pose_mat in enumerate(pose_matrices):
            pose_t_base = np.linalg.inv(pose_mat)
            for i, (pt_3d, traj) in enumerate(zip(points_3d, point_trajs)):
                if not np.abs(pt_3d).sum():
                    rez[i, j] = np.nan
                    continue

                pt_3d_h = np.ones(4)
                pt_3d_h[:3] = pt_3d
                pt_3d_t = (pose_t_base @ pt_3d_h)[:3]

                reproj = self.rgb_camera @ pt_3d_t
                rez[i, j] = reproj

        return rez


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default="make_3d", type=str, help="One of: make_3d")
    parser.add_argument('--dest_path', default="PycharmProjects/data/", type=str, help="where tree/bush data is stored")
    parser.add_argument('--bush_tree_name', default="bush_3_east", type=str, help="Tree or bush name")
    parser.add_argument('--annot', default="video_annot_final.json", type=str, help="which video annotation to use")
    parser.add_argument('--key_frame', default=-1, type=int, help="which key frame, set to -1 for all")
    parser.add_argument('--mask', default=-1, type=int, help="which mask, set to -1 for all")
    parser.add_argument('--mask_id', default=-1, type=int, help="which curve, set to -1 for all")
    parser.add_argument('--camera', default="azure", type=str, help="Camera, one of azure, intel TODO")
    parser.add_argument('--start_index', default=149, type=int, help="Start index for copy tree/bush")
    parser.add_argument('--end_index', default=-1, type=int, help="End index for copy tree/bush, -1 is all")
    parser.add_argument('--skip_index', default=30, type=int, help="skip frames for copy tree/bush, 10 for tree, 32 for blueberry")

    args = parser.parse_args()

    # Grab the current path
    path_start = FileNamesSubDirs.get_path()
    # Where the video_annot.json lives
    path_full = path_start + args.dest_path + args.bush_tree_name + "/"
    # The video_annot.json file
    va_fname = path_full + args.annot
    print(f"Annotation file {va_fname}")
    with open(va_fname, "r") as f:
        my_dict = json.load(f)
        va = VideoAnnotationData.read_json(my_dict)

    cam_rgb = CameraProjections(camera_fname=("azure_camera.json", "rgb_half_size"),
                                camera_calibration_fname=("azure_camera_calibration.json", "color"),
                                params={})
    cam_depth = CameraProjections(camera_fname=("azure_camera.json", "depth_narrow_unbinned"),
                                  camera_calibration_fname=("azure_camera_calibration.json", "depth"),
                                  params={})
    if args.camera == "azure":
        cam_rgb = CameraProjections(camera_fname=("azure_camera.json", "rgb_half_size"),
                                    camera_calibration_fname=("azure_camera_calibration.json", "color"),
                                    params={})
        cam_depth = CameraProjections(camera_fname=("azure_camera.json", "depth_narrow_unbinned"),
                                      camera_calibration_fname=("azure_camera_calibration.json", "depth"),
                                      params={})
    pt_kf = PointTrackerKeyFrames(va_data=va, rgb_camera=cam_rgb)
    pts_2d, vecs, valid_kfs = pt_kf.collect_background_points(b_is_horizontal=True)
    pts_2d_crvs = pt_kf.collect_crv_points(vecs, valid_kfs)

    print(f"Done {pts_2d}")
