#!/usr/bin/env python3

# From manually marked sketch curves on several key frames (plus some background points) do the following
#   Produce a consistant set of 3D points (from the curves) for each cane/branch
#   Calculate a 3D camera for each key frame
#
# Input is VideoAnnotationData file with 2 (or more) keyframes with sketched 2D points for canes/branches, plus
#   some background points that are the same across all the images
# Assumptions:
#  Version 1: Just use clicked points
#   Each keyframe has the same number of curves/canes with the same labels
#   Each keyframe has the same background points marked in the same order (can do some cleanup)
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
        self.image_size = rgb_camera.image_size
        self.tracked_pts_2d = None
        self.tracked_pts_3d = None
        self.camera_matrices = None

    def collect_background_points(self, b_is_horizontal=False, b_is_vertical=False):
        """ This is just the background points - fill in any missing points by interpolating locations
            Also tries to match points if they're in the wrong order
            Assumes first frame is correct
            @param b_is_horizontal - set True if you know the motion is left-right
            @param b_is_vertical - set True if you know the motion is up-down
            @return tracked_2d points as a numpy array"""

    def flatten_groups(self, grouped_pts):
        all_pts = []
        all_names = []
        for name, points in grouped_pts.items():
            all_pts.append(points)
            all_names.extend([name] * len(points))
        return np.concatenate(all_pts, axis=0), all_names

    def run_point_tracking(self, images_in, pts_2d_in, ref_idx=0):
        """ Run the point tracking, setting 2D points and 3D points and camera poses
        @param images_in - a list of 8 images, given as numpy cv2 arrays
        @param pts_2d_in - a list of n 2d points
        @ref_idx - which image is to be the center image """

        # Get images into torch format
        image_width = images_in[0].shape(1)
        image_height = images_in[0].shape(0)
        image_nchannels = images_in[0].shape(2)
        assert image_nchannels == 3
        self.images = np.zeros((8, image_height, image_width, image_nchannels)) # S,H,W,3
        for im_indx, im in enumerate(images_in):
            im_resize = cv2.resize(im, self.image_size)
            self.images[im_indx, :, :, :] = im_resize

        rgb_seq = torch.from_numpy(self.images).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, self.image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W

        self.tracked_pts_2d = np.zeros((len(pts_2d_in), 2))
        for indx, p in pts_2d_in:
            self.tracked_pts_2d[indx, :] = p[0]
            self.tracked_pts_2d[indx, :] = p[1]

        # xy0 = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2


        rgbs = np.stack(rgbs, axis=0)  # S,H,W,3
        rgbs = rgbs[:, :, :, ::-1].copy()  # BGR->RGB
        rgbs = rgbs[::timestride]
        S_here, H, W, C = rgbs.shape
        print('rgbs', rgbs.shape)

        self.images = np.stack()
        images = [info["image"] for info in image_info]
        targets, groups = self.flatten_groups(grouped_pts)
        trajs = self.tracker.track_points(targets, images)
        trajs = np.transpose(trajs, (1, 0, 2))  # Point, frame, coordinate

        pts_3d = None
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

    def update_tracker(self):
        if not self.image_queue.is_full:
            return

        print("Updating tracker")
        with self.current_request:
            response, trajs, groups = self.run_point_tracking(
                self.image_queue.as_list(), self.current_request, ref_idx=-1
            )
            self.tracked_3d_pub.publish(response)
            self.update_request_from_trajectory(trajs, groups)
            return response

    def unflatten_tracked_points(self, points, groups):
        rez = defaultdict(list)
        for point, group in zip(points, groups):
            rez[group].append(point)

        return rez

    def update_request_from_trajectory(self, trajs, groups):
        w = self.camera.width
        h = self.camera.height
        final_locs = trajs[-1]
        is_outside = (final_locs[:, 0] < 0) | (final_locs[:, 0] >= w) | (final_locs[:, 1] < 0) | (final_locs[:, 1] >= h)
        idx_to_stay = np.where(~is_outside)[0]
        new_req = {}

        update_locs = trajs[1]
        for idx in idx_to_stay:
            group = groups[idx]
            if group not in new_req:
                new_req[group] = []
            new_req[group].append(update_locs[idx])

        self.current_request.clear()
        for group, points in new_req.items():
            self.current_request[group] = np.array(points)
        return

    def reset(self, *_, **__):
        self.current_request.clear()
        self.image_queue.empty()
        return

    def debug_tracking(self, images, trajs, reprojs=None, output=None):
        import cv2
        from PIL import Image

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

        return final_imgs


class PointTriangulator:
    def __init__(self, camera, min_points=2):
        self.camera = camera
        self.min_points = min_points
        return

    @property
    def k(self):
        return self.camera.K

    def run_triangulation(self, pose_matrices, point_traj):
        """
        pose_matrices: List of N 4x4 matrices
        trajs: N x 2 array of point trajectories in typical image XY format
        """

        D = np.zeros((len(pose_matrices) * 2, 4))
        for i, (pose_mat, point) in enumerate(zip(pose_matrices, point_traj)):
            proj_mat = self.k @ np.linalg.inv(pose_mat)[:3]
            D[2 * i] = proj_mat[2] * point[0] - proj_mat[0]
            D[2 * i + 1] = proj_mat[2] * point[1] - proj_mat[1]

        _, _, v = np.linalg.svd(D, full_matrices=True)
        pts_3d = v[-1, :3] / v[-1, 3]
        return pts_3d

    def compute_3d_points(self, pose_matrices, point_trajs):
        """
        pose_matrices: List of N 4x4 matrices
        trajs: T x N x 2 array of point trajectories in typical image XY format
        """
        all_rez = []
        for traj in point_trajs:
            interior = (
                (traj[:, 0] >= 0)
                & (traj[:, 0] <= self.camera.width)
                & (traj[:, 1] >= 0)
                & (traj[:, 1] <= self.camera.height)
            )
            traj = traj[interior]
            if len(traj) < self.min_points:
                all_rez.append(np.zeros(3))
            else:
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

                reproj = self.camera.project3dToPixel(pt_3d_t)
                rez[i, j] = reproj

        return rez


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = PointTracker()
    rclpy.spin(node, executor=executor)


if __name__ == "__main__":
    main()
