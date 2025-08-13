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


from utils.video_annotation_data import VideoAnnotationData
from utils.camera_projections import CameraProjections
from utils.file_names_sub_dirs import FileNamesSubDirs
import numpy as np
import cv2


class PointTrackerKeyFrames():
    def __init__(self,
                 va_data: VideoAnnotationData,
                 rgb_camera: CameraProjections,
                 depth_camera: CameraProjections):

        # keep the input data
        self.va_data = va_data
        self.rgb_camera = rgb_camera
        self.depth_camera = depth_camera

        # State variables
        #   Using N as the number of camera frames/images, and T as number of points
        self.est_depth = 10.0 * (0.3 * self.depth_camera.depth_range[0] + 0.7 * self.depth_camera.depth_range[1])  # Estimated depth in cm
        self.image_size = rgb_camera.image_size
        self.kf_indices = []
        self.valid_keyframes = []
        self.crv_keyframes = []
        self.n_curves = 0
        self.n_pts_per_curve = []
        self.vecs_pts = []
        self.vecs_crvs = []
        self.center_frame = -1
        self.pose_matrices = []
        self.rot_matrices = []
        self.trans_vecs = []
        self.pt_locations_2d = []
        self.crv_pt_locations_2d = []
        self.pt_locations_3d = []
        self.crv_pt_locations_3d = []
        self.crv_depths_from_image = []
        self.depths_from_image = []
        self.b_vertical = False
        self.b_horizontal = False

    @staticmethod
    def _check_consistant(vec, pts2d_a, pts2d_b):
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

    def _consistent_vec(self, pts2d_a, pts2d_b):
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
                if self.b_horizontal:
                    vec[1] = 0
                if self.b_vertical:
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

        # Save for later
        self.b_vertical = b_is_vertical
        self.b_horizontal = b_is_horizontal

        # The 2D points we're collecting; an N-sized list by a T by 2d point
        self.pt_locations_2d = []
        # keep track of which key frames had background points
        self.valid_keyframes = []
        self.kf_indices = []
        # The 2D transform from the first marked frame to each valid keyframe
        self.vecs_pts = []
        # For each keyframe...
        for kf_index, kf in enumerate(self.va_data.keyframes):
            if len(kf.pts_2d_background) == 0:
                continue

            # If the keyframe has background points marked...
            self.pt_locations_2d.append([])
            if len(self.pt_locations_2d) == 1:
                # First keyframe with background points marked -make this the default ordering
                for pt in kf.pts_2d_background:
                    self.pt_locations_2d[-1].append([pt[0], pt[1]])
                self.valid_keyframes.append(kf)
                self.kf_indices.append(kf_index)
                self.vecs_pts.append([0, 0])
            else:
                # Poor person's ransac
                vec, match, valid = self._consistent_vec(self.pt_locations_2d[-2], kf.pts_2d_background)
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
                        pt_match = kf.pts_2d_background[best_match]
                        self.pt_locations_2d[-1].append([pt_match[0], pt_match[1]])
                self.valid_keyframes.append(kf)
                self.kf_indices.append(kf_index)
                self.vecs_pts.append([vec[0], vec[1]])   # From last frame to this one
        print("Done match background\n")
        return self.pt_locations_2d, self.vecs_pts, self.valid_keyframes

    def collect_crv_points(self, vecs, valid_keyframes):
        """ This is the cane/curve points - fill in any missing points by propagating by vecs
            Also tries to match points if they're in the wrong order
            @param vecs - the vector to use if no match is found
            @param valid_keyframes - The keyframes to use
            @return pt_locations_2d N x T list of points"""
        # List that is size N containing one list for each curve, with each curve having some number of points
        self.crv_pt_locations_2d = []
        self.crv_keyframes = []

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
            vec_avg = [0, 0]
            for mask_id_crv_list in kf.sketch_curves:
                vec_avg = [0, 0]
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
                        vec_avg[0] += vec[0]
                        vec_avg[1] += vec[1]
                    count += 1

            # Done with all curves for this keyframe
            n_crvs.append(count)

            if count > 0:
                self.vecs_crvs.append([vec_avg[0] / count, vec_avg[1] / count])
                vec_avg[0] /= count
                if last_valid_kf_with_curves != -1:
                    if count is not self.n_curves:
                        print(f"Warning, kf {kf.image_name} has wrong number of curves, got {count} expected {self.n_curves}")
                    for crv_indx, count_n_pts in enumerate(self.n_pts_per_curve):
                        if count_n_pts != len(self.crv_pt_locations_2d[kf_indx][crv_indx]):
                            print(f"Warning, kf {kf.image_name} has wrong number of points in curves, got {len(self.crv_pt_locations_2d[kf_indx][crv_indx])} expected {count_n_pts}")
                else:
                    self.n_curves = count
                    for n_crv_indx in range(0, count):
                        self.n_pts_per_curve.append(len(self.crv_pt_locations_2d[kf_indx][n_crv_indx]))
                last_valid_kf_with_curves = kf_indx
                self.crv_keyframes.append(kf_indx)

                self.n_curves = count
            else:
                self.vecs_crvs.append([0, 0])

        print(f"Curve counts {n_crvs}")
        for kf_indx in range(0, len(self.valid_keyframes)):
            if kf_indx in self.crv_keyframes:
                continue
            prev_indx = 0
            next_indx = len(self.va_data.keyframes) - 1
            for find_indx in self.crv_keyframes:
                if kf_indx > find_indx:
                    prev_indx = find_indx
                if kf_indx < find_indx:
                    next_indx = find_indx
                    break
            perc = (kf_indx - prev_indx) / (next_indx - prev_indx)
            print(f"kf {kf_indx}, p {prev_indx}, n {next_indx}")
            for crv_indx in range(0, self.n_curves):
                self.crv_pt_locations_2d[kf_indx].append([])
                for pt_indx in range(0, len(self.crv_pt_locations_2d[prev_indx][crv_indx])):
                    pt_prev = self.crv_pt_locations_2d[prev_indx][crv_indx][pt_indx]
                    pt_next = self.crv_pt_locations_2d[next_indx][crv_indx][pt_indx]
                    pt = [(1.0 - perc) * pt_prev[0] + perc * pt_next[0],
                          (1.0 - perc) * pt_prev[1] + perc * pt_next[1]]
                    self.crv_pt_locations_2d[kf_indx][crv_indx].append(pt)
        return self.crv_pt_locations_2d

    def _initial_cam_pose_from_essential(self):
        """ Get the initial camera poses from the 2d points and the essential matrix"""

        self.pose_matrices = []
        self.rot_matrices = []
        self.trans_vecs = []
        pts = np.array(self.pt_locations_2d, dtype=np.float64)

        # An example of calling undistortPoints, which is called within recover pose
        # pts_test = np.squeeze(pts[0, :])
        # Uncomment to check that undistort (used in finding the essential matrix) works
        # pts_out = cv2.undistortPoints(src=pts_test, cameraMatrix=cam_rgb.world_to_image,
        #                               distCoeffs=cam_rgb.image_distortion_coefs)

        # Get the (approximate) camera matrices
        indx_mid_im = pts.shape[0] // 2
        self.center_frame = indx_mid_im
        print(f"Mid frame {indx_mid_im}")
        pts1_for_mult = np.ones((pts.shape[1], 3))
        pts2_for_mult = np.ones((pts.shape[1], 3))
        for indx in range(pts.shape[0]):
            pts1 = np.squeeze(pts[indx_mid_im, :])
            pts2 = np.squeeze(pts[indx, :])
            # Since we're using the camera calibration matrix K, keep points in the image space coordinate
            #  system (width/height)
            essential_mat = cv2.findEssentialMat(points1=pts1,
                                                 points2=pts2,
                                                 cameraMatrix=cam_rgb.world_to_image)

            # If the essential matrix is (mostly) correct, pts1^t @ Kt @ E @ K @pts2 should be mostly zero
            pts1_for_mult[:, 0:2] = pts1[:, 0:2]
            pts2_for_mult[:, 0:2] = pts2[:, 0:2]
            #  Put K on either end so that can keep points in image coordinates
            #  From: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gab705726dc6b655acf50bc936942824ef
            #
            k_inv = np.linalg.inv(cam_rgb.world_to_image)
            # kt_e_k = cam_rgb.world_to_image.transpose() @ essential_mat[0] @ cam_rgb.world_to_image
            # Does a ransac, so masks says which points are kept. First argument is the actual matrix, second is the mask
            print(f"E {indx} {essential_mat[0]} \nMasks: {essential_mat[1].transpose()}")
            # Check to see how well the essential matrix worked (should be close to zero)
            for pt_indx in range(0, pts1.shape[0]):
                pt1 = k_inv @ pts1_for_mult[pt_indx, :]
                pt2 = k_inv @ pts2_for_mult[pt_indx, :]
                val = pt2.transpose() @ essential_mat[0] @ pt1
                print(f" {val}", end="")

            # This gets a rotation and translation out of the essential matrix
            cam_recover = cv2.recoverPose(E=essential_mat[0],
                                          points1=np.squeeze(pts[indx_mid_im, :]),
                                          points2=np.squeeze(pts[indx, :]),
                                          cameraMatrix=cam_rgb.world_to_image,
                                          mask=essential_mat[1])

            print(f"\nRot {cam_recover[0]}\n{cam_recover[1]}")
            print(f"Trans {cam_recover[2].transpose()}\n")

            # Keep the rotation and the translation seperately
            self.rot_matrices.append(cam_recover[1])
            self.trans_vecs.append(cam_recover[2])

            # Build the matrix - copy the rotation into the 3x3 in the upper left, the translation into the right most column
            cam_pose = np.identity(4)
            cam_pose[0:3, 0:3] = self.rot_matrices[-1]
            cam_pose[0:3, 3] = np.transpose(self.trans_vecs[-1])

            self.pose_matrices.append(cam_pose)

    def _cam_pose_from_vec(self):
        """ Get the initial camera poses for vertial or horizontal motion
        Assumes no rotation of the camera, just pan
        """

        self.pose_matrices = []
        self.rot_matrices = []
        self.trans_vecs = []

        # Get the (approximate) camera matrices
        indx_mid_im = len(self.pt_locations_2d) // 2
        self.center_frame = indx_mid_im
        print(f"Mid frame {indx_mid_im}")

        # Assumes the objects of interest are around 1/2 meter away
        d_est_per_pix_w = self.est_depth * np.sin(2.0 * self.rgb_camera.camera_width_angle) / self.rgb_camera.image_size[0]
        d_est_per_pix_h = self.est_depth * np.sin(2.0 * self.rgb_camera.camera_height_angle) / self.rgb_camera.image_size[1]
        if self.b_vertical:
            d_est_per_pix = d_est_per_pix_h
        elif self.b_horizontal:
            d_est_per_pix = d_est_per_pix_w
        else:
            d_est_per_pix = 0.5 * (d_est_per_pix_w + d_est_per_pix_h)

        for indx in range(len(self.pt_locations_2d)):
            vec = [0, 0]
            if indx < indx_mid_im:
                for nindx in range(indx, indx_mid_im):
                    vec[0] += self.vecs_pts[nindx][0]
                    vec[1] += self.vecs_pts[nindx][1]
            elif indx > indx_mid_im:
                for nindx in range(indx_mid_im+1, indx+1):
                    vec[0] += self.vecs_pts[nindx][0]
                    vec[1] += self.vecs_pts[nindx][1]
            else:
                pass

            # Rotation is identity
            self.rot_matrices.append(np.identity(3))
            vec[0] *= d_est_per_pix
            vec[1] *= d_est_per_pix
            trans_vec = np.array([vec[0] * d_est_per_pix, vec[1] * d_est_per_pix, 0], dtype=np.float64)
            self.trans_vecs.append(trans_vec)

            # Build the matrix - copy the rotation into the 3x3 in the upper left, the translation into the right most column
            cam_pose = np.identity(4)
            cam_pose[0:3, 0:3] = self.rot_matrices[-1]
            cam_pose[0:3, 3] = np.transpose(self.trans_vecs[-1])

            self.pose_matrices.append(cam_pose)

    def _run_triangulation(self, pts_2d, mask):
        """ Calculate the 3D location of a point given pose matrices for each camera
            and the 2D point location for each of the images (and an intrinsic matrix)
            Basically find x,y,z st proj * pose_n * [u,v]_n minimizes the distance between
            the projected point and the clicked location in the image
            Multi-view triangulation https://www.overleaf.com/project/643eafda3db3b7748aa902f3
        param pts_2d: N x 2 array of point trajectory for a single point in typical image XY format
        param mask: Which frames to ignore
        """

        # 3x2 matrix
        world_to_camera = np.identity(4)
        world_to_camera[0:3, 0:3] = self.rgb_camera.world_to_image

        n_keep = np.count_nonzero(mask)
        D = np.zeros((n_keep * 2, 4))
        # loops N times, one for each image
        indx = 0
        count = 0
        for pose_mat, pts in zip(self.pose_matrices, pts_2d):
            if not mask[indx]:
                indx += 1
                continue
            # Project 3D point to image
            proj_mat = world_to_camera @ np.linalg.inv(pose_mat)
            p = proj_mat[0, :3]
            # Difference in the x value when projected

            D[2 * count, :] = proj_mat[2, :] * pts[0] - proj_mat[0, :]
            # Difference in the y value when projected
            D[2 * count + 1, :] = proj_mat[2, :] * pts[1] - proj_mat[1, :]
            indx += 1
            count += 1

        assert count == n_keep
        _, _, v = np.linalg.svd(D, full_matrices=True)
        # 3D point
        pt_3d = v[-1, :3] / v[-1, 3]
        return pt_3d

    def _run_triangulation_all(self, pts_2d, mask):
        """Calculate the 3d locations of all of the points in 2d
        @param pts_2d - Nx2 set of points as numpy array
        @param mask - which frames to use
        @return 3d locations"""

        pts_3d = []
        for pt_indx in range(pts_2d.shape[1]):
            pts_2d_track = []
            for row_indx in range(pts_2d.shape[0]):
                pts_2d_track.append([pts_2d[row_indx][pt_indx][0], pts_2d[row_indx][pt_indx][1]])
            pts_3d.append(self._run_triangulation(pts_2d_track, mask))
        return pts_3d

    def _scale_3d_points(self, seg_length):
        """ Calculate the scale that will make the distances between the sketched curve points be seg_length
        @param seg_length - whatever you want point i - point i+1 to be a length for
        @:return flattened list of 3d points"""

        seg_lengths = []
        scl_quadratic = []
        for crv_indx in range(0, self.n_curves):
            print(f"crv {crv_indx}: ", end="")
            for pt_indx in range(0, self.n_pts_per_curve[crv_indx] - 1):
                pt1 = np.array(self.crv_pt_locations_3d[crv_indx][pt_indx])
                pt2 = np.array(self.crv_pt_locations_3d[crv_indx][pt_indx+1])
                len_seg = np.linalg.norm(pt1 - pt2)
                seg_lengths.append(len_seg)
                print(f" {len_seg:0.2}", end="")
                quad_form_a = np.sum(pt1 ** 2) + np.sum(pt2 ** 2)
                quad_form_b = -2.0 * (pt1 @ pt2.transpose())
                quad_form_c = -seg_length ** 2

                det = quad_form_b ** 2 - 4.0 * quad_form_a * quad_form_c
                if det < 0.0:
                    continue
                soln1 = (-quad_form_b + np.sqrt(det)) / (2.0 * quad_form_a)
                soln2 = (-quad_form_b - np.sqrt(det)) / (2.0 * quad_form_a)
                if soln1 < 0.0 and soln2 < 0.0:
                    continue
                if soln1 > 0.0:
                    scl_quadratic.append(soln1)
                else:
                    scl_quadratic.append(soln1)
                print("")
        scl_quad = np.mean(np.array(scl_quadratic))
        seg_len = np.mean(np.array(seg_lengths))
        scl = seg_length / seg_len

        for pts in self.pt_locations_3d:
            for indx in range(0, 3):
                pts[indx] *= scl

        crv_list = []
        for crv_indx in range(0, self.n_curves):
            for pt_indx in range(0, self.n_pts_per_curve[crv_indx]):
                for indx in range(0, 3):
                    self.crv_pt_locations_3d[crv_indx][pt_indx][indx] *= scl
                crv_list.append(self.crv_pt_locations_3d[crv_indx][pt_indx])
        pt1 = np.array(self.crv_pt_locations_3d[0][1])
        pt2 = np.array(self.crv_pt_locations_3d[0][0])
        print(f" end len {np.linalg.norm(pt1 - pt2)}")

        return np.array(crv_list, dtype=np.float64)

    def _error_per_frame(self, crv_2d_flattened, crv_3d_flattened):
        """ Returns the error per frame of the reprojection"""
        pts_3d = np.array(self.pt_locations_3d, dtype=np.float64)
        all_err = []
        all_err_crv = []
        for indx, (rmat, tvec) in enumerate(zip(self.rot_matrices, self.trans_vecs)):
            pts_proj = cv2.projectPoints(pts_3d, rmat, tvec,
                                         cameraMatrix=cam_rgb.world_to_image,
                                         distCoeffs=cam_rgb.image_distortion_coefs)
            err_sum = 0.0
            for pt_indx, pt_2d in enumerate(self.pt_locations_2d[indx]):
                pt_proj_2d = [pts_proj[0][pt_indx, 0, 0], pts_proj[0][pt_indx, 0, 1]]
                err = np.sqrt((pt_2d[0] - pt_proj_2d[0]) ** 2 + (pt_2d[1] - pt_proj_2d[1]) ** 2)
                err_sum += err
                # print(f" {err:.2f}", end="")
            # print(f" avg {err_sum / len(self.pt_locations_2d)}\n")
            all_err.append(err_sum / len(self.pt_locations_2d[indx]))

            if len(crv_2d_flattened[indx]) == 0:
                continue

            pts_proj = cv2.projectPoints(crv_3d_flattened, rmat, tvec,
                                         cameraMatrix=cam_rgb.world_to_image,
                                         distCoeffs=cam_rgb.image_distortion_coefs)
            err_sum = 0.0
            for pt_indx, pt_2d in enumerate(crv_2d_flattened[indx]):
                pt_proj_2d = [pts_proj[0][pt_indx, 0, 0], pts_proj[0][pt_indx, 0, 1]]
                err = np.sqrt((pt_2d[0] - pt_proj_2d[0]) ** 2 + (pt_2d[1] - pt_proj_2d[1]) ** 2)
                err_sum += err
                # print(f" {err:.2f}", end="")
            all_err_crv.append(err_sum / len(crv_2d_flattened[indx]))

        print(f"All err: {all_err}")
        print(f"All err crv: {all_err_crv}")
        return all_err

    def _triangulate_3d_crv_pts(self):
        """ Use the current pose matrices to project the curve points
        @ returns Nframes X sum(number points per curve) X 2 list of 2d points, sum(number points per curve) X 3 3D pts"""

        mask = np.ones((len(self.pose_matrices),)) < 0
        for kf_indx in self.crv_keyframes:
            mask[kf_indx] = True
        # One 2d point for every valid keyframe
        pts_2d = np.ones((len(self.valid_keyframes), 2))

        self.crv_pt_locations_3d = []

        crv_pts_3d_flattened = []
        crv_pts_2d_flattened = []
        for _ in self.valid_keyframes:
            crv_pts_2d_flattened.append([])
        for crv_indx in range(0, self.n_curves):
            self.crv_pt_locations_3d.append([])

            for pt_indx in range(0, self.n_pts_per_curve[crv_indx]):
                # Collect all the image points from all the valid key frames
                for kf_indx in range(0, len(self.crv_pt_locations_2d)):
                    pts_2d[kf_indx, 0] = self.crv_pt_locations_2d[kf_indx][crv_indx][pt_indx][0]
                    pts_2d[kf_indx, 1] = self.crv_pt_locations_2d[kf_indx][crv_indx][pt_indx][1]
                    crv_pts_2d_flattened[kf_indx].append(self.crv_pt_locations_2d[kf_indx][crv_indx][pt_indx])
                # Now actually run
                crv_pt_3d = self._run_triangulation(pts_2d, mask)
                self.crv_pt_locations_3d[-1].append(crv_pt_3d)
                crv_pts_3d_flattened.append(crv_pt_3d)

        return crv_pts_2d_flattened, np.array(crv_pts_3d_flattened, dtype=np.float64)

    def solve_3d_pts(self, b_no_camera_rotation=True, seq_length=1.0, iters=2):
        """ Assumes self.pt_locations_2d has been filled in"""

        # Initial guesses at the camera poses
        if b_no_camera_rotation:
            self._cam_pose_from_vec()
        else:
            self._initial_cam_pose_from_essential()

        np.set_printoptions(precision=3, suppress=True)
        # Use those matrices to get the 3D points - OpenCV likes everything in float 32 or float 64
        pts_2d = np.array(self.pt_locations_2d, dtype=np.float64)
        mask = np.ones((len(self.pose_matrices), )) > 0
        self.pt_locations_3d = self._run_triangulation_all(pts_2d, mask)
        print(f" {self.pt_locations_3d}")

        crv_2d, crv_3d = self._triangulate_3d_crv_pts()
        crv_3d = self._scale_3d_points(seq_length)

        all_err = self._error_per_frame(crv_2d, crv_3d)
        rvec = np.ones((3, 1), dtype=np.float64)
        for iter in range(0, iters):
            for kf_indx, (pts2d_im, rmat, tvec) in enumerate(zip(self.pt_locations_2d, self.rot_matrices, self.trans_vecs)):
                # Use the estimated 3d points to get better camera poses
                cv2.Rodrigues(rmat, rvec)  # Does the rotation in place, will fill in rvec
                print(f"{kf_indx} RVec {rvec.transpose()} tvec {tvec.transpose()}")
                if len(crv_2d[kf_indx]) > 0:
                    pts_3d = np.vstack((np.array(self.pt_locations_3d, dtype=np.float64), crv_3d))
                    pts2d_im_npa = np.vstack((np.array(pts2d_im, dtype=np.float64), np.array(crv_2d[kf_indx], dtype=np.float64)))
                else:
                    pts_3d = np.array(self.pt_locations_3d, dtype=np.float64)
                    pts2d_im_npa = np.array(pts2d_im, dtype=np.float64)
                cv2.solvePnPRefineLM(pts_3d, pts2d_im_npa,
                                     self.rgb_camera.world_to_image, self.rgb_camera.image_distortion_coefs,
                                     rvec, tvec)
                # Put the rotation back in matrix form
                print(f"   RVec {rvec.transpose()} tvec {tvec.transpose()}")
                cv2.Rodrigues(rvec, rmat)
                # We just got the object pose; so undo this one
                print(f"Before\n{self.pose_matrices[kf_indx]}")
                self.pose_matrices[kf_indx][0:3, 0:3] = rmat
                self.pose_matrices[kf_indx][0:3, 3] = tvec.transpose()
                self.pose_matrices[kf_indx] = np.linalg.inv(self.pose_matrices[kf_indx])
                print(f"After\n{self.pose_matrices[kf_indx]}")
            # This should drop projection error, with the new rvec/tvecs
            all_err = self._error_per_frame(crv_2d, crv_3d)
            if iter < iters-1:
                pts_3d = np.array(self.pt_locations_3d)
                print(f"pts_3d {pts_3d}")
                self.pt_locations_3d = self._run_triangulation_all(pts_2d, mask)
                pts_3d = np.array(self.pt_locations_3d)
                print(f"pts_3d {pts_3d}")
                print(f"\n")
                crv_2d, crv_3d = self._triangulate_3d_crv_pts()
                crv_3d = self._scale_3d_points(seq_length)
                print(f" {self.pt_locations_3d}")

                all_err = self._error_per_frame(crv_2d, crv_3d)
                # Oddly, this doesn't help - so don't do it
                # mask = np.array(all_err) < 100

    def proj_point(self, kf_indx, pt_3d, use_depth=False):
        """ Project the point into the image, either rgb or depth
        @param kf_indx - which key frame
        @param pt_3d - 3d point as list or np array
        @param use_depth - if True, use depth map camera"""

        cam = cam_depth if use_depth else cam_rgb
        pts_proj = cv2.projectPoints(np.array(pt_3d, dtype=np.float64),
                                     rvec=self.rot_matrices[kf_indx],
                                     tvec=self.trans_vecs[kf_indx],
                                     cameraMatrix=cam.world_to_image,
                                     distCoeffs=cam.image_distortion_coefs)
        return pts_proj[0][0, 0, :].flatten()

    def point_rgb(self, kf_indx, pt_2d_rgb, pt_3d):
        """ Get all of the possible locations of the background point
        @param kf_indx: which key frame
        @param pt_indx: which point
        @return all of the possible locations of the background point as a dictionary"""

        ret_dict = {"click_interp": pt_2d_rgb,
                    "proj_3d": self.proj_point(kf_indx, pt_3d)}

        return ret_dict


    def point_depth(self, kf_indx, pt_2d_rgb, pt_3d):
        """ Two ways to get from 2d rgb to 2d depth; our hand-built matrix OR kInvk
            Two ways to get 2d point: Either use click point or project from 3d
        @param kf_indx: which key frame
        @param pt_2d_rgb - the click point in image
        @param pt_3d - the 3d point as list or np array"""

        pt_2d_rgb = np.array(pt_2d_rgb, dtype=np.float64)
        pt_3d = np.array(pt_3d, dtype=np.float64)
        pt_3d_rgb = self.proj_point(kf_indx, pt_3d)
        mat_k_inv_k = cam_depth.world_to_image @ np.linalg.inv(cam_rgb.world_to_image)
        ret_dict = {"click_interp_matrix": self.va_data.matrix_rgb_to_depth @ pt_2d_rgb,
                    "proj_3d": self.proj_point(kf_indx, pt_3d, use_depth=True),
                    "proj_rgb_matrix": self.va_data.matrix_rgb_to_depth @ pt_3d_rgb,
                    "kInvk": mat_k_inv_k @ pt_2d_rgb,
                    "proj_rgb_kInvk": mat_k_inv_k @ pt_3d_rgb,
                    }
        return ret_dict

    def background_point_depth(self, kf_indx, pt_indx):
        """ Get all of the possible locations of the background point
        @param kf_indx: which key frame
        @param pt_indx: which point
        @return all of the possible locations of the background point as a dictionary"""

        pt_2d_rgb = np.array(self.pt_locations_2d[kf_indx][pt_indx], dtype=np.float64)
        pt_3d = np.array(self.pt_locations_3d[kf_indx][pt_indx], dtype=np.float64)

        return self.point_depth(kf_indx=kf_indx, pt_2d_rgb=pt_2d_rgb, pt_3d=pt_3d)

    def point_depth_from_image(self):
        pt_proj_2d = np.ones((3, 1))
        depth_values = [[] for i in range(len(self.pt_locations_3d))]
        width = 8
        n_pixs_box = 2 * width * 2 * width // 4

        d_min_world = self.depth_camera.depth_range[0]
        d_max_world = self.depth_camera.depth_range[1]
        for count, pose_matrix in enumerate(self.pose_matrices):
            pts_proj = cv2.projectPoints(np.array(self.pt_locations_3d, dtype=np.float64),
                                         rvec=self.rot_matrices[count],
                                         tvec=self.trans_vecs[count],
                                         cameraMatrix=cam_depth.world_to_image,
                                         distCoeffs=cam_depth.image_distortion_coefs)
            kf_indx = self.kf_indices[count]
            depth_image_name = self.va_data.get_depth_image_name((0, kf_indx, 0, 0))
            im_depth_full = cv2.imread(depth_image_name)
            im_depth_r = im_depth_full[:, :, 2]
            im_depth_g = im_depth_full[:, :, 1]
            im_depth_b = im_depth_full[:, :, 0]

            for pt_index in range(0, pts_proj[0].shape[0]):
                #  pt_proj_2d[0] = pts_proj[0][pt_index][0][0]
                #  pt_proj_2d[1] = pts_proj[0][pt_index][0][1]
                #  pt_proj_depth = self.va_data.matrix_rgb_to_depth @ pt_proj_2d
                pt_proj_depth = pts_proj[0][pt_index, 0, 0:2]

                if pt_proj_depth[0] <= width or pt_proj_depth[0] >= self.depth_camera.image_size[1] - width -1:
                    print(f"Pixel out of image {pt_proj_depth} kf {kf_indx} pt {pt_index}")
                if pt_proj_depth[1] <= width or pt_proj_depth[1] >= self.depth_camera.image_size[0] - width - 1:
                    print(f"Pixel out of image {pt_proj_depth} kf {kf_indx} pt {pt_index}")
                x = int(pt_proj_depth[0].flatten())
                y = int(pt_proj_depth[1].flatten())
                depth_pixels_r = im_depth_r[y-width:y+width, x-width:x+width].flatten()
                depth_pixels_g = im_depth_g[y-width:y+width, x-width:x+width].flatten()
                im_depth_full[y-width:y+width, x-width:x+width, 0] = 200
                im_depth_full[y-width:y+width, x-width:x+width, 1] = 200
                depth_pixels_g = depth_pixels_g[depth_pixels_r < 253]
                depth_pixels_r = depth_pixels_r[depth_pixels_r < 253]
                if depth_pixels_r.size > n_pixs_box:
                    depth_pixels = depth_pixels_r + (255.0 - depth_pixels_g)
                    depth_closest = depth_pixels.min()
                    d_val = d_min_world + depth_closest / 255.0 * (d_max_world - d_min_world)
                    depth_values[pt_index].append(d_val)
                else:
                    print(f"Not enough pixels kf {kf_indx} pt {pt_index} {depth_pixels_r.size}")

            fname_debug = self.va_data.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=True, b_add_tag=False) + "_depth_pts.png"
            cv2.imwrite(fname_debug, im_depth_full)

            fname_debug2 = self.va_data.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=True, b_add_tag=False) + "_depth_guess.png"
            im_depth_guess = (im_depth_r + (255.0 - im_depth_g)) / 2.0
            im_depth_guess = im_depth_guess.astype(np.uint8)
            cv2.imwrite(fname_debug2, im_depth_guess)

        self.depths_from_image = []
        for idx, depth_value_list in enumerate(depth_values):
            depth_from_image = -1.0
            if len(depth_value_list) > 0:
                depth_as_np = np.array(depth_value_list)
                diff = np.max(depth_as_np) - np.min(depth_as_np)
                count_min = np.count_nonzero(depth_as_np < 1.1 * np.min(depth_as_np))
                count_max = np.count_nonzero(depth_as_np > 0.9 * np.max(depth_as_np))

                depth_from_image = np.mean(depth_as_np)

            print(f"{self.pt_locations_3d[idx]}, {depth_from_image}")
            print(f"{depth_value_list}")
            self.depths_from_image.append(depth_from_image)

    def crv_point_depth_from_image(self):
        pt_proj_2d = np.ones((3, 1))
        width = 8
        n_pixs_box = 2 * width * 2 * width // 4

        d_min_world = self.depth_camera.depth_range[0]
        d_max_world = self.depth_camera.depth_range[1]
        crv_lens = []

        depth_values = [[] for i in range(len(self.crv_pt_locations_3d))]
        for indx, dv in enumerate(depth_values):
            for _ in range(0, len(self.crv_pt_locations_3d[indx])):
                dv.append([])

        pt_rgb_2d = np.ones((3, 1))
        for crv_indx in range(0, len(self.crv_pt_locations_3d)):
            crv_pts_3d = np.array(self.crv_pt_locations_3d[crv_indx], dtype=np.float64)
            for pt_index in range(0, len(self.crv_pt_locations_3d[crv_indx]) - 1):
                seg_dist = np.linalg.norm(crv_pts_3d[pt_index, :] - crv_pts_3d[pt_index + 1, :])
                crv_lens.append(seg_dist)

        for kf_indx, crv_pt_list in enumerate(self.crv_pt_locations_2d):
            if len(crv_pt_list) == 0:
                continue

            depth_image_name = self.va_data.get_depth_image_name((0, kf_indx, 0, 0))
            im_depth_full = cv2.imread(depth_image_name)
            im_depth_r = im_depth_full[:, :, 2]
            im_depth_g = im_depth_full[:, :, 1]

            for crv_indx, pts_all in enumerate(crv_pt_list):
                for pt_indx, pt in enumerate(pts_all):
                    pt_rgb_2d[0] = pt[0]
                    pt_rgb_2d[1] = pt[1]
                    pt_proj_depth = self.va_data.matrix_rgb_to_depth @ pt_rgb_2d

                    if pt_proj_depth[0] <= width or pt_proj_depth[0] >= self.depth_camera.image_size[1] - width - 1:
                        print(f"Pixel out of image {pt_proj_depth} kf {kf_indx} crv {crv_indx} pt {pt_indx}")
                    if pt_proj_depth[1] <= width or pt_proj_depth[1] >= self.depth_camera.image_size[0] - width - 1:
                        print(f"Pixel out of image {pt_proj_depth} kf {kf_indx} crv {crv_indx} pt {pt_indx}")

                    x = int(pt_proj_depth[0].flatten())
                    y = int(pt_proj_depth[1].flatten())
                    depth_pixels_r = im_depth_r[y - width:y + width, x - width:x + width].flatten()
                    depth_pixels_g = im_depth_g[y - width:y + width, x - width:x + width].flatten()
                    im_depth_full[y - width:y + width, x - width:x + width, 0] = 200
                    im_depth_full[y - width:y + width, x - width:x + width, 1] = 200
                    depth_pixels_g = depth_pixels_g[depth_pixels_r < 253]
                    depth_pixels_r = depth_pixels_r[depth_pixels_r < 253]
                    if depth_pixels_r.size > n_pixs_box:
                        depth_pixels = depth_pixels_r + (255.0 - depth_pixels_g)
                        depth_closest = depth_pixels.min()
                        d_val = d_min_world + depth_closest / 255.0 * (d_max_world - d_min_world)
                        depth_values[crv_indx][pt_indx].append(d_val)
                    else:
                        print(f"Not enough pixels kf {kf_indx} crv {crv_indx} pt {pt_indx} {depth_pixels_r.size}")

            fname_debug = self.va_data.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=True,
                                                            b_add_tag=False) + "_depth_crv_pts.png"
            cv2.imwrite(fname_debug, im_depth_full)

        self.crv_depths_from_image = []
        for crv_indx, crvs in enumerate(depth_values):
            self.crv_depths_from_image.append([])
            for pt_indx, crv_depths in enumerate(crvs):
                depth_from_image = -1.0
                if len(crv_depths) > 0:
                    depth_as_np = np.array(crv_depths)
                    diff = np.max(depth_as_np) - np.min(depth_as_np)
                    depth_from_image = np.mean(depth_as_np)

                print(f"{self.crv_pt_locations_3d[crv_indx][pt_indx]}, {depth_from_image}")
                print(f"{crv_depths}")
                self.crv_depths_from_image[crv_indx].append(depth_from_image)

    def debug_images(self, va : VideoAnnotationData):
        """ Produce images with tracks on both rgb and depth (if 3D points given)
        Assumes pt_locations_2d and/or crv_pt_locations_2d have been created already
        @param va - video annotation (for where to put the images)
        @param valid_kf the keyframes for which we have 2d points"""
        from draw_routines.image_draw_geom_utils import draw_cross, draw_line, draw_box

        count = 0
        thickness = 4
        pts_3d = np.array(self.pt_locations_3d, dtype=np.float64)
        # RGB images - draw sketch curves, projected sketch curves, background points
        #   Use boxes for user click points
        for kf_indx, kf in zip(self.kf_indices, self.valid_keyframes):
            fname_rgb = va.get_image_name((0, kf_indx, 0, 0))
            im_rgb = cv2.imread(fname_rgb)

            # Draw the vector in the middle of the screen in yellow
            pt_mid = [im_rgb.shape[1] // 2, im_rgb.shape[0] // 2]
            pt_end_mid = pt_mid + self.vecs_pts[count]
            draw_cross(im_rgb, pt_mid, color=[255, 255, 0], thickness=thickness // 2)
            draw_line(im_rgb, pt_mid, pt_end_mid, color=[255, 255, 0], thickness=thickness // 2)

            # Draw the original clicked background points in white
            for bp in kf.pts_2d_background:
                draw_box(im_rgb, bp, color=[200, 200, 200], width=4 * thickness)

            # Draw the original clicked curve points in white
            for mask in kf.sketch_curves:
                for crv in mask:
                    for pt in crv.backbone_pts:
                        draw_box(im_rgb, pt, color=[200, 200, 200], width=4 * thickness)

            # Draw the 2D points saved in pts_2d, with lines between them
            for pt_indx, pt in enumerate(self.pt_locations_2d[count]):
                draw_cross(im_rgb, pt, color=[20, 200, 200], thickness=thickness)
                if pt_indx == 0:
                    pt_prev = pt
                else:
                    pt_prev = self.pt_locations_2d[count][pt_indx - 1]
                draw_line(im_rgb, pt, pt_prev, color=[20, 200, 200], thickness=thickness)

            # Draw the curve points saved in crv_pts_2d, with lines between them
            for icrv, crv in enumerate(self.crv_pt_locations_2d[count]):
                idiv = 105 // self.n_curves
                for pt_indx, pt in enumerate(crv):
                    draw_cross(im_rgb, pt, color=[250, 20, 200], thickness=2 * thickness)
                    if pt_indx == 0:
                        pt_prev = pt
                    else:
                        pt_prev = crv[pt_indx - 1]

                    draw_line(im_rgb, pt, pt_prev, color=[250, 20, 100 + idiv * 30], thickness=thickness)

            # Draw the projected 3d curves
            if self.pose_matrices is not [] and self.crv_pt_locations_3d is not []:
                for crv_indx in range(0, len(self.crv_pt_locations_3d)):
                    crv_pts_3d = np.array(self.crv_pt_locations_3d[crv_indx], dtype=np.float64)
                    crv_pts_proj = cv2.projectPoints(crv_pts_3d,
                                                     rvec=self.rot_matrices[count],
                                                     tvec=self.trans_vecs[count],
                                                     cameraMatrix=cam_rgb.world_to_image,
                                                     distCoeffs=cam_rgb.image_distortion_coefs)
                    for pt_indx in range(0, crv_pts_3d.shape[0]):
                        pt = crv_pts_proj[0][pt_indx, 0, :]
                        draw_cross(im_rgb, pt, color=[20, 220, 200], thickness=thickness)
                        if pt_indx == 0:
                            pt_prev = pt
                        else:
                            pt_prev = crv_pts_proj[0][pt_indx - 1, 0, :]
                        draw_line(im_rgb, pt, pt_prev, color=[20, 20, 200], thickness=thickness // 2)


            # Draw the projected background points
            if self.pose_matrices is not [] and self.pt_locations_3d is not []:
                pts_proj = cv2.projectPoints(pts_3d,
                                             rvec=self.rot_matrices[count],
                                             tvec=self.trans_vecs[count],
                                             cameraMatrix=cam_rgb.world_to_image,
                                             distCoeffs=cam_rgb.image_distortion_coefs)
                for pt_indx in range(0, pts_proj[0].shape[0]):
                    pt_row = pts_proj[0][pt_indx]
                    draw_box(im_rgb, pt_row, color=[100, 150, 200], width=thickness * 2)

            count = count + 1
            fname = va.get_image_name((0, kf_indx, 0, 0), b_debug_path=True, b_add_tag=False) + "_pts2d.png"
            cv2.imwrite(fname, im_rgb)

        # Now do the depth images
        count = 0
        pt3d = np.ones((3, 1))
        for kf_indx, kf in zip(self.kf_indices, self.valid_keyframes):
            fname_depth = va.get_depth_image_name((0, kf_indx, 0, 0))
            im_depth = cv2.imread(fname_depth)

            # Draw the 2D background points mapped via the intrinsic matrix to the depth image
            k_rgb_inv = np.linalg.inv(self.rgb_camera.world_to_image)
            k_depth = self.depth_camera.world_to_image
            for bp in kf.pts_2d_background:
                pt3d[0] = bp[0]
                pt3d[1] = bp[1]
                pt_depth = k_depth @ k_rgb_inv @ pt3d
                draw_box(im_depth, pt_depth[0:2].transpose(), color=[200, 200, 200], width=4 * thickness)

            # Draw the curve points saved in crv_pts_2d, with lines between them
            pt_prev = np.copy(pt3d)
            pt_prev_direct = np.copy(pt3d)
            for crv in self.crv_pt_locations_2d[count]:
                for pt_indx, pt in enumerate(crv):
                    # Purple - just use the k matrices to map the 2d image point to the 2d depth point
                    pt3d[0] = pt[0]
                    pt3d[1] = pt[1]
                    pt_depth = k_depth @ k_rgb_inv @ pt3d
                    draw_cross(im_depth, pt_depth[0:2].transpose(), color=[250, 20, 200], thickness=2 * thickness)
                    if pt_indx != 0:
                        draw_line(im_depth, pt_depth[0:2].transpose(), pt_prev, color=[250, 20, 200], thickness=thickness)
                    pt_prev = np.copy(pt_depth[0:2].transpose())

                    # Blue - Use the hand-fitted matrix to map the rgb point to the depth image
                    pt_depth_direct = va.matrix_rgb_to_depth @ pt3d
                    draw_cross(im_depth, pt_depth_direct[0:2].transpose(), color=[250, 120, 100], thickness=2 * thickness)
                    if pt_indx != 0:
                        draw_line(im_depth, pt_depth_direct[0:2].transpose(), pt_prev_direct, color=[250, 120, 100], thickness=thickness)
                    pt_prev_direct = np.copy(pt_depth_direct[0:2].transpose())


            # Draw the projected 3d curves
            if self.pose_matrices is not [] and self.crv_pt_locations_3d is not []:
                for crv_indx in range(0, len(self.crv_pt_locations_3d)):
                    crv_pts_3d = np.array(self.crv_pt_locations_3d[crv_indx], dtype=np.float64)
                    crv_pts_proj = cv2.projectPoints(crv_pts_3d,
                                                     rvec=self.rot_matrices[count],
                                                     tvec=self.trans_vecs[count],
                                                     cameraMatrix=self.depth_camera.world_to_image,
                                                     distCoeffs=self.depth_camera.image_distortion_coefs)
                    for pt_indx in range(0, crv_pts_3d.shape[0]):
                        pt = crv_pts_proj[0][pt_indx, 0, :]
                        draw_cross(im_depth, pt, color=[20, 220, 200], thickness=thickness)
                        if pt_indx == 0:
                            pt_prev = pt
                        else:
                            pt_prev = crv_pts_proj[0][pt_indx - 1, 0, :]
                        # Red - Project the 3D point to the image using the depth camera information
                        draw_line(im_depth, pt, pt_prev, color=[20, 20, 200], thickness=thickness // 2)


            # Draw the projected background points
            if self.pose_matrices is not [] and self.pt_locations_3d is not []:
                pts_proj = cv2.projectPoints(pts_3d,
                                             rvec=self.rot_matrices[count],
                                             tvec=self.trans_vecs[count],
                                             cameraMatrix=self.depth_camera.world_to_image,
                                             distCoeffs=self.depth_camera.image_distortion_coefs)
                for pt_indx in range(0, pts_proj[0].shape[0]):
                    pt_row = pts_proj[0][pt_indx]
                    draw_box(im_depth, pt_row, color=[100, 150, 200], width=thickness * 2)

            count = count + 1
            fname = va.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=True, b_add_tag=False) + "_pts2d.png"
            cv2.imwrite(fname, im_depth)



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
    pt_kf = PointTrackerKeyFrames(va_data=va, rgb_camera=cam_rgb, depth_camera=cam_depth)
    pts_2d, vecs, valid_kfs = pt_kf.collect_background_points(b_is_horizontal=True)
    pts_2d_crvs = pt_kf.collect_crv_points(vecs, valid_kfs)
    # 10.16cm is 4 inches
    pt_kf.solve_3d_pts(seq_length=10.16, iters=6)
    pt_kf.debug_images(va)
    pt_kf.est_depth = -pt_kf.crv_pt_locations_3d[0][1][2]
    pt_kf.solve_3d_pts(seq_length=10.16, iters=6)
    pt_kf.debug_images(va)
    pt_kf.point_depth_from_image()
    pt_kf.crv_point_depth_from_image()

    print(f"Done")
