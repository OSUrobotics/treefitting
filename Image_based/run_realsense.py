import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from line_seg_2d import draw_line, draw_box, draw_cross, LineSeg2D
import cv2
import json
import time
import os
import glob

class RunRealSense():
    def __init__(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Read in the image, mask image, flow image, 2 rgb images
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @param b_recalc: Force recalculate the result, y/n"""

        self.path_debug = path + "DebugImages/"
        path_calculated = path + "CalculatedData/"

        self.name = image_name
        # Read in all images that have name_ and are not debugging images
        self.read_images(path, image_name)
        # Split the mask into connected components, each of which might be a vertical leader
        return
    

    def read_images(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Read in all of the mask, rgb, flow images
        If Edge image does not exist, create it
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @returns dictionary of images with image_type as keywords """

        images = {}
        # search_path = f"{path}{image_name}_*.npy"
        # fnames = glob(search_path)
        # if fnames is None:
        #     raise ValueError(f"No files in directory {search_path}")
        # Setup:
        f2 = '/home/josyula/Documents/DataAndModels/3AprilTrees_D435i/20230403_145650.bag'
        path_calculated = path + "CalculatedData/"

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(f2)
        profile = pipe.start(cfg)
        branch = MakeTreeGeometry("data")
        radmaxs = []
        radmins = []
        rad_picos = []
        radius_from_ransac = []
        rad_avgs = []
        for i in range(1000):
            #time single frame

            time_start = time.time()
            frameset = pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            color = np.asanyarray(color_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            ts = frameset.get_timestamp()
            image_name = str(ts).split('.')[0]
            print(image_name)
            # if image_name == str(1680559045964):

            colorizer = rs.colorizer()
            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            # Update color and depth frames:
            # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            path_temp = tempfile.mkdtemp("masks")
            search_path = f"{path}{image_name}_*.npy"
            fnames = glob.glob(search_path)
            i=0
            for n in fnames:
                images[image_name+"_np_mask"+"_{0}".format(i)] = np.load(n)
                images["RGB0"] = color
                images[image_name+"_np_mask"+"_{0}".format(i)] = self.depth_mask(images["RGB0"], images[image_name+"_np_mask"+"_{0}".format(i)], depth_frame, intrinsics)
                cv2.imwrite(path+"/RGB"+"/{0}.jpg".format(image_name), images["RGB0"])
                cv2.imwrite(path_temp+"/mask.jpg", images[image_name+"_np_mask"+"_{0}".format(i)]*255)
                images["Mask"] = cv2.imread(path_temp+"/mask.jpg")
                images["Mask"] = cv2.cvtColor(images["Mask"], cv2.COLOR_BGR2GRAY)
                images["Edge"] = cv2.Canny(images["Mask"], 50, 150, apertureSize=3)
                # bound_x , bound_y = np.where(images[image_name+"_np_mask"+"_{0}".format(i)].astype(int)==1) # np.where(images["Edge"] == 255)
                # bound_low_x1 = min(bound_y)
                # bound_low_x2 = min(bound_y)
                # bound_low_y1 = min(bound_x[bound_y==bound_low_x1])
                # bound_low_y2 = max(bound_x[bound_y==bound_low_x2])
                #
                # bound_high_x1 = max(bound_y)
                # bound_high_x2 = max(bound_y)
                # bound_high_y1 = min(bound_x[bound_y==bound_high_x1])
                # bound_high_y2 = max(bound_x[bound_y==bound_high_x2])

                cv2.imwrite(path + image_name + "_edge.png", images["Edge"])
                self.images = images
                self.name = image_name
                self.vertical_leader_masks = self.split_mask(self.images["Mask"], b_one_mask=True,
                                                            b_debug=b_output_debug)
                self.vertical_leader_stats = []
                self.vertical_leader_quads = []
                #extract 3d points for all points in the depth_mask


                all_3d_points = []
                for k in range(images[image_name+"_np_mask"+"_{0}".format(i)].shape[0]):
                    for l in range(images[image_name+"_np_mask"+"_{0}".format(i)].shape[1]):
                        if images[image_name+"_np_mask"+"_{0}".format(i)][k,l] == 1:
                            depth = depth_frame.get_distance(l,k)
                            if depth != 0:
                                pixel = rs.rs2_deproject_pixel_to_point(intrinsics, [l,k], depth)
                                all_3d_points.append(pixel)
                all_3d_points = np.array(all_3d_points)

                # For each component of the mask image, calculate or read in statistics (center, eigen vectors)
                print("Calculating stats")
                for j, mask in enumerate(self.vertical_leader_masks):
                    fname_stats = path_calculated + self.name + f"_mask_{j}.json"
                    if b_recalc or not os.path.exists(fname_stats):
                        stats_dict = self.stats_image(self.images["Mask"], mask)
                        for k, v in stats_dict.items():
                            try:
                                if v.size == 2:
                                    stats_dict[k] = [v[0], v[1]]
                            except:
                                pass
                        # If this fails, make a CalculatedData and DebugImages folder in the data/forcindy folder
                        with open(fname_stats, 'w') as f:
                            json.dump(stats_dict, f)
                    elif os.path.exists(fname_stats):
                        with open(fname_stats, 'r') as f:
                            stats_dict = json.load(f)

                    for k, v in stats_dict.items():
                        try:
                            if len(v) == 2:
                                stats_dict[k] = np.array([v[0], v[1]])
                        except:
                            pass
                    self.vertical_leader_stats.append(stats_dict)

                # For each of the masks, see if we have reasonable stats
                #   Save points in debug image
                if b_output_debug:
                    self.images["Mask_Stats"] = np.copy(self.images["Mask"])
                    self.images["RGB_Stats"] = np.copy(self.images["RGB0"])
                    for i, stats in enumerate(self.vertical_leader_stats):
                        self.images["Mask_Stats"] = self.images["Mask"] / 2
                        try:
                            p1 = stats["lower_left"]
                            p2 = stats["upper_right"]
                            self.images["Mask_Stats"][self.vertical_leader_masks[i]] = 255
                            draw_line(self.images["RGB_Stats"], p1, p2, (128, 128, 128), 2)
                            draw_line(self.images["Mask_Stats"], p1, p2, (128, 128, 128), 1)

                            pc = ["center"]
                            draw_cross(self.images["RGB_Stats"], pc, (128, 128, 128), 1, 2)
                            draw_cross(self.images["Mask_Stats"], pc, (180, 180, 128), 1, 3)

                        except:
                            pass

                        cv2.imwrite(self.path_debug + image_name + "_" + f"{i}_mask_points.png",
                                    self.images["Mask_Stats"])
                    # cv2.imwrite(self.path_debug + image_name + "_" + "rgb_points.png", self.images["RGB_Stats"])

                # Fit a quad to each vertical leader
                print("Fitting quads")
                for j, stats in enumerate(self.vertical_leader_stats):
                    print(f"  {image_name}, mask {j}")
                    image_mask = np.zeros(self.images["Mask"].shape, dtype=self.images["Mask"].dtype)
                    fname_quad = path_calculated + self.name + "_" + image_name + f"_{j}_quad.json"
                    fname_params = path_calculated + self.name + "_" + image_name + f"_{j}_quad_params.json"
                    quad = None
                    if os.path.exists(fname_quad) and not b_recalc:
                        quad = Quad([0, 0], [1, 1], 1)
                        quad.read_json(fname_quad)
                        with open(fname_params, 'r') as f:
                            params = json.load(f)
                    else:
                        image_mask[self.vertical_leader_masks[i]] = 255
                        quad, params = self.fit_quad(image_mask, pts=stats, b_output_debug=b_output_debug, quad_name=i)
                        quad.write_json(fname_quad)
                        with open(fname_params, 'w') as f:
                            json.dump(params, f)
                    self.vertical_leader_quads.append(quad)

                    if b_output_debug:
                        # Draw the edge and original image with the fitted quad and rects
                        im_covert_back = cv2.cvtColor(self.images["Edge"], cv2.COLOR_GRAY2RGB)
                        im_orig_debug = np.copy(self.images["RGB0"])

                        # Draw the original, the edges, and the depth mask with the fitted quad
                        quad.draw_quad(im_orig_debug)
                        if quad.is_wire():
                            draw_cross(im_orig_debug, quad.p0, (255, 0, 0), thickness=2, length=10)
                            draw_cross(im_orig_debug, quad.p2, (255, 0, 0), thickness=2, length=10)
                        else:
                            quad.draw_boundary(im_orig_debug, 10)
                            quad.draw_edge_rects(im_covert_back, step_size=params["step_size"],
                                                perc_width=params["width"])

                        im_both = np.hstack([im_orig_debug, im_covert_back])
                        cv2.imshow("Original and edge and depth", im_both)
                        cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_quad.png", im_both)

                        quad.draw_quad(self.images["RGB_Stats"])


                    ## ransac circle fit
                    r_ransac = self.fit_circle_to_3d_points(all_3d_points, quad, intrinsics, depth_frame)
                    radius_from_ransac.append(r_ransac)

                    ###average radius fit###
                    test = []
                    test_l = []
                    test_r = []
                    for t in np.arange(0, 1, 0.1 ):
                        pix1 = np.ceil(quad.pt_axis(t)).clip(0, max=np.array([720,1280])).astype(int)
                        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, pix1, depth_frame.get_distance(*pix1))
                        # print(point1)
                        pix_edge = np.ceil(quad.edge_pts(t)).clip(0, max=np.array([720,1280])).astype(int)
                        pixl = [pix_edge[0][0], pix_edge[0][1]]
                        pixr = [pix_edge[1][0], pix_edge[1][1]]
                        pointl = rs.rs2_deproject_pixel_to_point(intrinsics, pixl, depth_frame.get_distance(*pixl))
                        pointr = rs.rs2_deproject_pixel_to_point(intrinsics, pixr, depth_frame.get_distance(*pixr))
                        test_l.append(pointl)
                        test_r.append(pointr)
                        test.append(point1)

                        if any(np.isnan(point1)):
                            break
                    if len(test) != 10:
                        break
                # for quad in self.vertical_leader_quads:
                #     score = self.score_quad(quad)
                #     print(f"Score {score}")
                # get indices on the ledft and right test_l, test_r where the points are not zero
                test_l = np.array(test_l)
                test_r = np.array(test_r)
                test = np.array(test)
                #avearage radius along the left and right edges
                rad_avg = np.linalg.norm(test_l - test_r)/len(test_l)
                print("along branch ", rad_avg)
                rad_avgs.append(rad_avg)
                # rad_max = rad_min
                ind1 = np.where(test_l[:, 0] != 0)
                ind2 = np.where(test_r[:, 0] != 0)
                set_ind = set(ind1[0]).intersection(set(ind2[0]))
                ind = np.array(list(set_ind))
                rad_min = np.linalg.norm(test_l[ind[0]] - test_r[ind[0]])
                rad_max = np.linalg.norm(test_l[ind[-1]] - test_r[ind[-1]])
                points_on_axis = [test[0], test[int(len(test)/2)], test[-1]]
                ##alternate radius calculation
                #3d coordinates of bound_low
                mask_b = images[image_name + "_np_mask" + "_{0}".format(i)]
                top_left, top_right, bottom_left, bottom_right = self.find_four_corners_v1(mask_b)
                # top_left, top_right, bottom_left, bottom_right = self.find_four_corners_v2(mask_b, quad)
                try:
                    bound_low_left = rs.rs2_deproject_pixel_to_point(intrinsics, [bottom_left[0], bottom_left[1]], depth_frame.get_distance(bottom_left[0], bottom_left[1]))
                    bound_low_right = rs.rs2_deproject_pixel_to_point(intrinsics, [bottom_right[0], bottom_right[1]], depth_frame.get_distance(bottom_right[0], bottom_right[1]))
                    bound_high_left = rs.rs2_deproject_pixel_to_point(intrinsics, [top_left[0], top_left[1]], depth_frame.get_distance( top_left[0], top_left[1]))
                    bound_high_right = rs.rs2_deproject_pixel_to_point(intrinsics, [top_right[0], top_right[1]], depth_frame.get_distance(top_right[0], top_right[1]))

                    rad_min_alt = np.linalg.norm(np.array(bound_low_left) - np.array(bound_low_right))
                    rad_max_alt = np.linalg.norm(np.array(bound_high_left) - np.array(bound_high_right))
                    rad_min = min(rad_min, rad_min_alt)
                    rad_max = min(rad_max, rad_max_alt)
                    print(f"rad_min {rad_min}, rad_max {rad_max}")
                    radmins.append(rad_min)
                    radmaxs.append(rad_max)
                    depths = self.get_depths(images["RGB0"], depth_frame)

                    ## pico radius##
                    tw = TrunkWidthEstimator()
                    rad_pico = tw.get_width(images["RGB0"], depths, images[image_name + "_np_mask" + "_{0}".format(i)] )
                    print(f"rad_pico {rad_pico}")
                    rad_picos.append(rad_pico)
                    #WRITE BRANCH TO FILE
                    branch.make_branch_segment(points_on_axis[0], points_on_axis[1],
                                            points_on_axis[2],
                                            radius_start=rad_min, radius_end=rad_max,
                                            start_is_junction=True, end_is_bud=False)
                    vertex_locs = branch.write_mesh(self.path_debug + "_" + image_name + f"_{i}_branch.obj")
                    #project the vertices onto the image and draw them
                    vertex_2d = []
                    for v in vertex_locs:
                        for v1 in v:
                            pix = rs.rs2_project_point_to_pixel(intrinsics, v1)
                            pix = np.array(pix).clip(0, max=np.array([720,1280])).astype(int)
                            cv2.circle(self.images["RGB_Stats"], (int(pix[0]), int(pix[1])), 5, (0, 0, 255), -1)
                            vertex_2d.append([int(pix[0]), int(pix[1])])
                    # cv2.imshow("RGB_Stats", self.images["RGB_Stats"])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_branch.png", self.images["RGB_Stats"])
                    os.remove(path_temp+"/mask.jpg")
                    #MAKE ANNONATION
                    # self.make_image_annotation(vertex_2d, image_name, path, class_id=i)
                    i += 1
                    time_end= time.time()
                    print(f"branch {i} took {time_end - time_start} seconds")
                except:
                    import sys
                    print("Error in branch detection - no depth data for branch ", image_name)
                    print("error type: ", sys.exc_info()[0])
                    # os.remove(path_temp+"/mask.jpg")
                    continue
                time_end = time.time()
                print(f"branch {i} took {time_end - time_start} seconds")
        pipe.stop()
        radmins = np.array(radmins)
        radmaxs = np.array(radmaxs)
        rad_avgs = np.array(rad_avgs)

        rad_picos = np.array(rad_picos)
        avg_rad_min = np.mean(radmins)
        avg_rad_max = np.mean(radmaxs)
        avg_rad_pico = np.mean(rad_picos)
        print("along the branch avg ", rad_avgs.mean())
        print(f"avg_rad_min {avg_rad_min}, avg_rad_max {avg_rad_max}")
        np.save(self.path_debug + "n_radmins.npy", radmins)
        np.save(self.path_debug + "n_radmaxs.npy", radmaxs)
        np.save(self.path_debug + "n_rad_picos.npy", rad_picos)
        radactual = 15.34
        #plot error bar with radmin
        import seaborn as sns
        import pandas as pd
        #make dataframe
        df = pd.DataFrame({'radmin': radmins, 'radmax': radmaxs, 'rad_pico': rad_picos})
        sns.set()
        #histogram of radmin
        sns.distplot(df['radmin'], bins=3, kde=False, rug=True)

        plt.hist(radmins, bins=3, color='b', label='radmin')
        plt.errorbar(np.arange(len(radmins)), radmins, yerr=radactual-radmins, fmt='o')

        #plot bar graph of radii vs actual radii
        plt.bar(np.arange(len(radmins)), radmins, width=0.2, color='b', align='center', label='radmin')
        #plot histogram of radii
        plt.hist(radmins, bins=10, color='b', label='radmin')
        #plot actual as x-axis
        plt.plot([0, len(radmins)], [radactual, radactual], color='r', label='actual')
        return images