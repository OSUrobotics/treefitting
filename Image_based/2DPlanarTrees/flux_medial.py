#!/usr/bin/env python3
import numpy as np
import imageio
import math
from utils.file_names_sub_dirs import FileNamesSubDirs
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line


def get_hough(im_in, thetas):
    im = np.flip(im_in.transpose(), axis=0)
    h, theta, d = hough_line(im, theta=thetas)

    _, angles, dists = hough_line_peaks(h, theta, d)

    ret_ang = []
    for a, d in zip(angles, dists):
        x0 = np.float64(d * np.cos(a))
        y0 = np.float64(d * np.sin(a))
        slope = np.tan(a + np.pi/2.0)
        # axs.axline((x0, y0), slope=slope)
        b = y0 - slope * x0
        x1 = x0 + 1
        y1 = slope * x1 + b
        ang = -np.atan2(y1 - y0, x1 - x0)
        ret_ang.append(- (a + np.pi/2.0))
        # axs.plot([x0, x1], [y0, y1], '-b')
        # print(f"  ({x0}, {y0}), ({x1}, {y1}) slope {slope} ang = {a}, {ang}")
        #ret_ang.append(a + np.pi / 2.0)
    return ret_ang

def check_hough():
    import matplotlib.pyplot as plt
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    fig, axes_all = plt.subplots(4, 4)
    for indx, ang in enumerate(np.linspace(-np.pi / 2.0, np.pi / 2.0, 16)):
        axs = axes_all[indx // 4, indx % 4]
        im = np.zeros((20, 20, 3), dtype=np.uint8)
        for d in np.linspace(-7.0, 7.0, 20):
            x = 10.0 + d * np.cos(ang)
            y = 10.0 + d * np.sin(ang)
            im[int(x), int(y), :] = 255

        axs.imshow(im)
        ang_back = get_hough(im[:, :, 0], tested_angles)[0]
        print(f" ang {ang} {ang_back} {ang_back + np.pi} {ang_back - np.pi}")
        xn = 10.0 * np.cos(ang_back)
        yn = 10.0 * np.sin(ang_back)
        axs.plot([10.0, xn], [10.0, yn], '-r')
        axs.set_aspect('equal')
    plt.show()





def get_flow(pix):
    ang = pix[0]
    dx = pix[2] * np.cos(np.deg2rad(ang))
    dy = pix[2] * np.sin(np.deg2rad(ang))
    return np.sqrt(dx * dx + dy * dy), ang


def find_points_ma(im_ma, im_flow, im_dist):
    pts_where = np.where(im_ma[:,:] < 10)
    im_inv = 255 - im_ma
    pts_i = pts_where[0]
    pts_j = pts_where[1]

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    labels = np.zeros((im_ma.shape[0], im_ma.shape[1], 3))
    edges = {}
    junctions = {}
    points = {}

    # Find the longest line that can be fit to the pixel
    count_no_edge = 0
    for pt in zip(pts_i, pts_j):
        ix = int(pt[0])
        iy = int(pt[1])
        b_no_edges = True
        for lindx, w in enumerate([5, 10, 20]):
            ix = max(w, ix)
            ix = min(im_ma.shape[0] - w - 1, ix)
            iy = max(w, iy)
            iy = min(im_inv.shape[1] - w - 1, iy)
            angles = get_hough(im_inv[ix-w:ix+w+1, iy-w: iy+w+1], thetas=tested_angles)
            # h, theta, d = hough_line(im_inv[ix-w:ix+w+1, iy-w: iy+w+1], theta=tested_angles)
            # _, angles, dists = hough_line_peaks(h, theta, d)
            if len(angles) > 0:
                if not b_no_edges and len(angles) > 1:
                    continue

                b_no_edges = False
                angle = angles[0]
                # (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
                e0x = pt[0] + w * np.cos(angle)
                e0y = pt[1] + w * np.sin(angle)
                e1x = pt[0] + w * np.cos(angle + np.pi)
                e1y = pt[1] + w * np.sin(angle + np.pi)
                pt_mid = [0.5 * e0x + 0.5 * e1x, 0.5 * e0y + 0.5 * e1y]
                if not np.isclose(abs(pt_mid[0] - pt[0]) + abs(pt_mid[1] - pt[1]), 0.0):
                    print('Mismatch {pt_mid} {pt}')

                # slope_hough = np.tan(angle + np.pi / 2)
                # my_slope = (e1y - e0y) / (e1x - e0x)
                # my_atan = np.atan2( (e1y - e0y), (e1x - e0x) )
                edges[pt] = [(e0x, e0y), (e1x, e1y), angle, w]
                labels[pt[0], pt[1], 0:2] = 50 + lindx * 50
                points[pt] = (pt, 'Edge')
            if len(angles) == 2:
                if abs(angles[0] - angles[1]) > np.pi / 4:
                    junctions[pt] = (angles[0:2], w)

                labels[pt[0], pt[1], 1] = lindx * 50
                points[pt] = (pt, 'TJunc')
            elif len(angles) > 2:
                if abs(angles[0] - angles[1]) > np.pi / 4:
                    junctions[pt] = (angles[0:2], w)

                labels[pt[0], pt[1], 1] = lindx * 50
                labels[pt[0], pt[1], 2] = lindx * 50
                points[pt] = (pt, 'YJunc')
                #ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
        if b_no_edges:
            count_no_edge += 1

    print(f"No edges {count_no_edge}")
    points_all_data = {}
    for pt_key, item in points.items():
        dist_val = im_dist[pt_key[0], pt_key[1]]
        of_val, _ = get_flow(im_flow[pt_key[0], pt_key[1], :])
        points_all_data[pt_key] = (item, dist_val, of_val)

    return points_all_data, edges, junctions, labels


def find_long_curves(points, edges, junctions, labels):
    visited = np.zeros((labels.shape[0], labels.shape[1], 3))
    chains = []
    pix_in_chain = 255
    pix_visited_no_chain = 145
    pix_in_edge_and_search = 125
    pix_in_search_no_edge = 100
    for e_k, e_i in edges.items():
        chain = [e_k]
        e_k = (int(e_k[0]), int(e_k[1]))
        n_type, cur_dist, cur_of = points[e_k]
        if visited[e_k[0], e_k[1], 0] == pix_in_chain:
            continue

        visited[e_k[0], e_k[1], :] = pix_visited_no_chain
        done = False
        # Last point we connected to at either end
        pt_left_last = e_k
        pt_right_last = e_k
        # Current end point
        pt_left = e_i[0]
        pt_right = e_i[1]
        ang_left = e_i[2]
        ang_right = e_i[2] + np.pi
        if ang_right > np.pi:
            ang_right -= 2.0 * np.pi
        ang_out_left = np.atan2(pt_left[1] - pt_left_last[1], pt_left[0] - pt_left_last[0])
        ang_out_right = np.atan2(pt_right[1] - pt_right_last[1], pt_right[0] - pt_right_last[0])

        print(f"Starting chain: {e_k}, ang {e_i[2]}, dist {e_i[3]}")
        n_pix = 15
        while not done:
            # Keep trying to extend
            done = True
            for indx, (pt, pt_last, ang) in enumerate(zip((pt_left, pt_right), (pt_left_last, pt_right_last), (ang_left, ang_right))):
                if indx == 0:
                    print(f" Left {pt} {ang}")
                else:
                    print(f" right {pt} {ang}")

                # Try both ends - same code, basically
                dist_next = 0.0
                pt_next = (-1, -1)
                ang_next = ang
                ang_diff_next = np.pi
                vec_pix_centers = np.array([pt[0] - pt_last[0], pt[1] - pt_last[1]]).astype(np.float64)
                dist_vec_pix_centers = np.linalg.norm(vec_pix_centers)
                if dist_vec_pix_centers > 0:
                    vec_pix_centers /= dist_vec_pix_centers
                else:
                    print(f"Warning: Pix centers the same {pt} {pt_next}")
                vec_ang = np.array([np.cos(ang), np.sin(ang)])

                dot_pix_center_vec_ang = np.dot(vec_pix_centers, vec_ang)
                if dot_pix_center_vec_ang < 1.0 - np.pi / 4.0:
                    print(f"Warning, bad edge {e_k}, {ang}, {np.atan2(vec_pix_centers[1], vec_pix_centers[0])} {pt_last}, {pt}")

                # First get all the pixels that are along this ray
                pix_ray = {}
                # visited[visited[:, :, 0] < pix_visited_no_chain] = 0
                for d in range(1, n_pix+1):
                    box_w = max(1, np.abs(d) // 3)
                    for rw in range(-box_w, box_w+1):
                        for cw in range(-box_w, box_w+1):
                            pt_check = (np.int64(rw + pt[0] + d * vec_ang[0]), np.int64(cw + pt[1] + d * vec_ang[1]))
                            if pt_check in edges:
                                if pt_check[0] == pt[0] and pt_check[1] == pt[1]:
                                    continue
                                if visited[pt_check[0], pt_check[1], 0] == pix_in_chain:
                                    continue
                                pix_ray[pt_check] = (d, box_w)
                                visited[pt_check[0], pt_check[1], 0:2] = pix_in_edge_and_search
                            else:
                                if pt_check[0] < 0 or pt_check[1] < 0:
                                    continue
                                if pt_check[0] >= visited.shape[0] or pt_check[1] >= visited.shape[1]:
                                        continue
                                visited[pt_check[0], pt_check[1], 0] = pix_in_search_no_edge
                                visited[pt_check[0], pt_check[1], 1] = 255
                                visited[pt_check[0], pt_check[1], 2] = pix_in_search_no_edge

                blocked = []
                for pt_check_k, pt_check_items in pix_ray.items():
                    vec_next_edge = (np.array(pt_check_k) - np.array(pt)).astype(np.float64)
                    vec_next_edge /= np.linalg.norm(vec_next_edge)
                    next_edge = edges[pt_check_k]
                    vec_ang_edge = np.array([np.cos(next_edge[2]), np.sin(next_edge[2])])
                    dot_ang_ang = np.dot(vec_ang_edge, vec_ang)
                    dot_ang_pt_centers = np.dot(vec_next_edge, vec_ang)
                    ang_error = 1.0 - np.fabs(dot_ang_ang) + 1.0 - np.fabs(dot_ang_pt_centers)
                    if ang_error > np.pi / 8.0:
                        print(f" bad {next_edge[2]}, {ang}, {np.atan2(vec_next_edge[1], vec_next_edge[0])}")
                        continue

                    blocked.append((pt_check_k, pt_check_items[0]))
                    if ang_error <= (ang_diff_next + np.pi / 8.0) and pt_check_items[0] >= dist_next:
                        b_replace = True
                        if dist_next == pt_check_items[0]:
                            if ang_error < ang_diff_next:
                                b_replace = True
                            else:
                                b_replace = False
                        if b_replace:
                            ang_diff_next = ang_error
                            dist_next = pt_check_items[0]
                            pt_next = pt_check_k
                            ang_next = next_edge[2]
                            if np.dot(vec_next_edge, vec_ang) < 0.0:
                                ang_next *= -1
                            print(f"   NE ang {ang_next}, ang {ang} diff {ang_error} d {pt_check_items[0]}")

                print(f" Found {len(pix_ray)}, blocked {len(blocked)}")
                if pix_ray == {}:
                    continue

                if pt_next[0] == -1:
                    continue

                done = False
                visited[e_k[0], e_k[1], :] = pix_in_chain
                visited[pt_next[0], pt_next[1], :] = pix_in_chain
                for pt_check, pt_d in blocked:
                    if pt_d <= dist_next:
                        visited[pt_check[0], pt_check[1], :] = pix_in_chain

                if indx == 0:
                    chain.insert(0, pt_next)
                    pt_left_last = pt_left
                    pt_left = pt_next
                    ang_left = ang_next
                    print(f" next left {pt_next} ang {ang_left}")
                else:
                    chain.append(pt_next)
                    pt_right_last = pt_right
                    pt_right = pt_next
                    ang_right = ang_next
                    print(f" next right {pt_next} ang {ang_right}")

        if len(chain) > 1:
            chains.append(chain)
            imageio.imwrite("chains.png", visited.astype(np.uint8))
    return chains


def label_chains(pts, chains, im_dist, im_flow):
    chain_labels = []

    flow_all = []
    for pt in pts:
        mag, ang = get_flow(im_flow[pt[0], pt[1], :])
        flow_all.append(mag)
    flow_all - np.sort(np.array(flow_all))

    flow_min = flow_all[int(0.1 * len(flow_all))]
    flow_max = flow_all[int(0.9 * len(flow_all))]

    ts = np.linspace(0, 1.0, 25)
    for chain in chains:
        dist_along_chain = []
        flow_ang_along_chain = []
        flow_mag_along_chain = []
        len_chain = 0.0
        for pt1, pt2 in zip(chain[0:-1], chain[1:]):
            vec = [(pt2[0] - pt1[0]), (pt2[1] - pt1[1])]
            len_vec = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            len_chain += len_vec
            n_pix = int(len_vec) + 1
            ts = np.linspace(0.0, 1.0, max(2, n_pix))
            for t in ts[0:-1]:
                pt = (int(pt1[0] + t * vec[0]), int(pt1[1] + t * vec[1]))
                d = im_dist[pt[0], pt[1]]
                flow = im_flow[pt[0], pt[1], :]
                dist_along_chain.append(d)
                of_mag, of_ang = get_flow(flow)
                flow_ang_along_chain.append(of_ang)
                flow_mag_along_chain.append(of_mag)

        col = [255, 255, 255]
        flow_mag_along_chain = np.sort(np.array(flow_mag_along_chain))
        flow_avg = np.mean(flow_mag_along_chain)
        if len_chain < 10:
            col = [200, 125, 125]
            chain_labels.append("short")
        else:
            dist_along_chain = np.sort(np.array(dist_along_chain))
            d_avg = np.mean(dist_along_chain)

            indx_min = int(0.1 * np.size(dist_along_chain))
            indx_max = int(0.9 * np.size(dist_along_chain))
            d_spread = dist_along_chain[indx_max] - dist_along_chain[indx_min]
            if d_spread > d_avg * 1.1:
                print(f" spread {d_spread} {d_avg}, {len_chain}")
                col = [255, 125, 255]
                chain_labels.append("background")
            else:
                if flow_avg > 0.8 * flow_max:
                    col = [50, 255, 125]
                    chain_labels.append("both")
                else:
                    col = [255, 255, 255]
                    chain_labels.append("tree")

    for lab in ["short", "background", "both", "tree"]:
        im_chain_labels = np.copy(im_flow)
        col = [0, 0, 0]
        col = np.array(col).transpose()
        for indx, chain in enumerate(chains):
            if not chain_labels[indx] == lab:
                continue
            for pt1, pt2 in zip(chain[0:-1], chain[1:]):
                for t in ts:
                    pt = t * np.array(pt1) + (1-t) * np.array(pt2)
                    im_chain_labels[int(pt[0]), int(pt[1]), :] = col
        imageio.imwrite("chain_labels_" + lab + ".png", im_chain_labels.astype(np.uint8))
    return chain_labels


