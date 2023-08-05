"""
get cutout from depth image
    - get xyz for cross section (opencv point cloud), gets diameter of cylinder
    - can also take all depth values for part of cylinder in area, get mask,
      do triangulation (do along multiple parts of the curve)
    
    - meshlab from `bezier_cyl_3d`
    - Look in Josyula's code for (mesh, camera matrix, image size) -> get rendered 2D image
        - pymesh/trimesh (maybe even blender/opencv)

        - make seperate rendering method for debug


"""

import numpy as np
import cv2
import os
from fit_bezier_cyl_2d_edge import FitBezierCyl2DEdge
from bezier_cyl_3d import BezierCyl3D
from HandleFileNames import HandleFileNames


class FitBezierCyl3DDepth:
    def __init__(
        self,
        fname_rgb_image: str,
        fname_edge_image: str,
        fname_mask_image: str,
        fname_depth_image: str,
        fname_calculated: str = None,
        fname_debug: str = None,
        b_recalc: bool = False,
    ) -> None:
        """Read in the rgb image to generate quad fit, then read in the depth data
        to get start and end radii.
        @param fname_rgb_image: RGB image file name
        @param fname_edge_image: Edge image file name
        @param fname_mask_image: Mask image file name
        @param fname_depth_image: Depth image file name
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _crv.json
        @param fname_debug: the file name for a debut image showing the bounding box, etc
        @param b-recalc: Force recalculate the result, y/n
        """

        # Get the base edge image
        self.edge_crv = FitBezierCyl2DEdge(
            fname_rgb_image=fname_rgb_image,
            fname_edge_image=fname_edge_image,
            fname_mask_image=fname_mask_image,
            fname_calculated=fname_calculated,
            fname_debug=fname_debug,
            b_recalc=b_recalc,
        )

        # Get the calculated bezier curve from FitBezierCyl2DEdge
        self.bezier_crv = self.edge_crv.bezier_crv_fit_to_edge
        print(self.bezier_crv.__dict__)

        # Reverse depth image data TODO: change to Josyula's flow
        self.depth_im_mask = cv2.imread(fname_depth_image)
        self.depth_im = self._reverse_depth_mask(self.depth_im_mask)

        # Estimate the depth data
        self._estimate_bezier_crv_depths()

        # Get depth data at each point to add `z` dimension
        self.bezier_crv.p0 = np.concatenate((self.bezier_crv.p0, np.array([0])))
        self.bezier_crv.p1 = np.concatenate((self.bezier_crv.p1, np.array([0])))
        self.bezier_crv.p2 = np.concatenate((self.bezier_crv.p2, np.array([0])))

        # Create BezierCyl3D
        self.bezier_cyl_3d = BezierCyl3D(
            p1=self.bezier_crv.p0,
            p2=self.bezier_crv.p1,
            p3=self.bezier_crv.p2,
            start_radius=self.bezier_crv.start_radius,
            end_radius=self.bezier_crv.end_radius,
        )

        # Save mesh
        __here__ = os.path.dirname(__file__)
        mesh_name = os.path.basename(fname_rgb_image).strip(".png")
        print(mesh_name)
        self.bezier_cyl_3d.write_mesh(f"{__here__}/data/meshes/{mesh_name}.obj")

        return
    

    def _reverse_depth_mask(self, depth_image_rgb):
        # original:
        #       depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # convertScaleAbs(): takes values and returns a single 8-bit value
        #       dst(I) = saturate\_cast<uchar>(|src(I)*alpha + beta|)
        
        

        return

    def _estimate_bezier_crv_depths(self):
        # Triangulation..
        # vertical branches: 3/4 - 1.5 in thick
        # camera ~3/4 - 1.5m away
        camera_dist = 1.0 # m

        
        return

def main():
    # import pymesh
    import trimesh
    import os
    import glob

    __here__ = os.path.dirname(__file__)

    path_bpd = f"{__here__}/data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = False
    for ind in all_files.loop_masks():
        rgb_fname = all_files.get_image_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_add_tag=True)
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        depth_fname = all_files.get_depth_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            edge_fname_debug = ""
        else:
            edge_fname_debug = edge_fname_debug + "_crv_edge.png"

        edge_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not os.path.exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        if not os.path.exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")
        if not os.path.exists(depth_fname):
            raise FileNotFoundError

        FitBezierCyl3DDepth(
            rgb_fname, edge_fname, mask_fname, depth_fname, edge_fname_calculate, edge_fname_debug, b_recalc=b_do_recalc
        )

        input()
    print("foo")
    return


if __name__ == "__main__":
    main()
