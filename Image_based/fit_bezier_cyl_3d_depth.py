"""
get cutout from depth image
    - get xyz for cross section (opencv point cloud), gets diameter of cylinder
    - can also take all depth values for part of cylinder in area, get mask, do triangulation (do along multiple parts of the curve)
    
    - meshlab from `bezier_cyl_3d`
    - Look in Josyula's code for (mesh, camera matrix, image size) -> get rendered 2D image
        - pymesh/trimesh (maybe even blender/opencv)

        - make seperate rendering method for debug


"""

import numpy as np
from bezier_cyl_3d import BezierCyl3D
from HandleFileNames import HandleFileNames


class FitBezierCyl3DDepth:
    def __init__(self) -> None:
        

        return
    
    @staticmethod
    def create_bezier_crv_from_eigen_vectors(stats) -> BezierCyl3D:

        return
    
    def score_mask_fit(self, im_mask):
        return