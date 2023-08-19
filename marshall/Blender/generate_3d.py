import bpy
import bmesh
import os
import numpy as np
import camera as c
import params as p
from random import randint
from mathutils import Vector, Matrix, Euler
from math import radians
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from curve_fitting import BezierBasedDetection, Bezier
from bpy_extras.io_utils import unpack_list



class Generate_3D:   
    def __init__(self, full_arr, x, y, z, num_pix):
        self.full_arr = full_arr
        self.rad = []
        self.y = y
        self.x = x
        self.num_pix = num_pix
        self.z = z
        
    def create_mesh(self):
        self.set_y_x_rad()
        
        #bpy.data.objects.remove(bpy.data.objects['Cube'])

        crv = bpy.data.curves.new('crv', 'CURVE')
        crv.dimensions = '3D'
        crv.fill_mode = 'FULL'
        crv.bevel_depth = 0.045
        crv.resolution_u = 1 
        obj = bpy.data.objects.new('crv', crv)
        bpy.context.scene.collection.objects.link(obj)         
        spline = crv.splines.new(type='BEZIER')
        
        spline.bezier_points.add(len(self.full_arr)-1)
        for i in range(len(self.full_arr)):
           
            
            spline.bezier_points[i].co = self.full_arr[i]
          
            spline.bezier_points[i].radius = self.rad[i]
               
            print(spline.bezier_points[i].radius)
            
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(*self.full_arr.T)
        ax.axis('equal')
        plt.show()
        
        blend_file_path = bpy.data.filepath
        directory = os.path.dirname(blend_file_path)
        target_file = os.path.join(directory, 'thisisatest.obj')
        bpy.ops.export_scene.obj(filepath=target_file)  
        
    def set_y_x_rad(self):
        camera = c.PinholeCameraModel()
        camera.from_npz(p.cam)
        assert isinstance(camera, c.PinholeCameraModel)
        
        
        for i in range(len(self.full_arr)):
            self.full_arr[i][0] = camera.getDeltaX(self.x[i], self.z[i])
            self.full_arr[i][1] = camera.getDeltaY(self.y[i], self.z[i])
            self.rad.append(camera.getDeltaX(self.num_pix[i], self.z[i]))
            print(self.num_pix[i])
       
                 

        
