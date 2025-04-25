#!/usr/bin/env python3

import OpenGL.GL as GL
import cv2
from ctypes import c_uint8

import numpy as np


class DrawImages():
    im_dict = {"rgb": 0, "depth": 1, "edge": 2, "flow": 3, "mask": 4,
               "rgb_edge": 5, "rgb_mask": 6, "rgb_mask_edge": 7, "rgb_edge_rgb_next": 8}

    def __init__(self, gui, parent=None):
        self.crv_gl_list = -1
        self.image_gl_tex = []

        # Pointer back to gui window
        self.gui = gui

        self.aspect_ratio = 1.0
        self.im_size = (0, 0)

    def bind_texture(self, rgb_image, edge_image=None, mask_image=None, flow_image=None, depth_image=None, next_rgb_image=None):
        self.aspect_ratio = rgb_image.shape[0] / rgb_image.shape[1]
        self.im_size = (rgb_image.shape[1], rgb_image.shape[0])

        im_size = 512
        if rgb_image.shape[0] > 1024:
            im_size = 1024
        im_sq = cv2.resize(rgb_image, (im_size, im_size))

        if mask_image:
            im_sq_mask = cv2.cvtColor(cv2.resize(mask_image, (im_size, im_size)), cv2.COLOR_GRAY2RGB)
        else:
            im_sq_mask = im_sq

        if edge_image:
            if len(edge_image.shape) == 3:
                edge_flattened = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
            else:
                edge_flattened = edge_image
            im_sq_edge = cv2.cvtColor(cv2.resize(edge_flattened, (im_size, im_size)), cv2.COLOR_GRAY2RGB)
        else:
            im_sq_edge = im_sq

        if flow_image:
            im_sq_flow = cv2.resize(flow_image, (im_size, im_size))
        else:
            im_sq_flow = None

        if depth_image:
            im_sq_depth = cv2.resize(depth_image, (im_size, im_size))
        else:
            im_sq_depth = None

        if next_rgb_image:
            im_sq_next_rgb = cv2.resize(next_rgb_image, (im_size, im_size))
            im_sq_next_rgb = cv2.cvtColor(im_sq_next_rgb, cv2.COLOR_BGR2GRAY)
        else:
            im_sq_next_rgb = cv2.cvtColor(im_sq, cv2.COLOR_BGR2GRAY)

        im_sq_rgb_edge = im_sq // 2
        im_sq_rgb_mask = im_sq // 2
        im_sq_rgb_mask_edge = im_sq // 2
        im_sq_rgb_rgb_next = im_sq_next_rgb // 2
        for ch in (1, 2):
            im_sq_rgb_edge[:, :, ch] = im_sq_rgb_edge[:, :, ch] + im_sq_edge[:, :, ch] // 2
            im_sq_rgb_mask_edge[:, :, ch] = im_sq_rgb_mask_edge[:, :, ch] + im_sq_edge[:, :, ch] // 2
            im_sq_rgb_rgb_next = im_sq_rgb_mask_edge[:, :, ch] + im_sq_next_rgb[:, :, ch] // 2
        im_sq_rgb_mask[:, :, 0] = im_sq_rgb_mask[:, :, 0] + im_sq_mask[:, :, 0] // 2
        im_sq_rgb_mask_edge[:, :, 0] = im_sq_rgb_mask_edge[:, :, 0] + im_sq_mask[:, :, 0] // 2

        if len(self.image_gl_tex) == 0:
            n_textures = 10
            self.image_gl_tex = GL.glGenTextures(n_textures)
        for i, im in enumerate(
                # im_dict={"rgb": 0, "depth": 1, "edge": 2, "flow": 3, "mask": 4,
                #          "rgb_edge": 5, "rgb_mask": 6, "rgb_mask_edge": 7, "rgb_edge_rgb_next": 8}
                [im_sq, im_sq_depth, im_sq_edge, im_sq_flow, im_sq_mask, im_sq_rgb_edge, im_sq_rgb_mask, im_sq_rgb_mask_edge,
                 im_sq_rgb_rgb_next]):
            if im is None:
                self.image_gl_tex[i] = 100
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_gl_tex[i])
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                c_my_texture = (c_uint8 * im_size * im_size)()  # copying under correct ctype format (likely clumsy)
                c_my_texture.value = im[:, :, :]
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, 3, im_size, im_size, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                                c_my_texture.value)

    def draw_image(self, which_image):
        """ Draw the image
        @param which_image is one of im_dict entries"""
        if len(self.image_gl_tex) == 0:
            return

        tex_indx = DrawImages.im_dict[which_image]

        if self.image_gl_tex[tex_indx] != 100:
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_DECAL)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_gl_tex[tex_indx])
            GL.glEnable(GL.GL_TEXTURE_2D)

            quad_size_x = 1.0
            quad_size_y = self.aspect_ratio * quad_size_x
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2d(0.0, 1.0)
            GL.glVertex2f(-quad_size_x, -quad_size_y)
            GL.glTexCoord2d(1.0, 1.0)
            GL.glVertex2f(quad_size_x, -quad_size_y)
            GL.glTexCoord2d(1.0, 0.0)
            GL.glVertex2f(quad_size_x, quad_size_y)
            GL.glTexCoord2d(0.0, 0.0)
            GL.glVertex2f(-quad_size_x, quad_size_y)
            GL.glEnd()
