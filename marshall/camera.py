import copy
import cv2
import numpy
import math
from skimage import io

def mkmat(rows, cols, L):
    mat = numpy.matrix(L, dtype='float64')
    mat.resize((rows,cols))
    return mat


class PinholeCameraModel:

    """
    A pinhole camera is an idealized monocular camera.
    """

    def __init__(self):
        self.K = None
        self.D = None
        self.R = None
        self.P = None
        self.full_K = None
        self.full_P = None
        self.width = None
        self.height = None
        self.binning_x = None
        self.binning_y = None
        self.raw_roi = None
        self.tf_frame = None
        self.stamp = None

    def fromCameraInfo(self, msg):
        """
        :param msg: camera parameters
        :type msg:  sensor_msgs.msg.CameraInfo

        Set the camera parameters from the :class:`sensor_msgs.msg.CameraInfo` message.
        """
        self.K = mkmat(3, 3, msg.k)
        if msg.d:
            self.D = mkmat(len(msg.d), 1, msg.d)
        else:
            self.D = None
        self.R = mkmat(3, 3, msg.r)
        self.P = mkmat(3, 4, msg.p)
        self.full_K = mkmat(3, 3, msg.k)
        self.full_P = mkmat(3, 4, msg.p)
        self.width = msg.width
        self.height = msg.height
        self.binning_x = max(1, msg.binning_x)
        self.binning_y = max(1, msg.binning_y)
        self.resolution = (msg.width, msg.height)

        self.raw_roi = copy.copy(msg.roi)
        # ROI all zeros is considered the same as full resolution
        if (self.raw_roi.x_offset == 0 and self.raw_roi.y_offset == 0 and
            self.raw_roi.width == 0 and self.raw_roi.height == 0):
            self.raw_roi.width = self.width
            self.raw_roi.height = self.height

        # Adjust K and P for binning and ROI
        self.K[0,0] /= self.binning_x
        self.K[1,1] /= self.binning_y
        self.K[0,2] = (self.K[0,2] - self.raw_roi.x_offset) / self.binning_x
        self.K[1,2] = (self.K[1,2] - self.raw_roi.y_offset) / self.binning_y
        self.P[0,0] /= self.binning_x
        self.P[1,1] /= self.binning_y
        self.P[0,2] = (self.P[0,2] - self.raw_roi.x_offset) / self.binning_x
        self.P[1,2] = (self.P[1,2] - self.raw_roi.y_offset) / self.binning_y

    def to_npz(self, output_path):
        attribs = ['K', 'D', 'R', 'P', 'full_K', 'full_P', 'width', 'height', 'binning_x', 'binning_y',
                   'resolution']
        vals = {attrib: getattr(self, attrib) for attrib in attribs}
        numpy.savez(output_path, **vals)

    def from_npz(self, path):
        for attrib, val in numpy.load(path).items():
            setattr(self, attrib, val)

    def rectifyImage(self, raw, rectified):
        """
        :param raw:       input image
        :type raw:        :class:`CvMat` or :class:`IplImage`
        :param rectified: rectified output image
        :type rectified:  :class:`CvMat` or :class:`IplImage`

        Applies the rectification specified by camera parameters :math:`K` and and :math:`D` to image `raw` and writes the resulting image `rectified`.
        """

        self.mapx = numpy.ndarray(shape=(self.height, self.width, 1),
                           dtype='float32')
        self.mapy = numpy.ndarray(shape=(self.height, self.width, 1),
                           dtype='float32')
        cv2.initUndistortRectifyMap(self.K, self.D, self.R, self.P,
                (self.width, self.height), cv2.CV_32FC1, self.mapx, self.mapy)
        cv2.remap(raw, self.mapx, self.mapy, cv2.INTER_CUBIC, rectified)

    def rectifyPoint(self, uv_raw):
        """
        :param uv_raw:    pixel coordinates
        :type uv_raw:     (u, v)

        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to point (u, v) and returns the
        pixel coordinates of the rectified point.
        """

        src = mkmat(1, 2, list(uv_raw))
        src.resize((1,1,2))
        dst = cv2.undistortPoints(src, self.K, self.D, R=self.R, P=self.P)
        return dst[0,0]

    def project3dToPixel(self, point):
        """
        :param point:     3D point
        :type point:      (x, y, z)

        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`projectPixelTo3dRay`.
        """
        src = mkmat(4, 1, [point[0], point[1], point[2], 1.0])
        dst = self.P * src
        x = dst[0,0]
        y = dst[1,0]
        w = dst[2,0]
        if w != 0:
            return (x / w, y / w)
        else:
            return (float('nan'), float('nan'))

    def projectPixelTo3dRay(self, uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (uv[0] - self.cx()) / self.fx()
        y = (uv[1] - self.cy()) / self.fy()
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return (x, y, z)

    def getDeltaU(self, deltaX, Z):
        """
        :param deltaX:          delta X, in cartesian space
        :type deltaX:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta u, given Z and delta X in Cartesian space.
        For given Z, this is the inverse of :meth:`getDeltaX`.
        """
        fx = self.P[0, 0]
        if Z == 0:
            return float('inf')
        else:
            return fx * deltaX / Z

    def getDeltaV(self, deltaY, Z):
        """
        :param deltaY:          delta Y, in cartesian space
        :type deltaY:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta v, given Z and delta Y in Cartesian space.
        For given Z, this is the inverse of :meth:`getDeltaY`.
        """
        fy = self.P[1, 1]
        if Z == 0:
            return float('inf')
        else:
            return fy * deltaY / Z

    def getDeltaX(self, deltaU, Z):
        """
        :param deltaU:          delta u in pixels
        :type deltaU:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta X, given Z in cartesian space and delta u in pixels.
        For given Z, this is the inverse of :meth:`getDeltaU`.
        """
        fx = self.P[0, 0]
        return Z * deltaU / fx

    def getDeltaY(self, deltaV, Z):
        """
        :param deltaV:          delta v in pixels
        :type deltaV:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta Y, given Z in cartesian space and delta v in pixels.
        For given Z, this is the inverse of :meth:`getDeltaV`.
        """
        fy = self.P[1, 1]
        return Z * deltaV / fy

    def fullResolution(self):
        """Returns the full resolution of the camera"""
        return self.resolution

    def intrinsicMatrix(self):
        """ Returns :math:`K`, also called camera_matrix in cv docs """
        return self.K
    def distortionCoeffs(self):
        """ Returns :math:`D` """
        return self.D
    def rotationMatrix(self):
        """ Returns :math:`R` """
        return self.R
    def projectionMatrix(self):
        """ Returns :math:`P` """
        return self.P
    def fullIntrinsicMatrix(self):
        """ Return the original camera matrix for full resolution """
        return self.full_K
    def fullProjectionMatrix(self):
        """ Return the projection matrix for full resolution """
        return self.full_P

    def cx(self):
        """ Returns x center """
        return self.P[0,2]
    def cy(self):
        """ Returns y center """
        return self.P[1,2]
    def fx(self):
        """ Returns x focal length """
        return self.P[0,0]
    def fy(self):
        """ Returns y focal length """
        return self.P[1,1]

    def Tx(self):
        """ Return the x-translation term of the projection matrix """
        return self.P[0,3]

    def Ty(self):
        """ Return the y-translation term of the projection matrix """
        return self.P[1,3]

    def tfFrame(self):
        """ Returns the tf frame name - a string - of the camera.
        This is the frame of the :class:`sensor_msgs.msg.CameraInfo` message.
        """
        return self.tf_frame




# IGNORE THIS STUFF
def run_test_node():
    import rclpy
    from follow_the_leader.utils.ros_utils import TFNode
    import pickle
    class Temp(TFNode):
        def _handle_cam_info(self, msg):
            cam = PinholeCameraModel()
            cam.fromCameraInfo(msg)
            cam.to_npz('D435_640x480.camera.npz')
    rclpy.init()
    bla = Temp('temp', cam_info_topic='/camera/color/camera_info')
    rclpy.spin(bla)
    
