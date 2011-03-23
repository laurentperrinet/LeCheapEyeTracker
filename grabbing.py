#/usr/bin/env python
# -*- coding: utf8 -*-
"""

grabbing.py : grabs the current frame

- grabbing.py : grabs the current frame
- eyepair_tracker.py : detecting the eye pair by their movement not their shape
- detect.py : fits a circle to get eyeball position


Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$


"""
DEBUG = False
#DEBUG = True

downscale = 1
capture = None

import cv, os, time, sys

MOVIE="/home/gijs/Work/sonic-vision/data/wayne_cotter.mp4"
depth = cv.IPL_DEPTH_8U

class Source:
    def __init__(self, id, flip=True):
        self.flip = flip
        if id == -2:
            self.capture = cv.CaptureFromFile(MOVIE)
        else:
            self.capture = cv.CaptureFromCAM(id)

    def print_info(self):
        for prop in [ cv.CV_CAP_PROP_POS_MSEC, cv.CV_CAP_PROP_POS_FRAMES,
                cv.CV_CAP_PROP_POS_AVI_RATIO, cv.CV_CAP_PROP_FRAME_WIDTH,
                cv.CV_CAP_PROP_FRAME_HEIGHT, cv.CV_CAP_PROP_FPS,
                cv.CV_CAP_PROP_FOURCC, cv.CV_CAP_PROP_BRIGHTNESS,
                cv.CV_CAP_PROP_CONTRAST, cv.CV_CAP_PROP_SATURATION,
                cv.CV_CAP_PROP_HUE]:
            print cv.GetCaptureProperty(self.capture, prop)

    def grab_frame(self):
        self.frame = cv.QueryFrame(self.capture)
        if not self.frame:
            print "can't grap frame, or end of movie. Bye bye."
            sys.exit(2)
        if self.flip:
            cv.Flip(self.frame, None, 1)
        return self.frame
# 
#FindExtrinsicCameraParams2¶
#
#Comments from the Wiki
#
#FindExtrinsicCameraParams2(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=0) → None¶
#
#    Finds the object pose from the 3D-2D point correspondences
#    Parameters:	
#
#        * objectPoints (CvMat) – The array of object points in the object coordinate space, 3xN or Nx3 1-channel, or 1xN or Nx1 3-channel, where N is the number of points.
#        * imagePoints (CvMat) – The array of corresponding image points, 2xN or Nx2 1-channel or 1xN or Nx1 2-channel, where N is the number of points.
#        * cameraMatrix (CvMat) – The input camera matrix A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}
#        * distCoeffs (CvMat) – The input vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
#        * rvec (CvMat) – The output rotation vector (see Rodrigues2 ) that (together with tvec ) brings points from the model coordinate system to the camera coordinate system
#        * tvec (CvMat) – The output translation vector
#        * useExtrinsicGuess (int) – If true (1), the function will use the provided rvec and tvec as the initial approximations of the rotation and translation vectors, respectively, and will further optimize them.
#
#The function estimates the object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, i.e. the sum of squared distances between the observed projections imagePoints and the projected (using ProjectPoints2 ) objectPoints .
#
#The function’s counterpart in the C++ API is 
 
def webcam(downscale=downscale):

#    if len(sys.argv) > 1:
#        source_image = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
#    else:
#        url = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/stuff.jpg'
#        filedata = urllib2.urlopen(url).read()
#        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
#        cv.SetData(imagefiledata, filedata, len(filedata))
#        source_image = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)



    print "Press ESC to exit ..."
    # initialize webcam   
    snapshotTime = time.time()

    capture = cv.CaptureFromCAM(1)
    # check if capture device is OK
    if not capture:
        print "Error opening capture device"
        sys.exit(1)


    img = cv.QueryFrame(capture)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img.height / downscale)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img.width / downscale)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, depth)
    img = cv.QueryFrame(capture)
    print ' Startup time ', (time.time() - snapshotTime)*1000, ' ms'

    return capture

def grab(figname, downscale=downscale):
    global capture
    if (figname==None) or not(os.path.isfile(figname)):
        if (capture==None): 
            capture = webcam(downscale=downscale)
        # taking a snapshot 
        img = cv.QueryFrame(capture)
        if not(figname==None):
            cv.SaveImage(figname, img)
    else:        
        img = cv.LoadImage(figname, cv.CV_LOAD_IMAGE_COLOR)
    
    # convert to 32bits float
#    img32F = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)
#    cv.ConvertScale(img, img32F)
    # 
    # imgGS_32F = cv.CreateImage (cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
    # cv.CvtColor(img32F, imgGS_32F, cv.CV_RGB2GRAY)
    # 
    # imgGS = cv.CreateImage (cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
    # cv.ConvertScale(imgGS_32F, imgGS)
    # 
    return img
    

def cv2array(im):
    import numpy as np
    depth2dtype = {
        cv.IPL_DEPTH_32F: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
    # arrdtype = im.depth
    a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height, im.width, im.nChannels)
    return a


if __name__ == "__main__":


#    if len(sys.argv)==1:
#        capture = cv.CreateCameraCapture(0)
#    elif len(sys.argv)==2 and sys.argv[1].isdigit():
#        capture = cv.CreateCameraCapture(int(sys.argv[1]))
#    elif len(sys.argv)==2:
#        capture = cv.CreateFileCapture(sys.argv[1]) 
#
#    if not capture:
#        print "Could not initialize capturing..."
#        sys.exit(-1)

    if DEBUG:
       figname = 'snapshot.png'
       img = grab(figname)
       cv.ShowImage("Grabbing", img)

    else:
        while True:
           img = grab(None)
           cv.ShowImage("Grabbing", img)
    
           if(cv.WaitKey(10) != -1):
               break
        cv.DestroyWindow("Grabbing")

