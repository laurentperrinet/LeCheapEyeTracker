#/usr/bin/env python
# -*- coding: utf8 -*-
"""

detect.py : fits a circle to get eyeball position

- grabbing.py : grabs the current frame
- eyepair_tracker.py : detecting the eye pair by their movement not their shape
- detect.py : fits a circle to get eyeball position

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$

"""
DEBUG = False

import cv
import random

from eyepair_tracker import crop_eyepair
        
def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()
        

#ExtractSURF(image, mask, storage, params)-> (keypoints, descriptors)¶
#
#    Extracts Speeded Up Robust Features from an image.
#    Parameters:	
#
#        * image (CvArr) – The input 8-bit grayscale image
#        * mask (CvArr) – The optional input 8-bit mask. The features are only found in the areas that contain more than 50 % of non-zero mask pixels
#        * keypoints (CvSeq of CvSURFPoint) – sequence of keypoints.
#        * descriptors (CvSeq of list of float) – sequence of descriptors. Each SURF descriptor is a list of floats, of length 64 or 128.
#        * storage (CvMemStorage) – Memory storage where keypoints and descriptors will be stored
#        * params (CvSURFParams) –
#
#          Various algorithm parameters in a tuple (extended, hessianThreshold, nOctaves, nOctaveLayers) :
#              o extended 0 means basic descriptors (64 elements each), 1 means extended descriptors (128 elements each)
#              o hessianThreshold only features with hessian larger than that are extracted. good default value is ~300-500 (can depend on the average local contrast and sharpness of the image). user can further filter out some features based on their hessian values and other characteristics.
#              o nOctaves the number of octaves to be used for extraction. With each next octave the feature size is doubled (3 by default)
#              o nOctaveLayers The number of layers within each octave (4 by default)
#
#The function cvExtractSURF finds robust features in the image, as described in Bay06 . For each feature it returns its location, size, orientation and optionally the descriptor, basic or extended. The function can be used for object tracking and localization, image stitching etc.
#
#To extract strong SURF features from an image
#>>> import cv
#>>> im = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
#>>> (keypoints, descriptors) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, 30000, 3, 1))
#>>> print len(keypoints), len(descriptors)
#6 6
#>>> for ((x, y), laplacian, size, dir, hessian) in keypoints:
#...     print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)
#x=30 y=27 laplacian=-1 size=31 dir=69.778503 hessian=36979.789062
#x=296 y=197 laplacian=1 size=33 dir=111.081039 hessian=31514.349609
#x=296 y=266 laplacian=1 size=32 dir=107.092300 hessian=31477.908203
#x=254 y=284 laplacian=1 size=31 dir=279.137360 hessian=34169.800781
#x=498 y=525 laplacian=-1 size=33 dir=278.006592 hessian=31002.759766
#x=777 y=281 laplacian=1 size=70 dir=167.940964 hessian=35538.363281
#


#GetStarKeypoints(image, storage, params) → keypoints¶
#
#    Retrieves keypoints using the StarDetector algorithm.
#    Parameters:	
#
#        * image (CvArr) – The input 8-bit grayscale image
#        * storage (CvMemStorage) – Memory storage where the keypoints will be stored
#        * params (CvStarDetectorParams) –
#
#          Various algorithm parameters in a tuple (maxSize, responseThreshold, lineThresholdProjected, lineThresholdBinarized, suppressNonmaxSize) :
#              o maxSize maximal size of the features detected. The following values of the parameter are supported: 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128
#              o responseThreshold threshold for the approximatd laplacian, used to eliminate weak features
#              o lineThresholdProjected another threshold for laplacian to eliminate edges
#              o lineThresholdBinarized another threshold for the feature scale to eliminate edges
#              o suppressNonmaxSize linear size of a pixel neighborhood for non-maxima suppression
#
#The function GetStarKeypoints extracts keypoints that are local scale-space extremas. The scale-space is constructed by computing approximate values of laplacians with different sigma’s at each pixel. Instead of using pyramids, a popular approach to save computing time, all of the laplacians are computed at each pixel of the original high-resolution image. But each approximate laplacian value is computed in O(1) time regardless of the sigma, thanks to the use of integral images. The algorithm is based on the paper Agrawal08 , but instead of a square, hexagon or octagon it uses an 8-end star shape, hence the name, consisting of overlapping upright and tilted squares.
#
#Each keypoint is represented by a tuple ((x, y), size, response) :
#
#        * x, y Screen coordinates of the keypoint
#        * size feature size, up to maxSize
#        * response approximated laplacian value for the keypoint
#

def fit_ellipse(eyepair_bw, slider_pos): 
    """
        This function finds contours, draws them and their approximation by ellipses.
        
    """

    stor = cv.CreateMemStorage()
    # Find all contours.
    cont = cv.FindContours(eyepair_bw,
        stor,
        cv.CV_RETR_LIST,
        cv.CV_CHAIN_APPROX_NONE,
        (0, 0))

    image04 = cv.CreateImage(cv.GetSize(eyepair_bw), cv.IPL_DEPTH_32F, 3)
    cv.Zero(image04)
    for c in contour_iterator(cont):
        # Number of points must be more than or equal to 6 for cv.FitEllipse2
        if len(c) >= 6:
            # Copy the contour into an array of (x,y)s
            PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
            for (i, (x, y)) in enumerate(c):
                PointArray2D32f[0, i] = (x, y)
            
            # Draw the current contour in gray
            gray_ = cv.CV_RGB(100, 100, 100)
            cv.DrawContours(image04, c, gray_, gray_,0,1,8,(0,0))
            
            # Fits ellipse to current contour.
            (center, size, angle) = cv.FitEllipse2(PointArray2D32f)
            
            # Convert ellipse data from float to integer representation.
            center = (cv.Round(center[0]), cv.Round(center[1]))
            size = (cv.Round(size[0] * 0.5), cv.Round(size[1] * 0.5))
            angle = -angle
            
            # Draw ellipse in random color
            color = cv.CV_RGB(random.randrange(256),random.randrange(256),random.randrange(256))
            cv.Ellipse(image04, center, size,
                      angle, 0, 360,
                      color, 2, cv.CV_AA, 0)

    # Show image. HighGUI use.
    cv.ShowImage( "Eye", image04 )


def fit_circle(eyepair_bw, slider_pos, display=True): 
    """
    cannyThreshold (TColor)
    The higher threshold of the two passed to Canny edge detector (the lower one will be twice smaller).
    
    accumulatorThreshold (TColor)
        Accumulator threshold at the center detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first
    
    dp (Double)
        Resolution of the accumulator used to detect centers of the circles. For example, if it is 1, the accumulator will have the same resolution as the input image, if it is 2 - accumulator will have twice smaller width and height, etc
    
    minDist (Double)
        Minimum distance between centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed
    
    minRadius (Int32)
        Minimal radius of the circles to search for
    
    maxRadius (Int32)
    Maximal radius of the circles to search for
    """
    

    storage = cv.CreateMat(eyepair_bw.width, 1, cv.CV_32FC3)
#    edges = cv.CreateImage(cv.GetSize(eyepair_bw), 8, 1)
#    cv.Canny(eyepair_bw, edges, 10, 25, 3)
#    (cannyThreshold, accumulatorThreshold, dp, minDist, minRadius, maxRadius)
#HoughCircles(Mat& image, vector<Vec3f>& circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0, int maxRadius=0)
    cv.HoughCircles(eyepair_bw, storage, cv.CV_HOUGH_GRADIENT, 1, 1, slider_pos, 1, 1, 300)

    if display:
        print  storage.width
        if storage.width>1 :  print  storage.width, (storage[0, 0], storage[0, 1]), storage[0, 2]
        image04 = cv.CreateImage(cv.GetSize(eyepair_bw), 8, 3)
        cv.Zero(image04)
        for i in xrange(storage.width - 1):
            radius = storage[i, 2]
            center = (storage[i, 0], storage[i, 1])
    
            print (i, radius, center)
    
            cv.Circle(image04, center, radius, (0, 0, 255), 3, 8, 0)
    
#        cv.Circle(edges, center, radius, (0, 0, 255), 3, 8, 0)
        cv.ShowImage( "Eye", image04 )

if __name__ == "__main__":



    cv.NamedWindow("Eye", 1)
    cv.MoveWindow("Eye", 600 , 0)
    cv.DestroyWindow("detect_eyepair")
    slider_pos = 70
    cv.CreateTrackbar( "Threshold", "Eye", slider_pos, 255, fit_ellipse )
    

#    try:
#            if not(DEBUG): img = grab(None)
    while True:
        
        eyepair_bw = crop_eyepair(display=None)
        if not(eyepair_bw==None):
#            fit_ellipse(eyepair_bw, 1)
            fit_circle(eyepair_bw, 1)

#            key = cv.WaitKey(1)
##             if key == ord('r'): do_RF()
#            if key == 27 or DEBUG: break
        if(cv.WaitKey(10) != -1):
            break            

    cv.DestroyWindow("Eye")

# # Show the image.
#     cv.ShowImage("Source", source_image)
# 
#     fe = FitEllipse(source_image, 70)
# 
#     # create capture device
#     device = 0 # assume we want first device
#     capture = cv.CaptureFromCAM(0)
#     cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#     cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
# 
#     # check if capture device is OK
#     if not capture:
#         print "Error opening capture device"
#         sys.exit(1)
# 
#     while 1:
#         # do forever
# 
#         # capture the current frame
#         frame = cv.QueryFrame(capture)
#         if frame is None:
#             break
# 
#         # mirror
#         cv.Flip(frame, None, 1)
# 
#         # face detection
#         detect(frame)
# 
#         # display webcam image
#         cv.ShowImage('Raw', frame)
# 
#         # handle events
#         k = cv.WaitKey(10)
# 
#         if k == 0x1b: # ESC
#             print 'ESC pressed. Exiting ...'
#             break


