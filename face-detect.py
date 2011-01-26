#!/usr/bin/env python

"""

face-detect.py : detecting the eye ball

- motion_tracker.py :  grabs the current frame
- motion_tracker.py : gets the eyes position and crops the image
- face-detect.py : fits an ellipse to get eyeball position
- face-detect.py : infers eye position

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$

voir  /Users/lup/Desktop/opencv/samples/python/fitellipse.py /Users/lup/Desktop/ElCheapoEyeTracker/sonic-gesture/pysonic-gesture/finder.py 

"""
DEBUG = False

import sys
import time
import cv
import random
# from optparse import OptionParser
#import numpy as np
#import pylab
#import scipy
#import scipy.ndimage as image
# import color
# http://code.google.com/p/python-colormath/wiki/ColorObjects
from motion_tracker import detect

def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()


def highlight(bigeye):
    """
    some image processing to highlight the eyeball / remove the skin

    """    
    
    #                 mat = np.array(bigeye).astype(np.float)
    #                 if DEBUG: pylab.imshow(eye)
    #                 if DEBUG :
    #                     pylab.subplot(311)
    #                     pylab.imshow(mat[:,:,0])
    # 
    #                     pylab.subplot(312)
    #                     pylab.imshow(mat[:,:,1])
    # 
    #                     pylab.subplot(313)
    #                     pylab.imshow(mat[:,:,2])
    # 
    
    #                 contrast = mat.sum(axis=2)
    #                 contrast /= contrast.mean()
    #                 image.laplace(contrast, contrast)
    #                 if DEBUG: pylab.imshow(contrast)
    
    
    #                 hsv = cv.CreateImage(cv.GetSize(eye), cv.IPL_DEPTH_32F, 3)
    #                 cv.CvtColor(eye, hsv, cv.CV_RGB2Lab)
    # # 
    # # #                 mat = cv2array(hsv)
    # # #                 if DEBUG:
    # # #                     pylab.subplot(311)
    # # #                     pylab.imshow(mat[:,:,0])
    # # # 
    # # #                     pylab.subplot(312)
    # # #                     pylab.imshow(mat[:,:,1])
    # # # 
    # # #                     pylab.subplot(313)
    # # #                     pylab.imshow(mat[:,:,2])
    # # # scipy.ndimage.filters.laplace(input, output=None, mode='reflect', cval=0.0)
    # # #                 mat = color.rgb2lab(mat)
    # # #                 if DEBUG: pylab.matshow(mat)
    # 
    #                 hue = cv.CreateImage(cv.GetSize(eye), cv.IPL_DEPTH_32F, 1)
    #                 fundus = cv.CreateImage(cv.GetSize(eye), cv.IPL_DEPTH_32F, 1)
    #                 cv.Split(hsv, hue, None, fundus, None)
    #                 cv.EqualizeHist(fundus, fundus)
    # #                 cv.Threshold(fundus, fundus, 220, 255, cv.CV_THRESH_BINARY)
    # #                 for i in range(1):
    # #                     cv.Dilate(fundus, fundus, element, 1)
    # #                     cv.Erode(fundus, fundus, element, 1)
    # # 
    # # 
    #                 cv.Merge(hue, hue, fundus, None, hsv)
    #                 mat = cv2array(hsv)
    #                 mat[:, :, 2] = mat[:, :, 2] > 0.9 * mat[:, :, 2].max()
    #                 if True:
    #                     pylab.subplot(311)
    #                     pylab.imshow(mat[:,:,0])
    # 
    #                     pylab.subplot(312)
    #                     pylab.imshow(mat[:,:,1])
    # 
    #                     pylab.subplot(313)
    #                     pylab.imshow(mat[:,:,2])
    # 
    
    
    #                 cv.EqualizeHist(gray, gray)
    #                 fe = fit_ellipse(gray, 90)
    # 
    #                 cv.ShowImage("ROI", fundus)
    
    
    #                 gray = cv.CreateImage(cv.GetSize(eye), cv.IPL_DEPTH_32F, 1)
    #                 gray_ = cv.CreateMat(eye.width, eye.height, 8)
    #                 # convert color input image to grayscale
    #                 cv.CvtColor(eye, gray, cv.CV_BGR2GRAY)
    #                 cv.EqualizeHist(gray, gray)
    #                 cv.Sobel(gray_, gray_, xorder=1, yorder=0, apertureSize=3)
    # #                 cv.Threshold(gray, gray, 10, 255, cv.CV_THRESH_BINARY)
    #                 cv.ShowImage("Eye", gray)
    # 
    #                 cv.Ellipse(eye, center, axes, angle, start_angle, end_angle, color, thickness=1, lineType=8, shift=0)
    #                 cv.EqualizeHist(hue, hue)
    #                 cv.Max(hue, fundus, hue)
    #                 cv.ShowImage("Hue", hue)
#    planes = [cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 1) for i in range(3)]
#    laplace = cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 1)
#    
#    cv.Split(bigeye, planes[0], planes[1], planes[2], None)
#    for plane in planes:
#        cv.Laplace(plane, laplace, 3)
#    #         cv.ConvertScaleAbs(laplace, plane, 1, 0)
#    #        cv.Smooth(plane, plane, smoothtype=cv.CV_GAUSSIAN, param1=9, param2=0, param3=0, param4=0)
#    #        cv.Sobel(plane, plane, xorder=1, yorder=0)#, apertureSize=3)
#    cv.Merge(planes[0], planes[1], planes[2], None, image)
#
#    image = bigeye

    
    # convert color input image to grayscale
    cv.CvtColor(bigeye, gray, cv.CV_BGR2GRAY)
#    cv.Laplace(gray, gray)
    cv.Smooth(gray, gray, smoothtype=cv.CV_GAUSSIAN, param1=25, param2=0, param3=0, param4=0)
    cv.Laplace(gray, gray)
#    cv.ShowImage( "Eye", gray )

    return gray    
    
        
def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()
# 
def fit_ellipse(gray): 
    """
        This function finds contours, draws them and their approximation by ellipses.
        
    """
#    # create grayscale version
#    grayscale = cv.CreateImage(cv.GetSize(source_image), cv.IPL_DEPTH_32F, 1)
#    cv.CvtColor(source_image, grayscale, cv.CV_BGR2GRAY)

        
    # Create the destination images
    cols, rows = cv.GetSize(gray)
    image02 = cv.CreateMat(rows, cols, cv.CV_8UC1)
    cv.Zero(image02)

    # Threshold the source image. This needful for cv.FindContours().
#    gray = cv.CreateMat(rows, cols, cv.CV_8UC1)#cv.CloneImage(source_image)
    cv.ConvertScaleAbs(gray, image02, scale=.3, shift=0.)

    # equalize histogram
    cv.EqualizeHist(image02, image02)

#    cv.ShowImage( "Eye", image02 )

#    cv.Threshold(gray, image02, 200, 255, cv.CV_THRESH_BINARY)
#    cv.AdaptiveThreshold(gray, image02, 20)

    stor = cv.CreateMemStorage()
    # Find all contours.
    cont = cv.FindContours(image02,
        stor,
        cv.CV_RETR_LIST,
        cv.CV_CHAIN_APPROX_NONE,
        (0, 0))

    image04 = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 3)
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


if __name__ == "__main__":

    if DEBUG:
#         img = cv.fromarray(scipy.misc.lena().astype(np.float32)) # 
        img = cv.LoadImage('mona.jpg', cv.CV_LOAD_IMAGE_COLOR)
#         img = cv.LoadImage('/Users/lup/Desktop/opencv/doc/latex2sphinx/lena.jpg', cv.CV_LOAD_IMAGE_COLOR)
# 
    else:
        print "Press ESC to exit ..."

    #     cv.NamedWindow('Eye', cv.CV_WINDOW_AUTOSIZE)

        # inittialize webcam   

        snapshotTime = time.time()
        capture = cv.CaptureFromCAM(1)
        # check if capture device is OK
        if not capture:
            print "Error opening capture device"
            sys.exit(1)

        downsize = 1
        img = cv.QueryFrame(capture)
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img.height / downsize)
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img.width / downsize)
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, cv.IPL_DEPTH_32F)
        img = cv.QueryFrame(capture)
        print ' Startup time ', (time.time() - snapshotTime)*1000, ' ms'
#         snapshotTime = time.time()

    bigeye = cv.CreateImage((500, 200), cv.IPL_DEPTH_32F, 3)
    img8B = cv.CreateImage (cv.GetSize(img), cv.IPL_DEPTH_8U, 3)
    img32F = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)
    gray = cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 1)

    cv.NamedWindow("Eye", 1)
    cv.MoveWindow("Eye", 600 , 0)
    

    try:
#         cv.MoveWindow("Eye", 0*RF.width , 0)
#         cv.ResizeWindow("Eye", 2*RF.width, 2*RF.height)

        while True:
            snapshotTime = time.time()
            
            if not(DEBUG): img = cv.QueryFrame(capture)
#            cv.ConvertScale(img, img8B)
            cv.ConvertScale(img, img32F)

            left_eye = detect(img)
#             backshotTime = time.time()
#             fps = 1. / (backshotTime - snapshotTime)
#             cv.PutText(ret, str('%d'  %fps) + ' fps', (12/downsize, 24/downsize), font_, cv.RGB(255, 255, 255))
#             cv.PutText(ret, str('%d'  %fps) + ' fps', (12/downsize, 24/downsize), font, cv.RGB(0, 0, 0))
            if not(left_eye == None):
                eye = cv.GetSubRect(img32F, left_eye)
#                 cv.GetRectSubPix(src, dst, center)
                cv.Resize(eye, bigeye)
                gray = highlight(bigeye)
#                cv.ShowImage("Eye", gray)
                fit_ellipse(gray)



            key = cv.WaitKey(1)
#             if key == ord('r'): do_RF()
            if key == 27 or DEBUG: break

    finally:
        # Always close the camera stream
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

#    if len(sys.argv) > 1:
#        source_image = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
#    else:
#        url = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/stuff.jpg'
#        filedata = urllib2.urlopen(url).read()
#        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
#        cv.SetData(imagefiledata, filedata, len(filedata))
#        source_image = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)


# 
#     kalman = cv.CreateKalman(2, 1, 0)
#         kalman.transition_matrix[0,0] = 1
#         kalman.transition_matrix[0,1] = 1
#         kalman.transition_matrix[1,0] = 0
#         kalman.transition_matrix[1,1] = 1
# 
#         cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
#         cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))
#         cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
#         cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(1))
#         cv.RandArr(rng, kalman.state_post, cv.CV_RAND_NORMAL, cv.RealScalar(0), cv.RealScalar(0.1))
#               
#             prediction = cv.KalmanPredict(kalman)
#             predict_angle = prediction[0, 0] 
#             predict_pt = calc_point(predict_angle)
# 
#                  
#             cv.KalmanCorrect(kalman, measurement)
# 
#  

