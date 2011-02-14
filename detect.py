#!/usr/bin/env python

"""

detect.py : detecting the eye ball

- motion_tracker.py :  grabs the current frame
- motion_tracker.py : gets the eyes position and crops the image
- face-detect.py : fits an ellipse to get eyeball position
- face-detect.py : infers eye position

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$

voir  /Users/lup/Desktop/opencv/samples/python/fitellipse.py /Users/lup/Desktop/ElCheapoEyeTracker/sonic-gesture/pysonic-gesture/finder.py 

"""
DEBUG = False

import cv
import random

from eyepair_tracker import crop_eyepair
        
def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()

 
def fit_ellipse(eyepair_bw): 
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


if __name__ == "__main__":



    cv.NamedWindow("Eye", 1)
    cv.MoveWindow("Eye", 600 , 0)



    cv.DestroyWindow("detect_eyepair")
    

#    try:
#            if not(DEBUG): img = grab(None)

    while True:
        
        eyepair_bw = crop_eyepair(display=None)
        
        fit_ellipse(eyepair_bw)

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


