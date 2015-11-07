#/usr/bin/env python
# -*- coding: utf8 -*-
"""

eyepair_tracker.py : detecting the eye pair by their movement not their shape

- grabbing.py : grabs the current frame
- eyepair_tracker.py : detecting the eye pair by their movement not their shape
- detect.py : fits a circle to get eyeball position


Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$


inspired by :
Gijs Molenaar
http://gijs.pythonic.nl

requires opencv svn + new python api
"""
import numpy as np
# CHANGE ME
CAMERAID=-1 # -1 for auto, -2 for video
HAARCASCADE_face="/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml" # where to find haar cascade file for face detection
HAARCASCADE_eyepair="/usr/local/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml"
HAARCASCADE_eye="/usr/local/share/opencv/haarcascades/haarcascade_eye.xml"
# TODO: is there a haarcascade for the eyeball?

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

import cv
import time
import sys
import math
import cv

eyepair_size = (600, 180)
bottom = 20

depth = cv.IPL_DEPTH_8U


def hue_histogram_as_image(hist):
    """ Returns a nice representation of a hue histogram """
    histimg_hsv = cv.CreateImage( (320,200), depth, 3)

    mybins = cv.CloneMatND(hist.bins)
    cv.Log(mybins, mybins)
    (_, hi, _, _) = cv.MinMaxLoc(mybins)
    cv.ConvertScale(mybins, mybins, 255. / hi)

    w,h = cv.GetSize(histimg_hsv)
    hdims = int(cv.GetDims(mybins)[0])
    for x in range(w):
        xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
        val = int(mybins[int(hdims * x / w)] * h / 255)
        cv.Rectangle( histimg_hsv, (x, 0), (x, h-val), (xh,255,64), -1)
        cv.Rectangle( histimg_hsv, (x, h-val), (x, h), (xh,255,255), -1)

    histimg = cv.CreateImage( (320,200), 8, 3)
    cv.CvtColor(histimg_hsv, histimg, cv.CV_HSV2BGR)
    return histimg



def detect_eyepair(image):
#    
#class CvKalman
#
#Kalman filter state.
#
#    MP
#        number of measurement vector dimensions
#
#    DP
#        number of state vector dimensions
#
#    CP
#        number of control vector dimensions
#
#    state_pre
#        predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
#
#    state_post
#        corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
#
#    transition_matrix
#        state transition matrix (A)
#
#    control_matrix
#        control matrix (B) (it is not used if there is no control)
#
#    measurement_matrix
#        measurement matrix (H)
#
#    process_noise_cov
#        process noise covariance matrix (Q)
#
#    measurement_noise_cov
#        measurement noise covariance matrix (R)
#
#    error_cov_pre
#        priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q
#
#    gain
#        Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
#
#    error_cov_post
#        posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
#
#The structure CvKalman is used to keep the Kalman filter state. It is created by the CreateKalman function, updated by the KalmanPredict and KalmanCorrect functions . Normally, the structure is used for the standard Kalman filter (notation and the formulas below are borrowed from the excellent Kalman tutorial Welch95 )
#
#\begin{array}{l} x_k=A \cdot x_{k-1}+B \cdot u_k+w_k \\ z_k=H \cdot x_k+v_k \end{array}
#
#where:
#
#\begin{array}{l l} x_k \; (x_{k-1})& \text{state of the system at the moment \emph{k} (\emph{k-1})} \\ z_k & \text{measurement of the system state at the moment \emph{k}} \\ u_k & \text{external control applied at the moment \emph{k}} \end{array}
#
#w_k and v_k are normally-distributed process and measurement noise, respectively:
#
#\begin{array}{l} p(w) \sim N(0,Q) \\ p(v) \sim N(0,R) \end{array}
#
#that is,
#
#Q process noise covariance matrix, constant or variable,
#
#R measurement noise covariance matrix, constant or variable
#
#In the case of the standard Kalman filter, all of the matrices: A, B, H, Q and R are initialized once after the CvKalman structure is allocated via CreateKalman . However, the same structure and the same functions may be used to simulate the extended Kalman filter by linearizing the extended Kalman filter equation in the current system state neighborhood, in this case A, B, H (and, probably, Q and R) should be updated on every step.
#CreateKalman
#
#Comments from the Wiki
#
#CreateKalman(dynam_params, measure_params, control_params=0) -> CvKalman
#
#The function allocates CvKalman and all its matrices and initializes them somehow.
#KalmanCorrect
#
#Comments from the Wiki
#
#KalmanCorrect(kalman, measurement) -> cvmat
#
#    Adjusts the model state.
#    Parameters:	
#
#The function adjusts the stochastic model state on the basis of the given measurement of the model state:
#
#\begin{array}{l} K_k=P'_k \cdot H^T \cdot (H \cdot P'_k \cdot H^T+R)^{-1} \\ x_k=x'_k+K_k \cdot (z_k-H \cdot x'_k) \\ P_k=(I-K_k \cdot H) \cdot P'_k \end{array}
#
#where
#z_k 	given measurement ( mesurement parameter)
#K_k 	Kalman gain matrix.
#
#The function stores the adjusted state at kalman->state_post and returns it on output.
#KalmanPredict
#
#Comments from the Wiki
#
#KalmanPredict(kalman, control=None) -> cvmat
#
#    Estimates the subsequent model state.
#    Parameters:	
#
#The function estimates the subsequent stochastic model state by its current state and stores it at kalman->state_pre :
#
#\begin{array}{l} x'_k=A x_{k-1} + B u_k \\ P'_k=A P_{k-1} A^T + Q \end{array}
#
#where
#x'_k 	is predicted state kalman->state_pre ,
#x_{k-1} 	is corrected state on the previous step kalman->state_post (should be initialized somehow in the beginning, zero vector by default),
#u_k 	is external control ( control parameter),
#P'_k 	is priori error covariance matrix kalman->error_cov_pre
#P_{k-1} 	is posteriori error covariance matrix on the previous step kalman->error_cov_post (should be initialized somehow in the beginning, identity matrix by default),
#
#The function returns the estimated state.
#KalmanUpdateByMeasurement
#
#Synonym for KalmanCorrect
#KalmanUpdateByTime
#
#Synonym for KalmanPredict


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

    image_size = cv.GetSize(image)

    # create grayscale version
    grayscale = cv.CreateImage(image_size, depth, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    # equalize histogram
    cv.EqualizeHist(grayscale, grayscale)

    # create storage
    storage = cv.CreateMemStorage(0)

    # detect objects
    cascade = cv.Load(HAARCASCADE_eyepair)
    eyes = cv.HaarDetectObjects(image, cascade, storage, haar_scale, min_neighbors, haar_flags, min_size)

    if eyes:
        # TODO return each eye now...
        rect_eye = np.array(eyes[0][0])
        rect_eye[3] += bottom
        return rect_eye
    else:
        return None

def crop_eyepair(highlight=False, display=None):
    from grabbing import grab
    webcam = grab(None)
    rect_eye = detect_eyepair(webcam)
    img32F = cv.CreateImage(cv.GetSize(webcam), depth, 3)
    cv.ConvertScale(webcam, img32F)
    if not(rect_eye == None):
#        print rect_eye
        eye = cv.GetSubRect(img32F, (rect_eye[0], rect_eye[1], rect_eye[2], rect_eye[3]))
        eyepair = cv.CreateImage(eyepair_size, depth, 3)
        cv.Resize(eye, eyepair)
        if highlight: 
            eyepair_gray = do_highlight(eyepair)       
        else:
            eyepair_gray = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
            cv.CvtColor(eyepair, eyepair_gray, cv.CV_BGR2GRAY)

        if not(display==None): 
            eyepair8B = cv.CreateImage(cv.GetSize(eyepair), 8, 1)
            cv.ConvertScale(eyepair_gray, eyepair8B)
            cv.ShowImage(display, eyepair8B)
        return eyepair_gray



hist_size = 180
range_0 = [0, 360]
ranges = [ range_0 ]


def do_highlight(eyepair):
    """
    some image processing:
        - remove the skin, treansform to grayscale 
        - highlight the eyeball, transform to BW
        - remove reflection

    """    
    
    eyepair_bottom = cv.GetSubRect(eyepair, (0, 180-bottom, 600, bottom)) #   x,y,w,h 
    
    # Convert to HSV and keep the hue
    hsv = cv.CreateImage(cv.GetSize(eyepair_bottom), depth, 3)
    cv.CvtColor(eyepair_bottom, hsv, cv.CV_BGR2HSV)
    hue = cv.CreateImage(cv.GetSize(eyepair_bottom), depth, 1)
#    sat = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
    cv.Split(hsv, hue, None, None, None)
    hist = cv.CreateHist([hist_size], cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcArrHist([hue], hist)
#    (min_value, max_value, _, _) = cv.GetMinMaxHistValue(hist)
#    cv.Scale(hist.bins, hist.bins, float(hist_image.height) / max_value, 0)

    # Convert to HSV and keep the hue
    hsv = cv.CreateImage(cv.GetSize(eyepair), depth, 3)
    cv.CvtColor(eyepair, hsv, cv.CV_BGR2HSV)
    hue = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
#    sat = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
    cv.Split(hsv, hue, None, None, None)
#    print  cv.MinMaxLoc(hue)

#    return eyepair
    # Compute back projection
    backproject = cv.CreateImage(cv.GetSize(eyepair), depth, 1)

    cv.CalcArrBackProject( [hue], backproject, hist )
    
    return backproject
#
#    hist_img = hue_histogram_as_image(hist)
#
#    cv.ShowImage( "histogram", hist_img )

    
#    gray = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
#    cv.CvtColor(eyepair, gray, cv.CV_BGR2GRAY)
#    cv.Rectangle(gray, (0,180-bottom), (0+600, 180), (0,255,0)) # (x,y), (x+w,y+h) 
#    
#    return gray
#    
    #                 hsv = cv.CreateImage(cv.GetSize(eye), depth, 3)
    #                 cv.CvtColor(eye, hsv, cv.CV_RGB2Lab)
    # 
    #                 hue = cv.CreateImage(cv.GetSize(eye), depth, 1)
    #                 fundus = cv.CreateImage(cv.GetSize(eye), depth, 1)
    #                 cv.Split(hsv, hue, None, fundus, None)
    #                 cv.EqualizeHist(fundus, fundus)
    # #                 cv.Threshold(fundus, fundus, 220, 255, cv.CV_THRESH_BINARY)
    # #                 for i in range(1):
    # #                     cv.Dilate(fundus, fundus, element, 1)
    # #                     cv.Erode(fundus, fundus, element, 1)
    # # 
    # # 
    #                 cv.Merge(hue, hue, fundus, None, hsv)
    
    #                 cv.EqualizeHist(hue, hue)
    #                 cv.Max(hue, fundus, hue)
    #                 cv.ShowImage("Hue", hue)
    
    #
    #planes = [cv.CreateImage(cv.GetSize(img), 8, 1) for i in range(3)]
    #laplace = cv.CreateImage(cv.GetSize(img), depth, 1)
    #colorlaplace = cv.CreateImage((img.width, img.height), 8, 3)
    #
    #cv.Split(img, planes[0], planes[1], planes[2], None)
    #for plane in planes:
    #    cv.Laplace(plane, laplace, 3)
    #    cv.ConvertScaleAbs(laplace, plane, 1, 0)
    #
    #cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)
    #
    #cv.ShowImage("Color Laplacian", colorlaplace)
    #
    #
    #gray = cv.CreateImage(cv.GetSize(img), depth, 1)
    #cv.CvtColor(img32F, gray, cv.CV_BGR2GRAY)
    #laplace = cv.CreateImage((img.width, img.height), depth, 1)
    #cv.Laplace(gray, laplace, 3)
    #cv.ShowImage("Laplacian", laplace)
    
    
    # result = cv.CreateImage(cv.GetSize(img), 8, 1)
    # cv.Laplace(gray, result)
    # cv.Smooth(gray, gray, smoothtype=cv.CV_GAUSSIAN, param1=7, param2=0, param3=0, param4=0)
    # 
    # cv.EqualizeHist(gray, gray)
    # fe = fit_ellipse(gray, 90)
    #    # create grayscale version
    
    #                 cv.EqualizeHist(gray, gray)

    #                 cv.EqualizeHist(gray, gray)
    #                 cv.Sobel(gray_, gray_, xorder=1, yorder=0, apertureSize=3)
    # #                 cv.Threshold(gray, gray, 10, 255, cv.CV_THRESH_BINARY)
    #                 cv.ShowImage("Eye", gray)


#    gray = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
#    cv.CvtColor(hue, gray, cv.CV_BGR2GRAY)
#    cv.EqualizeHist(gray, gray)

    # equalize histogram
#    cv.EqualizeHist(gray, gray)
#    
#    eyepair_bw = cv.CreateImage(cv.GetSize(eyepair), depth, 1)
#        
#    # Create the destination images
#    cols, rows = cv.GetSize(gray)
#    eyepair_bw = cv.CreateMat(rows, cols, cv.CV_8UC1)
#    cv.Zero(eyepair_bw)
##
##    # Threshold the source image. This needful for cv.FindContours().
###    gray = cv.CreateMat(rows, cols, cv.CV_8UC1)#cv.CloneImage(source_image)
##    cv.ConvertScaleAbs(gray, image02, scale=.3, shift=0.)
#
##    cv.ShowImage( "Eye", image02 )
#
##    cv.Threshold(gray, eyepair_bw, 200, 255, cv.CV_THRESH_BINARY)
#    cv.AdaptiveThreshold(gray, eyepair_bw, 20)
#
#    # convert color input image to grayscale
##    cv.Laplace(gray, gray)
##    cv.Smooth(gray, gray, smoothtype=cv.CV_GAUSSIAN, param1=25, param2=0, param3=0, param4=0)
##    cv.Laplace(gray, gray)
##    cv.ShowImage( "Eye", gray )
#
#    return eyepair_bw    
    
    
if __name__ == "__main__":

    display = "detect_eyepair"
    cv.NamedWindow(display, 1)
    cv.MoveWindow(display, 600 , 0)
    
    while True:
        crop_eyepair(display=display)
        if(cv.WaitKey(10) != -1):
            break

    cv.DestroyWindow(display)


