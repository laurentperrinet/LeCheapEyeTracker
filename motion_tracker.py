#!/usr/bin/python

"""

motion_tracker.py : detecting the eyes

- motion_tracker.py :  grabs the current frame
- motion_tracker.py : gets the eyes position and crops the image
- face-detect.py : fits an ellipse to get eyeball position
- face-detect.py : infers eye position

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/ElCheapoEyeTracker

$Id$

voir  /Users/lup/Desktop/opencv/samples/python/fitellipse.py /Users/lup/Desktop/ElCheapoEyeTracker/sonic-gesture/pysonic-gesture/finder.py 

"""
import sys
import cv

bigeye_size = (500, 200)

def detect(image):
    image_size = cv.GetSize(image)

    # create grayscale version
    grayscale = cv.CreateImage(image_size, 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    # equalize histogram
    cv.EqualizeHist(grayscale, grayscale)

    # create storage
    storage = cv.CreateMemStorage(0)

    # detect objects
    cascade = cv.Load('/opt/local/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    eyes = cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING)
    if eyes:
        # TODO return each eye now...
        return eyes[0][0]
    else:
        return None

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
    eye = 0
    motion = 0
    capture = 0

    if len(sys.argv)==1:
        capture = cv.CreateCameraCapture(0)
    elif len(sys.argv)==2 and sys.argv[1].isdigit():
        capture = cv.CreateCameraCapture(int(sys.argv[1]))
    elif len(sys.argv)==2:
        capture = cv.CreateFileCapture(sys.argv[1]) 

    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)

    webcam = cv.QueryFrame(capture)
    img32F = cv.CreateImage(cv.GetSize(webcam), cv.IPL_DEPTH_32F, 3)
    bigeye = cv.CreateImage(bigeye_size, cv.IPL_DEPTH_32F, 3)
    image = cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 3)

    bigeye8B = cv.CreateImage (cv.GetSize(bigeye), 8, 3)

    cv.NamedWindow("image", 1)
    cv.MoveWindow("image", 600 , 0)
    while True:
        webcam = cv.QueryFrame(capture)
        rect_eye = detect(webcam)
        cv.ConvertScale(webcam, img32F)
        if not(rect_eye == None):
            eye = cv.GetSubRect(img32F, rect_eye)
            cv.Resize(eye, bigeye)
            cv.ConvertScale(bigeye, bigeye8B)
            cv.ShowImage("image", bigeye8B)

            if(cv.WaitKey(10) != -1):
                break
    cv.DestroyWindow("Motion")
