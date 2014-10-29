#!/usr/bin/python
import sys
import time
import cv

bigeye_size = (500, 200)
diff_threshold = 35
minimumComponent = 100
# inspired by samples/python/motempl.py
CLOCKS_PER_SEC = 1.0
MHI_DURATION = 1
MAX_TIME_DELTA = 0.5
MIN_TIME_DELTA = 0.05
N = 4
buf = range(10)
last = 0
mhi = None # MHI
orient = None # orientation
mask = None # valid orientation mask
segmask = None # motion segmentation map
storage = None # temporary storage
eye = 0
motion = None
capture = None


def detect(image):
    image_size = cv.GetSize(image)

    # create grayscale version
    grayscale = cv.CreateImage(image_size, 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    # create storage
    storage = cv.CreateMemStorage(0)

    # equalize histogram
    cv.EqualizeHist(grayscale, grayscale)

    # detect objects
    cascade = cv.Load('/opt/local/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    eyes = cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING)
    if eyes:
        # TODO return each eye now...
        return eyes[0][0]
    else:
        return None

def highlight(bigeye, image):
    planes = [cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 1) for i in range(3)]
    laplace = cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 1)

    cv.Split(bigeye, planes[0], planes[1], planes[2], None)
    for plane in planes:
        cv.Laplace(plane, laplace, 3)
#         cv.ConvertScaleAbs(laplace, plane, 1, 0)
#        cv.Smooth(plane, plane, smoothtype=cv.CV_GAUSSIAN, param1=9, param2=0, param3=0, param4=0)
#        cv.Sobel(plane, plane, xorder=1, yorder=0)#, apertureSize=3)
    cv.Merge(planes[0], planes[1], planes[2], None, image)

    image = bigeye

def update_mhi(img, dst, diff_threshold):
    global last
    global mhi
    global storage
    global mask
    global orient
    global segmask
    timestamp = time.clock() / CLOCKS_PER_SEC # get current time in seconds
    size = cv.GetSize(img) # get current frame size
    idx1 = last
    if not mhi or cv.GetSize(mhi) != size:
        for i in range(N):
            buf[i] = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
            cv.Zero(buf[i])
        mhi = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        cv.Zero(mhi) # clear MHI at the beginning
        orient = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        segmask = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        mask = cv.CreateImage(size,cv. IPL_DEPTH_8U, 1)
    
    cv.CvtColor(img, buf[last], cv.CV_BGR2GRAY) # convert frame to grayscale
    idx2 = (last + 1) % N # index of (last - (N-1))th frame
    last = idx2
    silh = buf[idx2]
    cv.AbsDiff(buf[idx1], buf[idx2], silh) # get difference between frames
    cv.Threshold(silh, silh, diff_threshold, 1, cv.CV_THRESH_BINARY) # and threshold it
    cv.UpdateMotionHistory(silh, mhi, timestamp, MHI_DURATION) # update MHI
    cv.CvtScale(mhi, mask, 255./MHI_DURATION,
                (MHI_DURATION - timestamp)*255./MHI_DURATION)
    cv.Zero(dst)
    cv.Merge(mask, None, None, None, dst)
    cv.CalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3)
    if not storage:
        storage = cv.CreateMemStorage(0)
    seq = cv.SegmentMotion(mhi, segmask, storage, timestamp, MAX_TIME_DELTA)
    global_angle, weight = 0, 0
    for (area, value, comp_rect) in seq:
        if comp_rect[2] + comp_rect[3] > minimumComponent: # reject very small  components(in size)
            color = cv.CV_RGB(255, 0,0)
            silh_roi = cv.GetSubRect(silh, comp_rect)
            mhi_roi = cv.GetSubRect(mhi, comp_rect)
            orient_roi = cv.GetSubRect(orient, comp_rect)
            mask_roi = cv.GetSubRect(mask, comp_rect)
            angle = 360 - cv.CalcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)
            weight_ = 1#comp_rect[2] * comp_rect[3]
            weight += weight_
            global_angle += weight_ * angle

            count = cv.Norm(silh_roi, None, cv.CV_L1, None) # calculate number of points within silhouette ROI
            if count < (comp_rect[2] * comp_rect[3] * 0.05):
                continue

#    if weight>0: print timestamp, global_angle/weight
    if weight>0: return timestamp, global_angle/weight, weight
    else: return timestamp, None, 0.
    
def eye_tracker():
    global bigeye8B
    global motion
    webcam = cv.QueryFrame(capture)
    img32F = cv.CreateImage(cv.GetSize(webcam), cv.IPL_DEPTH_32F, 3)
    bigeye = cv.CreateImage(bigeye_size, cv.IPL_DEPTH_32F, 3)
    image = cv.CreateImage(cv.GetSize(bigeye), cv.IPL_DEPTH_32F, 3)
    if not motion:
    
        bigeye8B = cv.CreateImage(cv.GetSize(bigeye), 8, 3)
        motion = cv.CreateImage(cv.GetSize(bigeye), 8, 3)
        cv.Zero(motion)

    rect_eye = detect(webcam)
    if not(rect_eye == None):
        #cv.ConvertScale(webcam, img32F)
        eye = cv.GetSubRect(img32F, rect_eye)
        cv.Resize(eye, bigeye)
        highlight(bigeye, image)
        cv.ConvertScale(bigeye, bigeye8B)
        return update_mhi(bigeye8B, motion, diff_threshold=diff_threshold)
    else:
        print 'no eye detected'
        return time.clock(), 0, None

if __name__ == "__main__":
#    global capture
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
#
#
#    while True:
#        timestamp, weight, global_angle = eye_tracker()
#        print timestamp, weight, global_angle
#        if(cv.WaitKey(10) != -1):
#            break

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
#     img8B = cv.CreateImage (cv.GetSize(webcam), cv.IPL_DEPTH_8U, 1)
# 
    timestamp, angle, weight = [], [], []
    while True:
        webcam = cv.QueryFrame(capture)
        rect_eye = detect(webcam)
        cv.ConvertScale(webcam, img32F)
        if not(rect_eye == None):
            eye = cv.GetSubRect(img32F, rect_eye)
            cv.Resize(eye, bigeye)
            highlight(bigeye, image)
            cv.ConvertScale(bigeye, bigeye8B)
            if(not motion):
                    motion = cv.CreateImage(cv.GetSize(bigeye), 8, 3)
                    cv.Zero(motion)
                    #motion.origin = image.origin
            timestamp_, angle_, weight_ = update_mhi(bigeye8B, motion, diff_threshold=diff_threshold)
            timestamp.append(timestamp_)
            angle.append(angle_)
            weight.append(weight_)
            
            print timestamp_, angle_, weight_
            if(cv.WaitKey(10) != -1) or timestamp_>10:
                break

    import pylab
    pylab.plot(timestamp, angle, 'r')