#!/usr/bin/env python
"""

One main file.

"""

HAARCASCADE="~/pool/libs/vision/opencv/data/haarcascades/haarcascade_frontalface_default.xml" # where to find haar cascade file for face detection

from multiprocessing.pool import ThreadPool
from collections import deque
import cv2

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class Camera:
    def __init__(self, w=640, h=480):
        self.h, self.w = h, w
        import cv2
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        self.threadn = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes = self.threadn)
        self.pending = deque()

        self.latency = StatValue()
        self.frame_interval = StatValue()
        self.last_frame_time = clock()        
        self.display = False
        self.ctime = []
        self.N = 0
        self.face_cascade = cv2.CascadeClassifier('/Users/laurentperrinet/pool/libs/vision/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('/Users/laurentperrinet/pool/libs/vision/opencv/data/haarcascades/haarcascade_eye.xml')

    def process_frame(self, frame, t0):

        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5) # image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]
        if len(faces) > 0:
            (x,y,w,h) = faces[0]
            roi_color = frame[y:y+h, x:x+w]
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            eyes = self.eye_cascade.detectMultiScale(roi_color, 1.1, 5, 4)
            if len(eyes) == 2:
                (ex,ey,ew,eh) = eyes[0]
                eye_L = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                (ex,ey,ew,eh) = eyes[1]
                eye_R = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                if eyes[0][0] < eyes[1][0]:
                    eye_L, eye_R = eye_R, eye_L
        return frame, t0

    def grab(self):
        ret, frame = self.cap.read()
        return frame
    
    def run(self, T=10):
        start = clock()
        while clock()-start <T:
            while len(self.pending) > 0 and self.pending[0].ready():
                res, t0 = self.pending.popleft().get()
                self.latency.update(clock() - t0)
                if self.display:
                    draw_str(res, (20, 40), "latency        :  %.1f ms" % (self.latency.value*1000))
                    draw_str(res, (20, 60), "frame interval :  %.1f ms" % (self.frame_interval.value*1000))
                    cv2.imshow('Webcam video', res)
                self.ctime.append(clock() - start)
                self.N += 1
            if len(self.pending) < self.threadn:
                frame = self.grab()
                t = clock()
                self.frame_interval.update(t - self.last_frame_time)
                self.last_frame_time = t
                task = self.pool.apply_async(self.process_frame, (frame.copy(), t))
                self.pending.append(task)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                self.close()
        return frame
    
    def close(self):
        self.cap.release()
        if self.display: cv2.destroyAllWindows()

def hue_histogram_as_image(hist):
    """ Returns a nice representation of a hue histogram """
    histimg_hsv = cv.CreateImage( (320,200), 8, 3)

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




class GetHands:
    def __init__(self):

        self.source = Source(CAMERAID)
        #self.source.print_info()
        self.threshold_value = THRESH
        self.hc = cv.Load(HAARCASCADE)
        self.ms = cv.CreateMemStorage()

        self.orig = self.source.grab_frame()

        self.width = self.orig.width
        self.height = self.orig.height
        self.size = (self.width, self.height)
        self.smallheight = WORKING_HEIGHT
        self.smallwidth = int(self.width * self.smallheight/self.height * 1.0)
        self.smallsize = (self.smallwidth, self.smallheight)

        # alloc mem for images
        self.small = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.visualize = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.bw = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.hsv = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.hue = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.sat = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.bp = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.th = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.morphed = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.temp = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        self.temp3 = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.result = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 3)
        self.hist_image = cv.CreateImage((320,200), cv.IPL_DEPTH_8U, 1)

     
        # make matrix for erode/dilate
        MORPH_SIZE = 3
        center = (MORPH_SIZE / 2) + 1
        self.morpher_small = cv.CreateStructuringElementEx(cols=MORPH_SIZE,
            rows=MORPH_SIZE, anchorX=center, anchorY=center,
            shape=cv.CV_SHAPE_ELLIPSE)
        MORPH_SIZE = 11
        center = (MORPH_SIZE / 2) + 1
        self.morpher = cv.CreateStructuringElementEx(cols=MORPH_SIZE,
            rows=MORPH_SIZE, anchorX=center, anchorY=center,
            shape=cv.CV_SHAPE_ELLIPSE)

        # alloc mem for face histogram
        self.face_hist = cv.CreateHist([HUEBINS, SATBINS], cv.CV_HIST_ARRAY,
            [[0, 180], [0, 255]], 1)

        # alloc mem for background histogram
        self.bg_hist = cv.CreateHist([HUEBINS, SATBINS], cv.CV_HIST_ARRAY,
            [[0, 180], [0, 255]], 1)


        # video writer
        if STORE:
            self.writer = cv.CreateVideoWriter(OUTPUT,
                cv.CV_FOURCC('M','J','P','G'), 15, cv.GetSize(self.combined),
                is_color=1)

        # make window
        cv.NamedWindow('Skin Detection') 
        cv.CreateTrackbar('Threshold', 'Skin Detection', self.threshold_value,
            255, self.change_threshold)


    def change_threshold(self, position):
        self.threshold_value = position


    def find_face(self, image):
        """ detect faces in image using haar object detector """
        faces = cv.HaarDetectObjects(self.bw, self.hc, self.ms, scale_factor=1.2,
            min_neighbors=2, flags=cv.CV_HAAR_DO_CANNY_PRUNING)
        if len(faces) > 0:
            (x, y, w, h), n = faces[0]
            if n > 5:
                return (x, y, w, h)
        return False


    def face_region(self, face, border):
        (x, y, w, h) = face
        x2 = int(x+w*0.2)
        y2 = int(y+h*0.15) # added to make rectange
        w2 = int(w-w*0.2*2)
        h2 = int(h-h*0.15*2)
        return (x2, y2, w2, h2)


    def draw_face(self, image, face, sub_face):
        (x, y, w, h) = face
        #cv.Rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))
        (x, y, w, h) = sub_face
        cv.Rectangle(image, (x, y), (x+w, y+h), (128, 150, 0))
 

    def update_histogram(self, region):
        cv.SetImageROI(self.hue, region)
        cv.SetImageROI(self.sat, region)
        cv.CalcArrHist([self.hue, self.sat], self.face_hist, 1)
        #cv.NormalizeHist(self.face_hist, 255)
        cv.ResetImageROI(self.hue)
        cv.ResetImageROI(self.sat)


    def backproject(self):
        """ do a backprojection of face histogram on full image """
        cv.CalcArrBackProject([self.hue, self.sat], self.bp, self.face_hist)
        return self.bp


    def normalize(self, image):
        """ scale image to max of 255 """
        minVal,maxVal,minLoc,maxLoc = cv.MinMaxLoc(image)
        if maxVal > 0:
            scaler = 255/maxVal
            cv.ConvertScale(image, image, scale=scaler, shift=0.0) 
        return image


    def threshold(self, image):
        """ binary threshold image """
        cv.Threshold(image, self.th, self.threshold_value, 255, cv.CV_THRESH_BINARY)
        #cv.AdaptiveThreshold(self.scaled, self.th, 255, adaptive_method=cv.CV_ADAPTIVE_THRESH_MEAN_C, threshold_type=cv.CV_THRESH_BINARY_INV, block_size=7, param1=5)
        return self.th


    def morphology(self, image):
        """ remove noisy pixels by doing erode and dilate """
        cv.MorphologyEx(self.th, self.temp, None, self.morpher_small,
            cv.CV_MOP_OPEN, iterations=1) 
        cv.MorphologyEx(self.temp, self.morphed, None, self.morpher,
            cv.CV_MOP_CLOSE, iterations=1) 
        cv.Dilate(self.morphed, self.temp, self.morpher) 
        cv.Copy(self.temp, self.morphed)
        return self.morphed


    def find_contours(self, image):
        return cv.FindContours(image, self.ms, mode=cv.CV_RETR_EXTERNAL,
            method=cv.CV_CHAIN_APPROX_SIMPLE, offset=(0, 0))


    def draw_contours(self, image, contours):
        if contours:
            cv.DrawContours(img=self.result, contour=contours,
                external_color=(50, 200, 50), hole_color=(50, 200, 50), max_level=1,
                thickness=1, lineType=1, offset=(0, 0))


    def find_limbs(self, contours):
        """ return the 3 biggest contours """
        blobs = []
        while contours:
            (i, center, radius) = cv.MinEnclosingCircle(contours)
            blobs.append((int(round(radius)), tuple([int(round(c)) for c in center])))
            contours = contours.h_next()
        blobs.sort()
        blobs.reverse()
        return blobs[:3]


    def draw_limbs(self, image, limbs):
        """ draw limb positions with circles into a image """        
        [left_hand, head, right_hand] = limbs

        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5) 

        c = (255, 255, 255)
        if left_hand:
            cv.PutText(image, 'L', left_hand[1], font, c)

        if head:
            cv.PutText(image, 'H', head[1], font, c)

        if right_hand:
            cv.PutText(image, 'R', right_hand[1], font, c)

        for (r, c) in [x for x in limbs if x]:
            #cv.Circle(self.result, c, r, (0, 0, 255))
            pass


    def sort_limbs(self, limbs):
        """ guess the limb type, and sort them lefthand, head, righthand """
        left_hand = None
        right_hand = None
        head = None

        # sort by x position of center
        sorted_limbs = sorted(limbs, key=lambda x: (x[1][0], x[0]))
        if len(limbs) == 3:
            [left_hand, head, right_hand] = sorted_limbs

        elif len(limbs) == 2:
            [left_hand, right_hand] = limbs
        elif len(limbs) == 1:
            [head] = limbs
            left_hand = right_hand = head

        return [left_hand, head, right_hand]


    def make_sound(self, limbs):
        """ translate limb positions to osc signals """
        [left_hand, head, right_hand] = limbs
#        if left_hand:
#            left = 100 - int(left_hand[1][1] / self.smallheight * 100)
#            osc.sendMsg("/left", [left], "127.0.0.1", OSC_PORT)
#        if right_hand:
#            right = 100 - int(right_hand[1][1] / self.smallheight * 100)
#            osc.sendMsg("/right", [right], "127.0.0.1", OSC_PORT)
#        

    def combine_images(self, images):
        """ render a list of images into one opencv frame """
        comb_width = self.smallwidth * XWINDOWS
        comb_height = self.smallheight * int(math.ceil(len(images) / float(XWINDOWS)))
        self.combined = cv.CreateImage((comb_width, comb_height), cv.IPL_DEPTH_8U, 3)

        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3)

        for i,(image, name) in enumerate(images):
            if image.nChannels == 1:
                cv.Merge(image, image, image, None, self.temp3)
            else:
                cv.Copy(image, self.temp3)
            xoffset = (i % XWINDOWS) * self.smallsize[0]
            yoffset = (i / XWINDOWS) * self.smallsize[1]
            cv.SetImageROI(self.combined, (xoffset, yoffset, self.smallsize[0],
                self.smallsize[1]))
            cv.Copy(self.temp3, self.combined)
            cv.PutText(self.combined, name, (5, 10), font, (0, 0, 200))
            cv.ResetImageROI(self.combined)
        return self.combined


    def pipeline(self):
        presentation = []
        self.orig = self.source.grab_frame()
        cv.Resize(self.orig, self.small)
        cv.CvtColor(self.small, self.hsv, cv.CV_BGR2HSV)
        cv.Split(self.hsv, self.hue, self.sat, self.bw, None)
        cv.Copy(self.small, self.visualize)
        presentation.append((self.visualize, 'input'))
        face = self.find_face(self.small)
        if face:
            sub_face = self.face_region(face, FACE_BORDER)
            self.update_histogram(sub_face)
            self.draw_face(self.visualize, face, sub_face)

            hue_bg = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
            sat_bg = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
            (x, y, w, h) = face
            cv.Copy(self.hue, hue_bg) 
            cv.Copy(self.sat, sat_bg) 
            cv.Rectangle(hue_bg, (x, y), (x+w, y+h), 0, cv.CV_FILLED)
            cv.Rectangle(sat_bg, (x, y), (x+w, y+h), 0, cv.CV_FILLED)
            cv.CalcArrHist([self.hue, self.sat], self.bg_hist, 1)
            cv.NormalizeHist(self.bg_hist, 255)

        bp_bg = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        cv.CalcArrBackProject([self.hue, self.sat], bp_bg, self.bg_hist)
        self.normalize(bp_bg)
        presentation.append((bp_bg, 'background bp'))

        bp = self.backproject()
        presentation.append((bp, 'forground bp'))
        self.normalize(bp)

        compare = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        compare_th = cv.CreateImage(self.smallsize, cv.IPL_DEPTH_8U, 1)
        cv.Cmp(bp, bp_bg, compare, cv.CV_CMP_GT)
        #cv.AddS(bp_bg, 1, bp_bg) 
        #cv.Div(bp, bp_bg, compare)
        self.normalize(compare)
        cv.Threshold(compare, compare_th, self.threshold_value, 255, cv.CV_THRESH_BINARY)
        presentation.append((compare_th, 'compare'))

        th = self.threshold(bp)
        presentation.append((th, 'normal th'))
        morphed = self.morphology(th)

        # make dark copy of original
        cv.Copy(self.small, self.result)
        cv.ConvertScale(self.result, self.result, 0.2)
        cv.Copy(self.small, self.result, morphed)
        contours = self.find_contours(morphed)
        self.draw_contours(self.result, contours)
        limbs = self.find_limbs(contours)
        limbs = self.sort_limbs(limbs)
        self.draw_limbs(self.result, limbs)
        presentation.append((self.result, 'result'))
        self.make_sound(limbs)
        
        # combine and show the results
        combined = self.combine_images(presentation)
        if face:
            sub_face = self.face_region(face, FACE_BORDER)
            self.draw_face(self.result, face, sub_face)

        cv.ShowImage('Skin Detection', combined )

        if STORE:
            cv.WriteFrame(self.writer, self.combined)

        #hist_img = hue_histogram_as_image(self.face_hist)
        #mat = self.face_hist.bins
        #x = cv.GetMat(mat, 0, 1)
        #print type(x)
        #print dir(x)
        #print x.width
        #print x[0][0]
        #cv.ShowImage( "histogram", x )


    def run(self):
        while True:
            t = time.time()
            self.pipeline()
            wait = max(2, (1000/FPS)-int((time.time()-t)*1000))
            cv.WaitKey(wait)


if __name__ == '__main__':
    start = time.time()
    cam = ThreadSource()
    ctime = cam.run()
    cam.close()





