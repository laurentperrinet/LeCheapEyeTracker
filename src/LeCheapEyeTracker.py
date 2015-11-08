#!/usr/bin/env python
"""

One main file.

"""

HAARCASCADE="~/pool/libs/vision/opencv/data/haarcascades/haarcascade_frontalface_default.xml" # where to find haar cascade file for face detection

import numpy as np
from multiprocessing.pool import ThreadPool
from collections import deque
import cv2, time

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

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class LeCheapEyeTracker:
    def __init__(self, w=640, h=480):
        self.h, self.w = h, w
        import cv2
        self.cap = cv2.VideoCapture(0)
        self.DOWNSCALE = 1
        W = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W/self.DOWNSCALE)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H/self.DOWNSCALE)

        self.threadn = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes = self.threadn)
        self.pending = deque()

        self.latency = StatValue()
        self.frame_interval = StatValue()
        self.last_frame_time = self.clock()
        self.display = False
        self.ctime = []
        self.eye_pos = []
        self.N = 0
        self.cascade = cv2.CascadeClassifier('/Users/laurentperrinet/pool/science/LeCheapEyeTracker/src/haarcascade_frontalface_default.xml')
        self.eye_template = cv2.imread('/Users/laurentperrinet/pool/science/LeCheapEyeTracker/src/my_eye.png')
        self.wt, self.ht = self.eye_template.shape[1], self.eye_template.shape[0]

    def clock(self):
        return cv2.getTickCount() / cv2.getTickFrequency()

    def process_frame(self, frame, t0):
        def get_just_one(image):
            features, minNeighbors = [], 1
            while len(features) == 0 and minNeighbors<20:
                features = self.cascade.detectMultiScale(image, 1.1, minNeighbors) 
                minNeighbors += 1
            return features[0], minNeighbors
        (x, y, w, h), minNeighbors = get_just_one(frame)
        half_w, quarter_w = w//2, w//4
        img_face = frame[(y+quarter_w):(y+quarter_w+half_w), x:x+half_w]
        res = cv2.matchTemplate(img_face, self.eye_template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return (max_loc[0] + self.wt/2, max_loc[1] + self.ht/2), t0

    def grab(self):
        ret, frame = self.cap.read()
        return frame
    
    def run(self, T=10):
        start = self.clock()
        while self.clock()-start <T:
            if False:
                while len(self.pending) > 0 and self.pending[0].ready():
                    res, t0 = self.pending.popleft().get()
                    self.eye_pos.append([res, t0])
                    self.latency.update(self.clock() - t0)
                    self.ctime.append(self.clock() - start)
                    self.N += 1
                if len(self.pending) < self.threadn:
                    frame = self.grab()
                    t = self.clock()
                    self.frame_interval.update(t - self.last_frame_time)
                    self.last_frame_time = t
                    task = self.pool.apply_async(self.process_frame, (frame.copy(), t))
                    self.pending.append(task)
            else:
                res, t0 = self.process_frame (frame.copy(), self.clock())
                self.eye_pos.append([res, t0])
                
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                self.close()

    
    def close(self):
        self.cap.release()
        if self.display: cv2.destroyAllWindows()

from vispy import app
from vispy import gloo

vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);

        // HACK: the image is in BGR instead of RGB.
        float temp = gl_FragColor.r;
        gl_FragColor.r = gl_FragColor.b;
        gl_FragColor.b = temp;
    }
"""

class Canvas(app.Canvas):
    def __init__(self, cam, stim):
        self.cam = cam
        self.stim, self.timeline = stim
        self.h, self.w, three = self.stim(0).shape
        app.use_app('pyglet')
        app.Canvas.__init__(self, keys='interactive', fullscreen=True, size=(1280, 960))#
        self.program = gloo.Program(vertex, fragment, count=4)
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.program['texture'] = np.zeros((self.h, self.w, 3)).astype(np.uint8)
        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.start = time.time()
        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear('black')
        if time.time()-self.start < self.timeline.max(): # + ret.sleep_time*2: 
            #i_t = min(self.timeline - time.time()-self.start)
            image = self.stim((time.time()-self.start)/self.timeline.max())
            self.program['texture'][...] = image.reshape((self.h, self.w, 3))
            self.program.draw('triangle_strip')
            
        else:
            self.close()
        
    def on_timer(self, event):
        frame = self.cam.grab()
        res, t0 = self.cam.process_frame (frame.copy(), self.clock())
        self.cam.eye_pos.append([res, t0])
        self.update()

if __name__ == '__main__':
    start = time.time()
    cam = ThreadSource()
    ctime = cam.run()
    cam.close()





