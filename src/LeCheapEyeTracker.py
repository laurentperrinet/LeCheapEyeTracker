#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

one main file.

"""

import time
import numpy as np
import cv2

from openRetina import PhotoReceptor
from multiprocessing.pool import ThreadPool
from collections import deque

class LeCheapEyeTracker:
    def __init__(self, threadn=0):
        import cv2
        self.threadn = threadn
        self.cam = PhotoReceptor()

        self.ctime = []
        self.eye_pos = []
        self.head_size = 486
        self.cascade = cv2.CascadeClassifier('../src/haarcascade_frontalface_default.xml') # TODO: use relative path
        self.eye_template = cv2.imread('../src/my_eye.png') # TODO: use relative path
        self.wt, self.ht = self.eye_template.shape[1], self.eye_template.shape[0]

    def init__threads(self):
        if self.threadn == 0 :
            self.threadn = cv2.getNumberOfCPUs()
            self.pool = ThreadPool(processes = self.threadn)
            self.pending = deque()

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
        img_face = cv2.resize(img_face, (self.head_size//2, self.head_size//2))
        res = cv2.matchTemplate(img_face, self.eye_template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return (max_loc[0] + self.wt/2, max_loc[1] + self.ht/2), t0

    def run(self, T=10):
        start = self.clock()
        self.init__threads()
        while self.clock()-start <T:
            if self.threadn > 1:
                while len(self.pending) > 0 and self.pending[0].ready():
                    res, t0 = self.pending.popleft().get()
                    self.eye_pos.append([res, t0])
                    self.ctime.append(self.clock() - start)
                if len(self.pending) < self.threadn:
                    frame = self.cam.grab()
                    if not frame is None:
                        task = self.pool.apply_async(self.process_frame, (frame.copy(), self.clock()))
                        self.pending.append(task)
            else:
                frame = self.cam.grab()
                res, t0 = self.process_frame (frame.copy(), self.clock())
                self.ctime.append(self.clock() - start)
                self.eye_pos.append([res, t0])

    def close(self):
        try:
            self.pool.terminate()
            self.pool.close()
        except:
            pass
        self.cam.close()

        
# VISUALIZATION ROUTINES
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
    def __init__(self, et, stim):
        self.et = et
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
        frame = self.et.cam.grab()
        if not frame is None:
            res, t0 = self.et.process_frame (frame.copy(), self.et.clock())
            self.et.eye_pos.append([res, t0])
        self.update()

if __name__ == '__main__':
    start = time.time()
    cam = ThreadSource()
    ctime = cam.run()
    cam.close()


