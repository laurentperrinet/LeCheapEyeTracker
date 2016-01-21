#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The server side.

"""

import time
import numpy as np
import cv2

from openRetina import PhotoReceptor
from multiprocessing.pool import ThreadPool
from collections import deque
from LeCheapEyeTracker.constants import *
#from constants import *

class Server:
    def __init__(self, threadn=0):
        import cv2
        self.threadn = threadn
        self.cam = PhotoReceptor()

        self.ctime = []
        self.eye_pos = []
        self.head_size = 486

        self.cascade = face_cascade
        self.eye_template = eye_image
        self.wt, self.ht = self.eye_template.shape[1], self.eye_template.shape[0]

    def init__threads(self):
        if self.threadn == 0 :
            self.threadn = cv2.getNumberOfCPUs()
            self.pool = ThreadPool(processes = self.threadn)
            self.pending = deque()

    def clock(self):
        return cv2.getTickCount() / cv2.getTickFrequency()

    def get_just_one(self, image, MinNeighbors=20, scale=1.1):
        features, minNeighbors = [], 1
        while len(features) == 0 and minNeighbors<MinNeighbors:
            features = self.cascade.detectMultiScale(image, scale, minNeighbors=MinNeighbors)
            minNeighbors += 1
        return features[0], minNeighbors

    def process_frame(self, frame, t0):
        (x, y, w, h), minNeighbors = self.get_just_one(frame)
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

        


if __name__ == '__main__':
    start = time.time()
    cam = ThreadSource()
    ctime = cam.run()
    cam.close()

