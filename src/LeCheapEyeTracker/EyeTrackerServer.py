#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The server side.

"""

import time
import numpy as np


class FaceExtractor:
    def __init__(self):

        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def get_bbox(self, frame, do_center=True, do_topcrop=True):
        N_X, N_Y, three = frame.shape
        dets = self.detector(frame, 1)
        bbox = dets[0]
        t, b, l, r = bbox.top(), bbox.bottom(), bbox.left(), bbox.right()
        if do_center:
            height = b - t
            # print(height, N_Y//2 - height, N_Y//2 + height)
            l = np.max((N_Y//2 - height, 0))
            r = np.min((N_Y//2 + height, N_Y))
            #TODO make a warning if we get out of the window?
            if do_topcrop:
                b = t + height//2
            return t, b, l, r
        else:
            return t, b, l, r

    def face_extractor(self, frame, bbox=None):
        if bbox is None:
            t, b, l, r = self.get_bbox(frame)
        else:
            t, b, l, r = bbox
        face = frame[t:b, l:r, :]
        return face

class Server:
    def __init__(self, w=640, h=480, threadn=1):
        import cv2
        self.threadn = threadn
        self.cam = None
        from openRetina import PhotoReceptor
        if self.cam is None: self.cam = PhotoReceptor(w=w, h=h)

        self.eye_x_t = []
        self.head_size = 486
        self.F = FaceExtractor()
        #self.ml = self.model.load_state_dict(torch.load(path))

    def init__threads(self):
        if self.threadn == 0 :
            from multiprocessing.pool import ThreadPool
            self.threadn = cv2.getNumberOfCPUs()
            self.pool = ThreadPool(processes = self.threadn)
            from collections import deque
            self.pending = deque()

    def clock(self):
        return cv2.getTickCount() / cv2.getTickFrequency()

    def run(self, T=10):

        start = self.clock()
        if self.threadn > 1: self.init__threads()
        while self.clock()-start <T:
            # if self.threadn > 1:
            #     while len(self.pending) > 0 and self.pending[0].ready():
            #         img_face, res, t0 = self.pending.popleft().get()
            #         x, y = res
            #         self.eye_x_t.append((x, self.clock() - start))
            #     if len(self.pending) < self.threadn:
            #         frame = self.cam.grab()
            #         if not frame is None:
            #             task = self.pool.apply_async(self.process_frame, (frame.copy(), self.clock()))
            #             self.pending.append(task)
            # else:
            frame = self.cam.grab()[:, :, ::-1]
            img_face = self.F.face_extractor(frame)
            pred = 'DEBUG' #self.ml.classify(img_face, ml.dataset.test_transform)
            self.eye_x_t.append((pred, self.clock() - start))

    def close(self):
        try:
            self.pool.terminate()
            self.pool.close()
        except:
            pass
        self.cam.close()

if __name__ == '__main__':
    start = time.time()
    cam = Server()
    ctime = cam.run()
    cam.close()
