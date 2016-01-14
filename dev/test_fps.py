#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
from openRetina import PhotoReceptor

def moyFPS(nb_trials, downscale, N_frame = 100):
    
    acc = []
    
    for i in range(nb_trials):
        start = time.time()
        
        cam = PhotoReceptor(DOWNSCALE=downscale)
        for j in range(N_frame):
            img = cam.grab()
        cam.close()

        stop = time.time()
        acc.append(N_frame/(stop-start))
    
    return np.array(acc)

def pretty_print(acc):
    return(' 🍺 FPS = {mean} +/- {std} (in frames per second)'.format(mean=acc.mean(), std=acc.std()))

if __name__ == '__main__':

    for downscale in [16, 8, 4, 2, 1]:
        print ('Downscale = %d ' % downscale)
        print (pretty_print((moyFPS(nb_trials=16, downscale=downscale, N_frame=100))))