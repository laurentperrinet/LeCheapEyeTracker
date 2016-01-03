import cv2
import time
from LeCheapEyeTracker import LeCheapEyeTracker, Canvas
from vispy import app
import numpy as np

N_frame = 42
et = LeCheapEyeTracker()
img0 = et.cam.grab()
def stim(t):
    img0 = et.cam.grab()
    H, W, three = img0.shape
    img = img0.copy()
    img = cv2.circle(img, (W//2, H//2), 12, (0,0,255), -1)
    return img

screen = Canvas(et, (stim, np.linspace(0, 3., 100)))
app.run()
et.close()