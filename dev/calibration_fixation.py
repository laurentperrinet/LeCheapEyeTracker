import cv2
import time
from LeCheapEyeTracker import Server, Client
from vispy import app
import numpy as np

N_frame = 42
et = Server()
img0 = et.cam.grab()
def stim(t):
    img0 = et.cam.grab()
    H, W, three = img0.shape
    img = img0.copy()
    img = cv2.circle(img, (W//2, H//2), 12, (0,0,255), -1)
    return img

T, fps = 7., 30
timeline = np.linspace(0, T, T*fps)
screen = Client(et, timeline)
app.run()
et.close()