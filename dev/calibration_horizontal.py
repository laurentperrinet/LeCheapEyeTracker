import cv2
import time
from LeCheapEyeTracker import Camera, Canvas
from vispy import app
import numpy as np

N_frame = 42
cam = Camera()
img0 = cam.grab()
H, W, three = img0.shape
img0 = np.zeros_like(img0)
timeline = np.linspace(0, 8., 100)
def stim(t):
    img = img0.copy()
    pos = W/2 + W/2 * np.sin(2*np.pi*t)
    img = cv2.circle(img, (int(pos), H//2), 12, (0,0,255), -1)
    return img

screen = Canvas(cam, (stim, timeline))
app.run()
cam.close()
print(screen.cam.eye_pos)