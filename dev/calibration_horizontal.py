import cv2
import time
from LeCheapEyeTracker import Client, Server
from vispy import app
import numpy as np

N_frame = 42
et = Server()
img0 = et.cam.grab()
H, W, three = img0.shape
img0 = np.zeros_like(img0)
fps = 100
timeline = np.linspace(0, 7., 3*fps)
def stim(t):
    img = img0.copy()
    pos = W/2 + W/2 * np.sin(2*np.pi*t)
    img = cv2.circle(img, (int(pos), H//2), 12, (0,0,255), -1)
    return img

screen = Client(et, timeline)
app.run()
et.close()
print(screen.et.eye_pos)