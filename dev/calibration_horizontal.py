import cv2
import time
from LeCheapEyeTracker import Client, Server
from vispy import app
import numpy as np

et = Server()
T, fps = 7., 30
timeline = np.linspace(0, T, T*fps)

screen = Client(et, timeline)
app.run()
et.close()
print(screen.et.eye_pos)