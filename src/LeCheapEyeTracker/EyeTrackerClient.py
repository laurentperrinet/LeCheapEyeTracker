#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The client side.

"""

import numpy as np
import time
import cv2

# VISUALIZATION ROUTINES
from vispy import app
from vispy import gloo

#--------------------------------------------------------------------------

class Stimulation(object):
    """
    A stimulation is an ensemble of stimuli and its properties. This class
    allows creating and running stimulations and is handled by Client.

    To set positions we will use OpenCV's coordinate system in which the origin
    is at the top-right on the screen and coordinates are normalized.

    """

#----Public methods----

    def __init__(self, window_w, window_h, stim_type = 'calibration'):
        self.stim_type = stim_type
        self.window_h = window_h
        self.window_w = window_w
        self.stim_details()

    def run(self):
        """
        Provide a continuum-like use of the class
        """
        t0 = time.time()
        i = 0
        while (time.time()-t0 < self.duration or i <= len(self.tabPos)-1):
            if time.time()-t0 < self.transition_lag*(i+1):
                image = self.draw_stimulus(self.tabPos[i])
                #Use image as you like (Feeding a vizualisation tool)
            print ("stim en cours")
            i += 1

    def get_stimulus(self, t0, t):
        """
        Provide a discrete use of the class
        Return a cliche of stimulation at t, t0 the start of stimulation
        """
        i = int((t-t0)//self.transition_lag)
        return self.draw_stimulus(self.tabPos[i])

#----Privates methods-----

    def stim_details(self):
        """
        (private) Set stimulation properties according to stimulation desired type
        """
        if self.stim_type == 'calibration':
            self.duration = 10
            #Origin in OpenCV is at the right-up corner
            self.tabPos = [(0.5, 0.5), (0.5, 0.1), (0.9, 0.5), (0.5, 0.5), (0.5, 0.9), (0.1, 0.5), (0.5, 0.5)]
            self.transition_lag = 1 # how long (in seconds) a calibration dot is shown
            self.stimulus = 'target'
        else :
            print ("This type of stimulation is not implemented for the moment")

    def draw_stimulus(self, pos):
        """
        (private) Draw a stimulus at a normalized position given.
        Compute the 'true' position.
        pos : the stimulus normalized localization

        """
        img0 = np.zeros((self.window_h, self.window_w, 3)).astype(np.uint8)
        posX, posY = pos
        pos = (int(self.window_w*posX), int(self.window_h*posY))
        if self.stimulus == 'target':
            img = img0.copy()
            img = cv2.circle(img, pos, 12, (0, 0, 255), 1)
            img = cv2.circle(img, pos, 3, (0, 0, 255), -1)

        return img
#---------------------------------------------------------------------------
# TODO import these from constants.py
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
    }
"""

class Client(app.Canvas):
    """
    The client initializes and updates the display where stimulations and
    camera take will occur.
    """
    def __init__(self, et, timeline, stim_type='calibration'):
        self.et = et
        self.timeline = timeline
        app.use_app('pyglet')
        app.Canvas.__init__(self, keys='interactive', title=stim_type, fullscreen=True, size=(1280, 720))
        self.width, self.height = self.physical_size
        print ('window size : ', self.physical_size)
        self.stimulation = Stimulation(self.width, self.height, stim_type=stim_type)
        self.program = gloo.Program(vertex, fragment, count=4)
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.program['texture'] = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        gloo.set_viewport(0, 0, self.width, self.height)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.start = time.time()
        self.show()

    def on_resize(self, event):
        self.width, self.height = event.physical_size
        gloo.set_viewport(0, 0, self.width, self.height)

    def on_draw(self, event):
        gloo.clear('black')
        if time.time()-self.start < self.timeline.max(): # + ret.sleep_time*2:
            #image = np.random.rand(self.height, self.width, 3)*255
            image = self.stimulation.get_stimulus(t0 = self.start, t = time.time())
            #frame = self.et.cam.grab()
            self.program['texture'][...] = image.astype(np.uint8).reshape((self.height, self.width, 3))
            #self.program['texture'] = frame
            self.program.draw('triangle_strip')
        else:
            self.close()

    def on_timer(self, event):
        try:
            frame = self.et.cam.grab()
            if not frame is None:
                res, t0 = self.et.process_frame(frame.copy(), self.et.clock())
                self.et.eye_pos.append([res, t0])
        except:
            if not self.et is None: print('could not grab a frame / detect the eye''s position')
        self.update()

if __name__ == '__main__':

    import cv2
    import numpy as np
    from LeCheapEyeTracker import Server

    fps = 100 # the maximum FPS we try in this experiment
    screen = Client(et=Server(), timeline=np.linspace(0, 7., 3.*fps))
    screen.app.run()
