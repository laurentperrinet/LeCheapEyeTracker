#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The client side.

"""

import numpy as np
import time


class Stimulation():

    def __init__


    def run(t):

   

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
    }
"""

class Client(app.Canvas):
    """
    
    Coucou
    
    """
    def __init__(self, et, stim, timeline):
        self.et = et
        self.stim, self.timeline = stim, timeline
        self.h, self.w, three = self.stim(0).shape
        app.use_app('pyglet')
        app.Canvas.__init__(self, keys='interactive', fullscreen=True)#, size=(1280, 960))#
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
    img0 = np.zeros((780, 1280, 3)).astype(np.uint8)
    H, W, three = img0.shape

    def stim(t):
        img = img0.copy()
        pos = W/2 + .8 * W/2 * np.sin(2*np.pi*t)
        img = cv2.circle(img, (int(pos), H//2), 12, (0,0,255), -1)
        return img

    fps = 100
    screen = Client(et=None, stim=stim, timeline=np.linspace(0, 3., 3.*fps))
    app.run()

