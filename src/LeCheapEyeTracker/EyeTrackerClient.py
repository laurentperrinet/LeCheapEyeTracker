#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The client side.

"""

import numpy as np
import time
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
        // HACK: the image is in BGR instead of RGB.
        float temp = gl_FragColor.r;
        gl_FragColor.r = gl_FragColor.b;
        gl_FragColor.b = temp;
    }
"""

class Client(app.Canvas):
    """
    
    Coucou
    
    """
    def __init__(self, et, stim):
        self.et = et
        self.stim, self.timeline = stim
        self.h, self.w, three = self.stim(0).shape
        app.use_app('pyglet')
        app.Canvas.__init__(self, keys='interactive', fullscreen=True, size=(1280, 960))#
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
        frame = self.et.cam.grab()
        if not frame is None:
            res, t0 = self.et.process_frame (frame.copy(), self.et.clock())
            self.et.eye_pos.append([res, t0])
        self.update()

if __name__ == '__main__':
    start = time.time()
    cam = ThreadSource()
    ctime = cam.run()
    cam.close()


