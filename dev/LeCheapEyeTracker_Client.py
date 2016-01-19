
import numpy as np
import zmq

from vispy import app
from vispy import gloo
from LeCheapEyeTracker import *

# At this point, we need to receive the stim from the server

#eyeT = LeCheapEyeTracker()
#scene = Canvas(eyeT, )

context = zmq.Context()

# Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    socket.send(b"Hello")

    # Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (request, message))

#eyeT = LeCheapEyeTracker()
#scene = Canvas(eyeT, )