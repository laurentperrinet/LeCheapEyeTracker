import numpy as np
import time
from openRetina import PhotoReceptor

def moyFPS(nb, scale):
    N = 100
    ctime = np.zeros(N)
    
    acc = 0
    
    for i in range(nb):
        start = time.time()
        
        cam = PhotoReceptor(scale)
        for j in range(N):
            img = cam.grab()
            ctime[j] = time.time()-start
        cam.close()
        
        acc = acc + N/(ctime[-1]-ctime[0])
    
    return acc/nb
print (moyFPS(16,8))