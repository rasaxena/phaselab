import numpy as np
import thread
import time
from scipy import signal

unknown_positions = np.random.rand(40,16*99,16*99)
conv = np.random.rand(16*99,16*99)
# known_positions = np.random.rand(400*99*99,256)
for ii, inx in enumerate(unknown_positions):

    bling = signal.fftconvolve(inx, conv)

    print(np.max(bling))

#
# for inx in unknown_positions:
#     # for inx in range(len(unknown_positions)):
#     error = np.abs(np.subtract(known_positions, inx))
#     # print(error)
#     # error = np.square(error)
#     # print(error)
#     error = np.sum(error,axis=1)
#     # print(error)
#     smallest = np.min(error)
#     print(smallest)
#     print "%s: %s" % (threadName, time.ctime(time.time()))
