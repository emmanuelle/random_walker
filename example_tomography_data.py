from random_walker import random_walker
import numpy as np
import matplotlib.pyplot as plt
try:
    import pyamg
    amg_loaded = True
except ImportError:
    amg_loaded = False
import time

# Data
filename = 'raw_tomography_data.dat'
data = np.fromfile(filename, dtype=np.float32)
data.shape = (80, 200, 200)


# Build markers
markers = np.zeros_like(data)
markers[data < -0.6] = 1
markers[data > 0.6] = 2

# Segmentation
if amg_loaded:
    t1 = time.time()
    labels = random_walker(data, markers, mode='cg_mg', beta=50, tol=5.e-3)
    t2 = time.time()
    print t2 - t1
else:
    labels = random_walker(data[:10], markers[:10], mode='cg', beta=100) 

# Show results
i = 20
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(data[i], cmap=plt.cm.gray, vmin=-2, vmax=2,interpolation='nearest')
plt.contour(labels[i], [1.5])
plt.title('data')
plt.subplot(122)
plt.imshow(markers[i], cmap=plt.cm.gray)
plt.title('markers')
plt.show()
