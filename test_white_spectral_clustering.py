import numpy as np
from diffusions import _build_laplacian
import pyamg

l = 100
X, Y = np.indices((l, l))

c1 = (28, 24)
c2 = (40, 50)
c3 = (67, 58)
c4 = (24, 80)
c5 = (83, 34)

r1, r2, r3, r4, r5 = 16, 14, 15, 24, 14

m1 = (X - c1[0])**2 + (Y - c1[1])**2 < r1**2
m2 = (X - c2[0])**2 + (Y - c2[1])**2 < r2**2
m3 = (X - c3[0])**2 + (Y - c3[1])**2 < r3**2
m4 = (X - c4[0])**2 + (Y - c4[1])**2 < r4**2
m5 = (X - c5[0])**2 + (Y - c5[1])**2 < r5**2


#m1[::4, ::4] = False
m = (m1 + m2 + m3 + m4).astype(bool)
mask = np.copy(m)
m = m.astype(float)

m /= 2
m += 0.5

m += 0.1 * np.random.randn(*m.shape)

res, scores = separate_in_regions(m, mask, mode='amg', take_first=True)
