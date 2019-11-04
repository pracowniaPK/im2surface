import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

from formulas import e1_f


n = 20
u0 = np.zeros([n, n])
e1 = np.zeros([n-2, n-2])
p1 = [1, 1, 1]
e2 = np.zeros([n-2, n-2])
p2 = [-1, 1, 1]


for i in range(n):
    for j in range(n):
        # print([i, j])
        u0[i, j] = pnoise2(i/n, j/n)

for i in range(1, n-1):
    for j in range(1, n-1):
        e1[i-1, j-1] = e1_f(u0[i, j-1], u0[i+1, j], u0[i, j+1], u0[i-1, j], p1[0], p1[1], p1[2], 1)
        e2[i-1, j-1] = e1_f(u0[i, j-1], u0[i+1, j], u0[i, j+1], u0[i-1, j], p2[0], p2[1], p2[2], 1)

# print(u0)
plt.subplot(131)
im = plt.imshow(u0)
plt.colorbar(im)

plt.subplot(132)
print(e1)
im = plt.imshow(e1)
plt.colorbar(im)

plt.subplot(133)
print(e2)
im = plt.imshow(e2)
plt.colorbar(im)

plt.show()



