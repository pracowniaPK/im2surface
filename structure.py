import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

from formulas import e1_f


n = 50
u0 = np.zeros([n, n])
e1 = np.zeros([n-2, n-2])
e2 = np.zeros([n-2, n-2])
e3 = np.zeros([n-2, n-2])
p = [[1, 1, 1], [0, 1, 1], [-1, 1, 1]]


# generowanie losowej powierzchni
for i in range(n):
    for j in range(n):
        # print([i, j])
        u0[i, j] = pnoise2(i/n, j/n)

# liczenie E dla źródeł światła
for i in range(1, n-1):
    for j in range(1, n-1):
        e1[i-1, j-1] = e1_f(u0[i, j-1], u0[i+1, j], u0[i, j+1], u0[i-1, j], p[0][0], p[0][1], p[0][2], 1)
        e2[i-1, j-1] = e1_f(u0[i, j-1], u0[i+1, j], u0[i, j+1], u0[i-1, j], p[1][0], p[1][1], p[1][2], 1)
        e3[i-1, j-1] = e1_f(u0[i, j-1], u0[i+1, j], u0[i, j+1], u0[i-1, j], p[2][0], p[2][1], p[2][2], 1)

# generowanie e z szumem (na potrzeby liczenia miary błędu)
d = 0.005
g1 = e1.copy()
g1 = g1 + np.random.rand(n-2, n-2)*d - d/2

# # liczenie błędu
norm = false
score = 0
for i in range(n):
    for j in range(n):
        pass


# print(u0)
plt.subplot(221)
im = plt.imshow(u0)
plt.colorbar(im)

plt.subplot(222)
im = plt.imshow(e1)
plt.colorbar(im)

plt.subplot(223)
im = plt.imshow(g1)
plt.colorbar(im)

plt.subplot(224)
im = plt.imshow(e1 - g1)
plt.colorbar(im)

plt.show()



