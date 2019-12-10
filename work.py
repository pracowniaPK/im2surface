from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from structure import (apply_gradient, basic_grad_loop, generate_surface, 
    gradient, interpolate_border, score, surface2im)
from utils import get_name


def f(s, g):
    basic_grad_loop(64, s, 0, g, gif_batch_name='_tst', 
        do_border_interpolation=False, surface_type='perlin')
    
if __name__ == "__main__":

    for i in range(1):
        p = Process(target=f, args=(50, 0.001*(i+1)))
        p.start()


# t = np.arange(len(s_l))
# plt.plot(t, s_l)
# plt.show()

# cm = plt.cm.get_cmap('viridis')

# plt.subplot(231)
# im = plt.imshow(u, cmap=cm)
# plt.colorbar(im)

# plt.subplot(232)
# im = plt.imshow(guess_l[1], cmap=cm)
# # plt.contour(e1, colors='black')
# plt.colorbar(im)

# plt.subplot(233)
# im = plt.imshow(guess_l[10], cmap=cm)
# # plt.contour(s, colors='black')
# plt.colorbar(im)

# plt.subplot(234)
# im = plt.imshow(guess_l[50], cmap=cm)
# plt.colorbar(im)

# plt.subplot(235)
# im = plt.imshow(guess_l[100], cmap=cm)
# # plt.contour(e1, colors='black')
# plt.colorbar(im)

# plt.subplot(236)
# im = plt.imshow(guess_l[-1], cmap=cm)
# # plt.contour(s, colors='black')
# plt.colorbar(im)

# plt.show()

# def plot_grid(frames, counts):
#     x = len(frames)
#     y = len(counts)
    
#     cm = plt.cm.get_cmap('viridis')
#     for i in range(x):
#         for j in range(y):
#             print(i, j, 1 + j + i*x)
#             plt.subplot(x, y, 1 + j + i*y)
#             im = plt.imshow(frames[i][counts[j]], cmap=cm)
#             plt.colorbar(im)
#     plt.show()

# plot_grid([guess_l, grad_l], [0, 199, -1])



# przykładowy obrazek:
# cm = plt.cm.get_cmap('gray')
# cm2 = plt.cm.get_cmap('plasma')
# fig = plt.figure()

# plt.subplot(232)
# im = plt.imshow(es[0], cmap=cm)

# plt.subplot(233)
# im = plt.imshow(es[1], cmap=cm)

# plt.subplot(235)
# im = plt.imshow(es[2], cmap=cm)

# plt.subplot(236)
# im = plt.imshow(es[3], cmap=cm)

# plt.subplot(234)
# im = plt.imshow(es[-1], cmap=cm)

# X, Y = np.mgrid[0:n, 0:n]
# ax = fig.add_subplot(2, 3, 1, projection='3d')
# ax.plot_surface(X, Y, u, cmap=cm2, linewidth=0, antialiased=False)
# ax.plot([0, 0], [0, 0], [2, -1], 'w')

# plt.show()
