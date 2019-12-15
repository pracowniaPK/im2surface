from multiprocessing import Pool, Process
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from structure import (apply_gradient, basic_grad_loop, generate_surface, 
    gradient, interpolate_border, score, surface2im)
from utils import get_name


def f(u_n, i_n):
    basic_grad_loop(
        64, 500, 0.0005,
        surface_type='central',
        guess_type='u',
        imgs_noise=i_n,
        u_noise=u_n,
        gif_batch_name='noise',
        gif_steps=5,
        log_filename='noise_img.log',
    )
    
if __name__ == "__main__":

    pool = Pool(3)

    pool.starmap(f, product([i*0.02 for i in [2, 10]], [0.01, 0.05, 0.1, 0.2]))


    # ps = []

    # for i in range(1,10):
    #     ps.append(
    #         Process(target=basic_grad_loop,
    #         args=(64, 200, 0.0003),
    #         kwargs={
    #             'guess_type':'u',
    #             'u_noise':0.02*i,
    #             'gif_batch_name':'tmp',
    #             'log_filename':'tmp.log',
    #         }
    #         )
    #     )


    # for p in ps:
    #     p.start()

    # for p in ps:
    #     p.join()


    # for i in range(1):
    #     p = Process(target=f, args=(10, 0.001*(i+1)))
    #     p.start()


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
