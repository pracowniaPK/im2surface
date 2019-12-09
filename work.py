import csv
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from structure import (apply_gradient, generate_surface, gradient, 
    interpolate_border, score, surface2im)
from utils import get_name


start_time = time.time()
n = 128
steps = 500
do_border_interpolation = False
# do_border_interpolation = True
imgs_noise = 0.05
grad_coef = 0.0005
log_filename = 'grad.log'
surface_type = 'perlin'

out_num = 6
out_name = get_name(out_num, n, steps, imgs_noise, do_border_interpolation)
gif_steps = 5

vs = [[-1, 2, -2], [-1, -1, -2], [0, 0, -2]]
u = generate_surface(n, type=surface_type, perlin_scale=1.5)
# u = generate_surface(n, type='central')
es = []
for v in vs:
    es.append(surface2im(u, v) 
        + (imgs_noise * np.random.rand(n-2,n-2) - (imgs_noise / 2)))
guess_l = [np.ones([n, n])/2]
s_l = [0]
for e, v in zip(es, vs):
    s_l[-1] += score(guess_l[-1], e, v, per_pixel=True)

grad_work = 0
for i in range(steps-1):
    if 0 == i % np.floor(steps/10):
        print(i, s_l[-1])
    guess_work = guess_l[-1].copy()
    score_work = 0
    for e, v in zip(es, vs):
        grad_work = gradient(guess_work, e, v)
        guess_work = apply_gradient(guess_work, grad_work, grad_coef)
        if do_border_interpolation:
            guess_work = interpolate_border(guess_work)
        score_work += score(guess_work, e, v, per_pixel=True)
    guess_l.append(guess_work)
    s_l.append(score_work)

d_time = time.time() - start_time
print('time: {} s'.format(d_time))


# logging

with open(log_filename, 'a+', newline='') as f:
    cw = csv.writer(f)
    cw.writerow([
        steps,
        n,
        d_time,
        s_l[0],
        np.min(s_l),
        s_l[-1],
        surface_type,
        grad_coef,
        imgs_noise,
        0,
    ])


# plotting

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.set_tight_layout(True)

im = ax1.imshow(guess_l[-1])
cbar = ax1.figure.colorbar(im, ax=ax2)
line, = ax2.plot(np.arange(steps), s_l, '-') 
dot, = ax2.plot([0], [s_l[0]], 'o')

def update(i):
    i = i * gif_steps
    label = 'step {0}'.format(i)
    if 0 == i % np.floor(steps/10):
        print(label)
    
    im.set_data(guess_l[i])
    ax1.set_xlabel(label)
    dot.set_xdata([i])
    dot.set_ydata([s_l[i]])
    # fig.suptitle(label)
    # return im, ax

anim = FuncAnimation(fig, update, 
    frames=np.arange(0, int(len(guess_l)/gif_steps)), interval=200)
anim.save(out_name, dpi=80, writer='imagemagick')

print('out: ' + out_name)

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
