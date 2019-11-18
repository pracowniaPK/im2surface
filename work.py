import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from structure import apply_gradient, generate_surface, gradient, score, surface2ims


n = 25
vs = [[-1, 2, -2], [-1, -1, -2], [0, 0, -2]]
u = generate_surface(n, type='perlin', perlin_scale=1.2)
u = generate_surface(n, type='central')

es = surface2ims(u, vs)
guess_l = [np.ones([n, n])/2]
grad_l = [gradient(guess_l[-1], es, vs)]
s_l = [score(guess_l[-1], es, vs, per_pixel=True)]
for i in range(250):
    if i % 20 is 0:
        print(i, s_l[-1])
    grad_l.append(gradient(guess_l[-1], es, vs))
    for grd in grad_l[-1]:
        guess_l.append(apply_gradient(guess_l[-1], grd, 0.005))
    s_l.append(score(guess_l[-1], es, vs, per_pixel=True))




# t = np.arange(len(s_l))
# plt.plot(t, s_l)
# plt.show()

cm = plt.cm.get_cmap('viridis')

plt.subplot(231)
im = plt.imshow(u, cmap=cm)
plt.colorbar(im)

plt.subplot(232)
im = plt.imshow(guess_l[2], cmap=cm)
# plt.contour(e1, colors='black')
plt.colorbar(im)

plt.subplot(233)
im = plt.imshow(guess_l[5], cmap=cm)
# plt.contour(s, colors='black')
plt.colorbar(im)

plt.subplot(234)
im = plt.imshow(guess_l[20], cmap=cm)
plt.colorbar(im)

plt.subplot(235)
im = plt.imshow(guess_l[400], cmap=cm)
# plt.contour(e1, colors='black')
plt.colorbar(im)

plt.subplot(236)
im = plt.imshow(guess_l[-1], cmap=cm)
# plt.contour(s, colors='black')
plt.colorbar(im)

plt.show()



def plot_grid(frames, counts):
    x = len(frames)
    y = len(counts)
    
    cm = plt.cm.get_cmap('viridis')
    for i in range(x):
        for j in range(y):
            print(i, j, 1 + j + i*x)
            plt.subplot(x, y, 1 + j + i*y)
            im = plt.imshow(frames[i][counts[j]], cmap=cm)
            plt.colorbar(im)
    plt.show()

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
