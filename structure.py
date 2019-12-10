import csv
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from noise import pnoise2

from formulas import e1_f, de4_f
from utils import get_name


def generate_surface(n, type='perlin', perlin_scale=2):
    """generate surfaces to work on

    n - size of output matrix
    type - chosen algorithm to generate surface
    """
    u = np.zeros([n, n])
    if type is 'perlin':
        for i in range(n):
            for j in range(n):
                u[i, j] = pnoise2(perlin_scale*i/n, perlin_scale*j/n)
    if type is 'central':
        for i in range(n):
            for j in range(n):
                u[i, j] = ((n/2 - i)/(n/2))**2 + ((n/2 - j)/(n/2))**2

    return u

def surface2im(u, v):
    """calculates images of given surface

    u - surface (heights matrix)
    v - light vector
    """
    n = u.shape[0]
    e = np.zeros([n-2, n-2])
    for i in range(n-2):
        for j in range(n-2):
            e[i, j] = e1_f(u[i+1, j], u[i+2, j+1], u[i+1, j+2], u[i, j+1], v[0], v[1], v[2], 2/n)
    
    return e

def score(guess, e, v, per_pixel=False):
    """calculates cost function of given surface

    guess - surface to measure
    e - image of original surface
    v - light vector used to iluminate original surface
    per_pixel - if True, cost value is divided by number comparision points
    """
    e_guess = surface2im(guess, v)
    cost = (e_guess - e)**2
    if per_pixel:
        cost = np.average(cost)
    return cost

def gradient(guess, e, v):
    """calculates gradient of cost function

    guess - work surface
    e - image of original surface
    v - light vector used to iluminate original surface
    """
    n = guess.shape[0]
    de = np.zeros([n-6, n-6])
    for i in range(n-6):
        for j in range(n-6):
            de[i, j] = de4_f(
                guess[i+3, j+3],
                guess[i+3, j+1], guess[i+4, j+2], guess[i+5, j+3], guess[i+4, j+4], 
                guess[i+3, j+5], guess[i+2, j+4], guess[i+1, j+3], guess[i+2, j+2], 
                e[i+2, j+1], e[i+3, j+2], e[i+2, j+3], e[i+1, j+2], 
                v[0], v[1], v[2], 2/n
            )
    return de

def apply_gradient(guess, grd, k):
    """applies gradient to the guess as optimizatoin step

    guess - surface to apply gradient to
    grd - gradient
    k - step size coefficient
    """
    n = guess.shape[0]
    guess_new = guess.copy()
    for i in range(n-6):
        for j in range(n-6):
            guess_new[i+3, j+3] = guess[i+3, j+3] - k*grd[i, j]
    return guess_new

def interpolate_border(guess):
    """interpolates 3px border of surface

    guess - surface with border to smooth
    """
    n = guess.shape[0]
    guess_new = guess.copy()
    for i in range(n):
        d1 = guess_new[i, 3] - guess_new[i, 4]
        guess_new[i, 2] = guess_new[i, 3] + d1
        guess_new[i, 1] = guess_new[i, 2] + d1
        guess_new[i, 0] = guess_new[i, 1] + d1
        d2 = guess_new[i, n-4] - guess_new[i, n-5]
        guess_new[i, n-3] = guess_new[i, n-4] + d2
        guess_new[i, n-2] = guess_new[i, n-3] + d2
        guess_new[i, n-1] = guess_new[i, n-2] + d2
        d3 = guess_new[3, i] - guess_new[4, i]
        guess_new[2, i] = guess_new[3, i] + d3
        guess_new[1, i] = guess_new[2, i] + d3
        guess_new[0, i] = guess_new[1, i] + d3
        d4 = guess_new[n-4, i] - guess_new[n-5, i]
        guess_new[n-3, i] = guess_new[n-4, i] + d4
        guess_new[n-2, i] = guess_new[n-3, i] + d4
        guess_new[n-1, i] = guess_new[n-2, i] + d4
    return guess_new

# n = 128
# steps = 500
# do_border_interpolation = False
# imgs_noise = 0.05
# grad_coef = 0.0005
# log_filename = 'grad.log'
# surface_type = 'perlin'
# gif_batch_name = 6
# gif_steps = 5
def basic_grad_loop(n, steps, imgs_noise, grad_coef, 
    do_border_interpolation=False,
    surface_type='central', do_log=True, log_filename='grad.log', 
    do_gif=True, gif_batch_name='', gif_steps=1):

    start_time = time.time()

    vs = [[-1, 2, -2], [-1, -1, -2], [0, 0, -2]]
    u = generate_surface(n, type=surface_type, perlin_scale=1.2)
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
    if do_log:
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
    if do_gif:
        out_name = get_name(gif_batch_name, n, steps, grad_coef, imgs_noise, do_border_interpolation)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.set_tight_layout(True)

        ax1.set_xlabel('γ:{}\nmin:{}'.format(grad_coef, np.min(s_l)))
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
            ax1.set_title(label)
            dot.set_xdata([i])
            dot.set_ydata([s_l[i]])
            # fig.suptitle(label)
            # return im, ax

        anim = FuncAnimation(fig, update, 
            frames=np.arange(0, int(len(guess_l)/gif_steps)), interval=200)
        anim.save('plots/'+out_name, dpi=80, writer='imagemagick')

        print('out: ' + out_name)
