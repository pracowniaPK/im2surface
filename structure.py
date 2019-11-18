import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

from formulas import e1_f, de4_f


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

def surface2ims(u, vs):
    """calculates images of given surface

    u - surface (heights matrix)
    vs - light vectors
    """
    n = u.shape[0]
    es = []
    for v in vs:
        es.append(np.zeros([n-2, n-2]))
        for i in range(n-2):
            for j in range(n-2):
                es[-1][i, j] = e1_f(u[i+1, j], u[i+2, j+1], u[i+1, j+2], u[i, j+1], v[0], v[1], v[2], 2/n)
    
    return es

def score(guess, es, vs, per_pixel=False):
    """calculates cost function of given surface

    guess - surface to measure
    es - images of original surface
    vs - light vector used to iluminate original surface
    per_pixel - if True, cost value is divided by number comparision points
    """
    costs = []
    for e, v in zip(es, vs):
        e_guess = surface2ims(guess, [v])[0]
        costs.append((e_guess - e)**2)
    if per_pixel:
        per_pixel_cost = 0
        n = 0
        for c in costs:
            n += 1
            per_pixel_cost += np.average(c)
        costs = per_pixel_cost/n
    return costs

def gradient(guess, es, vs):
    """calculates gradient of cost function

    guess - work surface
    es - images of original surface
    vs - light vector used to iluminate original surface
    """
    n = guess.shape[0]
    des = []
    for e, v in zip(es, vs):
        des.append(np.zeros([n-6, n-6]))
        for i in range(n-6):
            for j in range(n-6):
                des[-1][i, j] = de4_f(
                    guess[i+3, j+3],
                    guess[i+3, j+1], guess[i+4, j+2], guess[i+5, j+3], guess[i+4, j+4], 
                    guess[i+3, j+5], guess[i+2, j+4], guess[i+1, j+3], guess[i+2, j+2], 
                    e[i+2, j+1], e[i+3, j+2], e[i+2, j+3], e[i+1, j+2], 
                    v[0], v[1], v[2], 2/n
                )
    return des

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
            # print(u_new[i+3, j+3], u[i+3, j+3], k*grd[i, j])
    return guess_new
