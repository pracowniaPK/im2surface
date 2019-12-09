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
