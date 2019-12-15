import csv

from sympy import latex

pl = lambda s: print(latex(s))

def get_name(im_number, n, steps, gamma, noise, noise2, interpolation_on=False):
    """returns name of output file based on chosen options"""
    name = 'im'
    name += '_{}_n{}_s{}_{}_{}_{}'.format(im_number, n, steps, gamma, noise, noise2)
    if interpolation_on:
        name += '_i'
    name += '.gif'
    return name

def init_log(log_name):
    with open(log_name, 'w+', newline='') as f:
        cw = csv.writer(f)
        cw.writerow([
            'liczba kroków',
            'rozmiar powierzchni',
            'czas',
            'score w 0 kroku',
            'score min',
            'score na koniec',
            'rodzaj powierzchni startowej',
            'gamma_gradientu',
            'szum na obrazkach',
            'szum na powierzchni roboczej',
        ])

