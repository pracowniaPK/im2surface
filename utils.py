import csv

from sympy import latex

pl = lambda s: print(latex(s))

def get_name(im_number, n, steps, noise, interpolation_on=False):
    """returns name of output file based on chosen options"""
    noise = str(noise).replace('.', 'o')
    name = 'im'
    name += '{}_n{}_s{}_{}'.format(im_number, n, steps, noise)
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

