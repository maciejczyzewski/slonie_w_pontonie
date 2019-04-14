import numpy as np
R = np.load("results.npz")["results"]
from pprint import pprint
#print(R)

DEBUG = True

# lewy_abs

def culnorm2(arr):
    a = []; s = 0
    for x in arr:
        v = (s*3+x**(1.12))/(2+2)
        s = v
        a.append(v)
    return a

import math

degarr = {}
def gen_plot(R, name):
    Y_all = [d[name] for d in R]
    X = list(range(0, len(R)))
    #pprint(Y_all)
    Y1 = [row[0] for row in Y_all]
    Y2 = [row[1] for row in Y_all]
    Y3 = [row[2] for row in Y_all]

    Y4 = [max((row[2]+row[1])/2, row[2]) for row in Y_all]
    Y4 = culnorm2(Y4)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.suptitle(name, fontsize=20)

    ax.plot(X, Y1, color="red")
    ax.plot(X, Y2, color="yellow")
    ax.plot(X, Y3, color="green")
    ax.plot(X, Y4, color="blue")

    """
    m,b = np.polyfit(X, Y4, 1)
    ax.plot(X, Y4, 'yo', X, m*X+b, '--k', color="black")
    """
    fit = np.polyfit(X,Y4,1)
    fit_fn = np.poly1d(fit)
    deg = np.rad2deg(np.arctan(fit[0]))
    degnorm = deg*100
    print("FIT for {}".format(name), fit, degnorm)
    degarr[name] = degnorm
    plt.plot(X,Y4, 'bo', X, fit_fn(X), '--k')

    xname = name.split("_")
    xname = [xname[1], xname[0]]
    xname = "_".join(xname)
    print("XNAME", xname)

    if DEBUG:
        plt.savefig('output/{}.png'.format(xname),bbox_inches = 'tight',pad_inches = 0)

net_names = ['lewy_abs', 'prawy_abs', 'prawa_klatka', 'lewa_klatka', 'prawy_biceps', 'lewy_biceps', 'prawe_ramie', 'lewe_ramie', 'prawe_udo', 'lewe_udo', 'prawa_lydka', 'lewa_lydka']

from tqdm import tqdm
for name in tqdm(net_names):
    if name in R[0]:
        gen_plot(R, name)

from glob import glob
for fname in glob("output/*"):
    print(fname)

pprint(degarr)

bal = {'l': [], 'p': []}
[bal[x[0]].append(degarr[x]) for x in degarr]

bal['l'] = sum(bal['l'])/len(bal['l'])
bal['p'] = sum(bal['p'])/len(bal['p'])

print("LEWA", bal['l'])
print("PRAWA", bal['p'])

przetrenowana_strona = abs(bal['l']-bal['p'])>4
przetrenowana_strona_ktora = math.copysign(1, bal['l']-bal['p']) == -1

print("CZY JAKAS STRONA JEST PRZETRENOWANA?", przetrenowana_strona)

import os
os.system("montage input_series/*  -geometry 100x100+1+1 inputseries.png")
os.system("montage output/*  -tile 2x -geometry 300x300+1+1 timeseries_rap.png")
"""
$ montage input_series/*  -geometry 100x100+1+1 inputseries.png
$ montage output/*  -geometry 300x300+1+1 timeseries.png
"""

from jinja2 import Environment, FileSystemLoader
import os

root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(root, 'templates')
env = Environment( loader = FileSystemLoader(templates_dir) )
template = env.get_template('index.html')


filename = os.path.join(root, 'html', 'index.html')
with open(filename, 'w') as fh:
    fh.write(template.render(
        przetrenowana_strona = przetrenowana_strona,
        przetrenowana_strona_ktora = przetrenowana_strona_ktora,
        show_two = False,
        names    = ["Foo", "Bar", "Qux"],
    ))

# jinja2???? FIXME

# FIXME: rysunek z zaznaczona wada

# (1) inputs
# (2) tabela
# (3) wykresy
# (4) komunikaty
# (5) moze ten czlowieczek???

#pprint(X)

# wykresy dla miesni w czasie
# ------> duzo ich zrobic
