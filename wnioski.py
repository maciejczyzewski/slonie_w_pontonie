import numpy as np
import workouts
R = np.load("results.npz")["results"]
from pprint import pprint
#print(R)

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

    plt.savefig('output/{}.png'.format(name),bbox_inches = 'tight',pad_inches = 0)

net_names = ['lewy_abs', 'prawy_abs', 'prawa_klatka', 'lewa_klatka', 'prawy_biceps', 'lewy_biceps', 'prawe_ramie', 'lewe_ramie', 'prawe_udo', 'lewe_udo', 'prawa_lydka', 'lewa_lydka']

from tqdm import tqdm
for name in tqdm(net_names):
    if name in R[0]:
        gen_plot(R, name)

from glob import glob
for fname in glob("output/*"):
    print(fname)

bal = {'l': [], 'p': []}
[bal[x[0]].append(degarr[x]) for x in degarr]

bal['l'] = sum(bal['l'])/len(bal['l'])
bal['p'] = sum(bal['p'])/len(bal['p'])

print("LEWA", bal['l'])
print("PRAWA", bal['p'])
print("min, max: ", min(bal['l'], bal['p']) / max(bal['l'], bal['p']))

dgr = degarr
dgr.pop('prawe_udo')
dgr.pop('lewe_udo')

odpowiedz = ""

odpowiedz += ("Twoim najszybciej rozwijającym się mięśniem jest " + max(degarr, key=degarr.get) + "\n")
odpowiedz += ("Dobra robota! Trzymaj to tempo :)\n")
odpowiedz += ("Twoim najwolniej rozwijającym się mięśniem jest " + min(degarr, key=degarr.get) + ".\n")
odpowiedz += ("Być może problem leży w doborze ćwiczeń. Oto ich propozycje: \n")
if min(degarr, key=degarr.get) in ['lewa_klatka', 'prawa_klatka']:
    odpowiedz += (workouts.klatka)
elif min(degarr, key=degarr.get) in ['lewy_biceps', 'prawy_biceps']:
    odpowiedz += (workouts.biceps)
elif min(degarr, key=degarr.get) in ['lewy_abs', 'prawy_abs']:
    odpowiedz += (workouts.brzuch)
elif min(degarr, key=degarr.get) in ['lewe_udo', 'prawe_udo']:
    odpowiedz += (workouts.uda)
elif min(degarr, key=degarr.get) in ['lewa_lydka', 'prawa_lydka']:
    odpowiedz += (workouts.lydki)
elif min(degarr, key=degarr.get) in ['lewe_ramie', 'prawe_ramie']:
    odpowiedz += (workouts.przedramie)

if min(bal['l'], bal['p']) / max(bal['l'], bal['p']) < 0.8:  # IMBA
    odpowiedz += ("\nOj! Nie ćwiczysz równomiernie.\n")
    if(bal['l'] > bal['p']):
        odpowiedz += ("twoja lewa strona jest większa od prawej o ")
    else:
        odpowiedz += ("twoja prawa strona jest większa od lewej o ")
    odpowiedz += str(min(bal['l'], bal['p']) / max(bal['l'], bal['p']) * 100) + " %\n"
