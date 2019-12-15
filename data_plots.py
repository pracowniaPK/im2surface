import csv

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.subplots()

X = []
Y = []

with open('tmp.log', 'r') as f:
    cr = csv.reader(f)
    for row in cr:
        if row[1] == '500':
            X.append(float(row[3]))
            Y.append(float(row[5]))

for z in zip(X, Y):
    print(z)

ax.plot(X, Y, '+')
plt.show()
