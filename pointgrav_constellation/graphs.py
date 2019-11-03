import numpy as np
import os

from matplotlib import pyplot as plt


plt.figure(1)

extra_tars = 12
iterations = 50

for extra in range(1, extra_tars + 1):
    for iteration in range(iterations):

        path = f"data/tars/targets_{extra}_{iteration}.csv"
        if os.path.exists(path):
            coors = np.genfromtxt(path, delimiter=",")

            if coors.ndim != 2:
                coors = np.array([coors])

            plt.plot(coors[:, 0]*180/np.pi, coors[:, 1]*180/np.pi, c="#8262AB", marker="o")

plt.show()
