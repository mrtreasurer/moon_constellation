import numpy as np

from matplotlib import pyplot as plt

from constants import _target_coors_deg as target_coors


plt.figure(1)

plt.xlim(-180, 180)
plt.ylim(-90, 90)

plt.xticks(np.linspace(-180, 180, 13))
plt.yticks(np.linspace(-90, 90, 13))

plt.plot(target_coors[:, 1] - (target_coors[:, 1] > 180)*360, target_coors[:, 0], 'ko')

path = "data/tars/targets_{extra}_{iteration}.csv"

extra_tars = 12
iterations = 50

colors = ["b", "r", "g"]

for extra in range(1, extra_tars + 1):
    for iteration in range(iterations):
        try:
            coors = np.genfromtxt(path.format(extra=extra, iteration=iteration), delimiter=",")
            print(iteration, coors)

            if extra == 1:
                plt.plot((coors[1] - (coors[1] > np.pi)*2*np.pi)*180/np.pi, coors[0]*180/np.pi, marker="o", c=colors[extra - 1])

            else:
                plt.plot(coors[:, 1] - (coors[:, 1] > 180)*360, coors[:, 0], marker="o", c=colors[extra - 1])

        except OSError:
            pass
    
    break

plt.show()