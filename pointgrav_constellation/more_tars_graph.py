import numpy as np

from matplotlib import pyplot as plt

from constants import _target_coors_deg as target_coors


def init_graph():
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    plt.xticks(np.linspace(-180, 180, 13))
    plt.yticks(np.linspace(-90, 90, 13))

    plt.plot(target_coors[:, 1] - (target_coors[:, 1] > 180)*360, target_coors[:, 0], 'kx')

path = "data/tars/targets_{extra}_{iteration}.csv"

extra_tars = 12
iterations = 50

colors = ["b", "r", "g"]

for extra in range(1, extra_tars + 1):
    plt.figure(extra + 1)

    init_graph()

    for iteration in range(iterations):
        try:
            coors = np.genfromtxt(path.format(extra=extra, iteration=iteration), delimiter=",")

            if extra == 1:
                plt.plot((coors[1] - (coors[1] > np.pi)*2*np.pi)*180/np.pi, coors[0]*180/np.pi, "bo")

            else:
                for coor in coors:
                    plt.text((coor[1] - (coor[1] > np.pi)*2*np.pi)*180/np.pi, coor[0]*180/np.pi, f"{iteration + 1}")

        except OSError:
            pass
    
    # if extra == 2:
    #     break

plt.show()