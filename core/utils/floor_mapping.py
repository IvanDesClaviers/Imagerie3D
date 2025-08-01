from global_config import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mats_to_compare = []
    path = os.path.join(FILES_DIR, "floorcomp")
    for filename in sorted(os.listdir(path)):
        file = open(os.path.join(path, filename), 'r')
        lines = file.readlines()

        mat = np.zeros((100, 100, 5))
        for line in lines:
            if "=" in line:
                string = line.split('=')
                indexes = string[0].split('][')
                i, j = int(indexes[0].replace('[', '')), int(indexes[1].replace(']', ''))
                vals = string[1].split(',')
                for k in range(5):
                    mat[i, j, k] = float(vals[k].strip())

        mats_to_compare.append(mat)
    diff_mat = mats_to_compare[0][:, :, 2] - mats_to_compare[1][:, :, 2]

    nbpt = 100
    X1 = mats_to_compare[0][:, :, 0].flatten()
    Y1 = mats_to_compare[0][:, :, 1].flatten(),
    Z1 = mats_to_compare[0][:, :, 2].flatten()

    X2 = mats_to_compare[1][:, :, 0].flatten()
    Y2 = mats_to_compare[1][:, :, 1].flatten(),
    Z2 = mats_to_compare[1][:, :, 2].flatten()

    print(diff_mat)

    # Plot X,Y,Z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(X1[0:nbpt], Y1[0:nbpt], Z1[0:nbpt], color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(X1[0:nbpt], Y1[0:nbpt], Z1[0:nbpt], c='red')

    ax.plot_trisurf(X2[0:nbpt], Y2[0:nbpt], Z2[0:nbpt], color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(X2[0:nbpt], Y2[0:nbpt], Z2[0:nbpt], c='green')
    plt.show()
