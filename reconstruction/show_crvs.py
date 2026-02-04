import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    nrows = 2
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols)

    for r, fname in enumerate(("solve_3d_pts.json", "solve_3d_pts2.json")):
        with open(fname, "r") as f:
            my_data = json.load(f)

        pts_background = np.array(my_data["background_pts"]).transpose()
        pts_crvs = []
        for crv in my_data["crv_pts"]:
            pts_crvs.append(np.array(crv).transpose())

        dims = [(0, 1), (2, 1)]
        for ic, ds in enumerate(dims):
            cols = ["r-", "g:", "b--"]
            # axs[r, ic].scatter(x=pts_background[ds[0], :], y=pts_background[ds[1], :], c='k', marker='x')
            for crv, col in zip(pts_crvs, cols):
                axs[r, ic].plot(crv[ds[0], :], crv[ds[1], :], col)
                axs[r, ic].plot(crv[ds[0], 0], crv[ds[1], 0], 'x')

            axs[r, ic].set_aspect('equal')
    plt.show()
    print(f"done")