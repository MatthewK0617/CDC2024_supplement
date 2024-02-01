import numpy as np
import matplotlib.pyplot as plt


def histogram(name):
    norms_all = np.load(name)
    print(norms_all.shape)

    norms_all = norms_all.flatten()
    norms_all_weights = np.ones_like(norms_all) / len(norms_all)

    # Create the histogram
    plt.hist(
        norms_all, weights=norms_all_weights
    )  # You can adjust the number of bins as needed
    plt.title("Histogram of Norms")
    plt.xlabel("Loss Value (norms)")
    plt.ylabel("Frequency")
    plt.show()
    # plt.savefig("gen/n_step_norms/plots/")


histogram("gen/n_step_norms/sphero_poly_3/s1_norms_all.npy")
histogram("gen/n_step_norms/sphero_poly_3/s5_norms_all.npy")
histogram("gen/n_step_norms/sphero_poly_3/s10_norms_all.npy")
