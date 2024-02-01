import os
import dill
import warnings
import configparser
import numpy as np
import pykoopman as pk
from plot import plot_comparison
from fitK import pkFitK

# suppress warnings
warnings.filterwarnings("ignore")

model_name = "Dubins" # CHANGE THIS
## read from config file
config = configparser.ConfigParser()
config.read("config.cfg")

# load model and parameters
model_cfg = config[model_name]
data_name = model_cfg["DataPath"]
N_x = int(model_cfg["Nx"])
N_u = int(model_cfg["Nu"])
N_d = int(model_cfg["Nd"])
t0 = float(model_cfg["t0"])
th = float(model_cfg["th"])
obs_tags = model_cfg["obs"].split(", ")


current_dir = os.getcwd()
target_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
path = os.path.join(target_dir, "Data", data_name)
Xfull = np.load(path)

# compute other parameters
tf = th * (Xfull.shape[1] - 1)
N_xud = [N_x, N_u, N_d]
t_arr = np.arange(t0, tf + th, th)
n_plots, N_T = (
    24,
    Xfull.shape[1],
)  # Number of trajectories to plot, Length of ea. Trajectory

## separate train and test data
train_p = 0.6
idxs = np.arange(Xfull.shape[2])
np.random.shuffle(idxs)
plot_idxs = idxs[:n_plots]

num_train_samples = int(Xfull.shape[2] * (train_p))
train_ix = idxs[:num_train_samples]
test_ix = idxs[num_train_samples:]

# print(Xfull.shape)
for obs_tag in obs_tags:
    model = pkFitK(Xfull, train_ix, obs_tag, N_xud)
    print(type(model))

    path = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path = os.path.join(path, "gen", "models", f"{model_name}_{obs_tag}.npy")
    print(path)
    # with open(path, "wb") as file:
    np.save(path, model, allow_pickle=True)

    ## Compare True and EDMDc
    plot_comparison(
        model_name, obs_tag, Xfull, N_T, t_arr, N_xud, plot_idxs, model, train_ix
    )

    # save in gen/models/{model_name}/{obs_tag}/ using dill and name it {model_name}_{obs_tag}_{obs_params}.dill
