import numpy as np

file = np.load("gen/models/Sphero_poly_3.npy", allow_pickle=True)
model = file.item()
print(type(model))
