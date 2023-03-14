import visualizer as vis
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grav_sim as g
import numpy as np

filename = "test200"

# n_samples = 200
# n_components = 2

# fig, ax = plt.subplots()

# X, y_true = make_blobs(
#     n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
# )
# X = X[:, ::-1]

# # make the array of positions into an array of dummy particles
# particles = np.array([g.Particle((pos[0], pos[1]), 1, [0, 0]) for pos in X])

# g.record_simulation(particles, 100, filename)

vis.animate_simulation(filename)