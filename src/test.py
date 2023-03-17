from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grav_sim as g
import visualizer as vis
import numpy as np

particles = g.create_particle_blobs(500, 1, [0, 0], 5)
tree = g.generate_tree(particles)

fig, ax = plt.subplots()
vis.plot_tree(tree, particles, fig, ax)