import grav_sim as g
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_tree (tree, particles, fig, ax):
    """
    Plot the tree generated by grav_sim for given particles, fig, ax
    """
    # the reason we can't store this information in the tree even though we 
    # had to compute for the tree construction is that the tree will be kept
    # in memory during simulation, and we don't need to store the cell dimensions
    # in the tree structure for the simulation to work
    positions = np.array([p.position for p in particles])
    min_x = np.amin(positions[:, 0])
    min_y = np.amin(positions[:, 1])
    max_x = np.amax(positions[:, 0])
    max_y = np.amax(positions[:, 1])
    square_side = max(max_x - min_x, max_y - min_y)
    lower_left = np.array([min_x, min_y]) 

    _plot(tree, lower_left, square_side, fig, ax)

def _plot (tree, ll, side, fig, ax):
    """
    Recursive helper method for the plot_tree function
    """
    # plot this square if it non-zero mass
    if tree.mass > 0:
        rect = patches.Rectangle(ll, side, side, linewidth=1, 
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if tree.children:
        # subdivide this square
        half_side = side / 2
        x, y = ll
        ll_1 = [x + half_side, y + half_side]
        ll_2 = [x, y + half_side]
        ll_3 = [x, y]
        ll_4 = [x + half_side, y]
        lls = [ll_1, ll_2, ll_3, ll_4]

        for i in range(4):
            _plot(tree.children[i], lls[i], half_side, fig, ax)



