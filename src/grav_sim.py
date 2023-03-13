"""
An 2D implementation of the Barnes-Hut algorithm
"""
import numpy as np

class Cell:
    def __init__(self, children, particles):
        """
        Input:
            children: list(Cell)
            particles: list(Particle)
        """
        masses = np.array([p.mass for p in particles])
        self.mass = np.sum(masses)
        weighted_pos = np.array([p.position * p.mass for p in particles])
        self.com = np.average(weighted_pos)
        self.children = children  


class Particle:
    def __init__(self, position, mass, velocity):
        self.position = position # note: this is a tuple
        self.mass = mass
        self.velocity = velocity


def generate_tree (particles):
    """
    Generate the full cell tree for this set of particles

    Input:
        particles: np.ndarray(Particle)
    """
    # figure out the square within which we are working
    positions = np.array([p.position for p in particles])
    min_x = np.amin(positions[:, 0])
    min_y = np.amin(positions[:, 1])
    max_x = np.amax(positions[:, 0])
    max_y = np.amax(positions[:, 1])
    square_side = max(max_x - min_x, max_y - min_y)
    lower_left = np.array([min_x, min_y])

    # load data into cell, subdivide it using recursive method
    init_cell = Cell([], particles)
    divide_cell(init_cell, particles, lower_left, square_side)
    return init_cell


def divide_cell (cell, particles, ll, side):
    """
    Subdivide this given cell until each leaf node contains at most one particle
    """
    if len(particles) <= 1:
        # subdivision complete
        return

    # get coords of subdivision squares
    x, y = ll[0], ll[1]
    half_side = side / 2
    ll_1 = (x + half_side, y + half_side)
    ll_2 = (x, y + half_side)
    ll_3 = (x, y)
    ll_4 = (x + half_side, y)
    lls = [ll_1, ll_2, ll_3, ll_4]
    
    pos = np.array([p.position for p in particles])

    # recurse
    for i in range(4):
        in_rect = np.all(np.logical_and(lls[i] <= pos,
                                        pos <= lls[i] + half_side), axis=1)
        cell_i_particles = particles[in_rect]
        cell_i = Cell([], cell_i_particles)
        cell.children.append(cell_i)
        divide_cell(cell_i, cell_i_particles, lls[i], half_side)