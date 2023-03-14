"""
An 2D implementation of the Barnes-Hut algorithm
"""
import numpy as np

G = 0.01 # gravity constant 

class Cell:
    def __init__(self, children, particles, width):
        """
        Input:
            children: list(Cell)
            particles: list(Particle)
        """
        self.width = width
        masses = np.array([p.mass for p in particles])
        self.mass = np.sum(masses)
        weighted_pos = np.array([p.position * p.mass for p in particles])
        self.com = np.average(weighted_pos)
        self.children = children


class Particle:
    def __init__(self, position, mass, velocity):
        # need to convert to float to make sure no addition issues occur during
        # simulation
        self.position = np.array([float(position[0]), float(position[1])])
        self.mass = mass
        self.velocity = np.array([float(velocity[0]), float(velocity[1])])


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
    init_cell = Cell([], particles, square_side)
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
        cell_i = Cell([], cell_i_particles, half_side)
        cell.children.append(cell_i)
        divide_cell(cell_i, cell_i_particles, lls[i], half_side)


def record_simulation(particles, time, filename, theta = 1):
    """
    Process the simulation, store the result in a [filename].sim file in the 
    data folder
    """
    # TODO: implement multithreading

    # create file 
    file = open("data/"+filename+".sim", "w")

    for i in range(time):
        calculate_tick(particles, theta, file)

    file.close()


def calculate_tick(particles, theta, file):
    """
    Process this tick of the simulation
    """
    # construct tree
    tree = generate_tree(particles)

    # move each particle, then compute the force on that particle
    for p in particles:
        # NOTE: it's ok to change particles' positions iteratively like this
        # since we are basing calculations only off of the tree, not the 
        # 'current' positions of the particles. this would be bad if we were 
        # working with the naive algorithm
        p.position += p.velocity
        force_on_p = compute_force(tree, p, theta)
        acceleration = force_on_p/p.mass
        p.velocity += acceleration
        
        # record the position of this particle in the file
        data_str = str(p.position[0]) + " " + str(p.position[1]) + "\n"
        file.write(data_str)
    
    # signal that this tick is done
    file.write("#\n")


def compute_force(tree, particle, theta):
    """
    Compute force on this particle from the tree that starts at this node
    """
    l = tree.width
    D = np.linalg.norm(particle.position - tree.com)
    
    if l/D < theta:
        # use the root cell for force computations
        force = -1 * (G * particle.mass * tree.mass)/(D * D)
        return force * (particle.position - tree.com)
    else:
        # recurse down
        return sum([compute_force(c, particle, theta) for c in tree.children])