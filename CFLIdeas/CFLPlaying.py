#!/usr/bin/env python
from __future__ import division
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar


class Quantity(object):
    """Generic quantity class for storing model variables"""

    def __init__(self, n_grid, n_time):

        self.n_grid = n_grid

        # Time stepping arrays
        self.prev = np.empty(n_grid)
        self.now = np.empty(n_grid)
        self.next = np.empty(n_grid)

        # Storage for full time history
        self.store = np.empty((n_grid, n_time))


    def store_timestep(self, time_step, attr='next'):

        self.store[:, time_step] = self.__getattribute__(attr)


    def shift(self):

        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)


def initial_conditions(c, c0, cr):

    c.prev[:] = 0
    c.prev[len(c.prev) // 2] = c0


def boundary_conditions(c_array, cr, n_grid):

    c_array[n_grid - 1] = c_array[n_grid - 2]
    c_array[0] = c_array[1]


def leap_frog(c, u, D, dt, dx, n_grid):

    for pt in np.arange(1, n_grid - 1):

        c.next[pt] = (
            c.prev[pt]
            - (u * dt / dx) * (c.now[pt + 1] - c.now[pt - 1])
            + (D * 2 * dt / (dx ** 2)) *
            (c.now[pt + 1] - 2 * c.now[pt] + c.now[pt - 1])
        )


def make_graph(c, dt, n_time):

    fig, ax_c = plt.subplots(1, 1, figsize=(10, 10))

    fig.text(0.25, 0.95,
             'Results from t = %.3fs to %.3fs' % (0, dt * n_time))

    ax_c.set_ylabel('c [kg/L]')
    ax_c.set_xlabel('Grid Point')

    cmap = plt.get_cmap('viridis')

    cNorm = colors.Normalize(vmin=0, vmax=1. * n_time)
    cNorm_seconds = colors.Normalize(vmin=0, vmax=1. * n_time * dt)

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    interval = int(np.ceil(n_time / 20))

    for time in range(0, n_time, interval):

        colorVal = scalarMap.to_rgba(time)
        ax_c.plot(c.store[:, time], color=colorVal)

    ax_c = fig.add_axes([0.95, 0.05, 0.05, 0.9])
    cb1 = colorbar.ColorbarBase(ax_c, cmap=cmap, norm=cNorm_seconds)

    cb1.set_label('Time (s)')


def check_cfl(u, dt, dx, D):
    """Check CFL stability conditions"""

    courant = abs(u) * dt / dx
    diffusion = D * dt / dx**2

    print("\n--- CFL Stability Check ---")
    print("Courant number (u dt / dx):", courant)
    print("Diffusion number (D dt / dx^2):", diffusion)

    if courant <= 1:
        print("Advection condition satisfied")
    else:
        print("Advection condition VIOLATED")

    if diffusion <= 0.5:
        print("Diffusion condition satisfied")
    else:
        print("Diffusion condition VIOLATED")



def adv_dif_1D(args):

    n_time = int(args[0])
    n_grid = int(args[1])

    # Model parameters
    cr = 0.1
    u = 0
    dt = 0.01
    dx = 1
    c0 = 0.1
    D = 0.1

    # Check CFL conditions
    check_cfl(u, dt, dx, D)

    c = Quantity(n_grid, n_time)

    initial_conditions(c, c0, cr)

    c.store_timestep(0, 'prev')

    boundary_conditions(c.now, cr, n_grid)

    c.store_timestep(1, 'now')

    for t in np.arange(2, n_time):

        leap_frog(c, u, D, dt, dx, n_grid)

        boundary_conditions(c.next, cr, n_grid)

        c.store_timestep(t)

        c.shift()

    make_graph(c, dt, n_time)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        adv_dif_1D((50, 9))
        plt.show()

    elif len(sys.argv) == 3:

        adv_dif_1D(sys.argv[1:])
        plt.show()

    else:

        print('Usage: rain n_time n_grid')
        print('n_time = number of time steps')
        print('n_grid = number of grid points')