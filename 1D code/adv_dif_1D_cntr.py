#!/usr/bin/env python
"""Calculate the values of nitrate concentration (c) in time and space following the advection-diffusion equations.

Example usage from the notebook::

import adv_dif_1D_arr
# Run 5 time steps on a 9 point grid
adv_dif_1D_arr.adv_dif_1D(5,9)

Example usage from the shell::

  # Run 5 time steps on a 9 point grid
  $ adv_dif_1D_arr.py 5 9

The graph window will close as soon as the animation finishes.  And
the default run for 5 time steps doesn't produce much of interest; try
at least 100 steps.

Example usage from the Python interpreter::

  $ python
  ...
  >>> import adv_dif_1D_arr
  >>> # Run 200 time steps on a 9 point grid
  >>> adv_dif_1D_arr.adv_dif_1D((200, 9))
"""
from __future__ import division
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import os,glob

def init(x,n_grid):
    """Set the initial condition values."""
    b = x[len(x) // 2]
    a = 1
    s = 1
    c0 = a*np.exp(-((x-b)**2)/2*(s**2))
    return c0

def boundary_conditions(c_array, n_grid):
    """Set the boundary condition values."""
    #c_array[n_grid - 1] = 0
    #c_array[0] = 1
    c_array[n_grid - 1] = c_array[n_grid - 2]
    c_array[0] = c_array[1]
    return c_array


#def fw_euler(c_old, c_now, u, D, dt, dx, n_grid):
#    """Calculate the next time step values using the forward Euler scheme"""
#    c_new = np.zeros(n_grid)
#    for pt in np.arange(1, n_grid - 1):
#        c_new[pt] = c_old[pt] - (u*dt/(dx)) * (c_now[pt + 1] - c_now[pt - 1]) + (D*dt/(2*dx**2)) * (c_now[pt + 1] - 2*c_now[pt] + c_now[pt - 1])
#    return c_new

def fw_euler(c_old, u, D, dt, dx, n_grid):
    """Calculate the next time step values using the forward Euler scheme"""
    c_new = np.zeros(n_grid)
    for pt in np.arange(1, n_grid - 1):
        c_new[pt] = c_old[pt] - (u*dt/(2*dx)) * (c_old[pt + 1] - c_old[pt - 1]) + (D*dt/(dx**2)) * (c_old[pt + 1] - 2*c_old[pt] + c_old[pt - 1])
    return c_new

def adv_dif_1D(args):
    """Run the model.

    args is a 4-tuple; (number-of-time-steps, number-of-grid-points, L, T)
    """
    n_grid = int(args[0])
    n_time = int(args[1])
    L = int(args[2])
    T = int(args[3])
    
#     Alternate implementation:
#     n_time, n_grid = map(int, args)

    # Constants and parameters of the model
    u = 0.1                       # flow field [m/s]
    dx = L / (n_grid-1)        # grid spacing [m]
    dt = T / (n_time-1)        # time step [s]
    c0 = 0.1                   # initial concentration [kg/L]
    D = 0                    # Diffusivity [m^2/s]
    x = np.linspace(0,L,n_grid) # length coordinate [m]
    t = np.linspace(0,T,n_time) # time coordinate [s]
    # Impose initial conditions
    c_vals = np.zeros((n_grid,n_time))
    c_old = init(x,n_grid)
    # Impose boundary conditions
    c_old = boundary_conditions(c_old,n_grid)
    # Putting intial conditions into results array
    c_vals[:,0] = c_old
    #c_now = c_old
    # Time step loop using forward euler scheme
    for ts in np.arange(1, n_time):
        # Advance the solution and apply the boundary conditions
        #c_new=fw_euler(c_old, c_now, u, D, dt, dx, n_grid)
        c_new=fw_euler(c_old, u, D, dt, dx, n_grid)
        c_new=boundary_conditions(c_new,n_grid)
        # Store the values in the results array
        #c_old = c_now
        c_vals[:,ts] = c_new
        c_old = c_new
    return c_vals
