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

def init(x,y,n_x,n_y):
    """Set the initial condition values."""
    c0 = np.zeros((n_x,n_y))
    return c0

def boundary_conditions(c_array, n_x, n_y,dt):
    """Set the boundary condition values."""

    c_array[0,:] = 0
    c_array[n_x - 1,:] = 0
    c_array[:,n_y - 1] = 0
    c_array[:,0] = c_array[:,1]

    N_flux = 0.01
    c_array[n_x//2,0] = c_array[n_x//2,0] + dt * N_flux

    return c_array

def fw_euler(c_old, u, D, dt, dx, dy, n_x, n_y):
    """Calculate the next time step values using the forward Euler scheme"""
    c_new = np.zeros((n_x, n_y))
    for i in np.arange(1, n_x - 1):
        for j in np.arange(1, n_y - 1): 
            c_new[i,j] = c_old[i,j] - (u[j]*dt/(2*dx)) * (c_old[i+1, j] - c_old[i-1, j]) + (D*dt/(dx**2)) * (c_old[i+1, j] - 2*c_old[i, j] + c_old[i-1, j])  + (D*dt/(dy**2)) * (c_old[i, j+1] - 2*c_old[i, j] + c_old[i, j-1])
    return c_new

def adv_dif_2D(args):
    """Run the model.

    args is a 4-tuple; (number-of-time-steps, number-of-grid-points, L, T)
    """
    n_x = int(args[0])
    n_y = int(args[1])
    n_time = int(args[2])
    Lx = int(args[3])
    Ly = int(args[4])
    T = int(args[5])
    
#     Alternate implementation:
#     n_time, n_grid = map(int, args)

    # Constants and parameters of the model    
    dx = Lx / (n_x-1)        # grid spacing [m]
    dy = Ly / (n_y-1)        # grid spacing [m]
    dt = T / (n_time-1)        # time step [s]
    c0 = 0.1                   # initial concentration [kg/L]
    D = 0.01                   # Diffusivity [m^2/s]
    x = np.linspace(0,Lx,n_x) # length coordinate [m]
    y = np.linspace(0,Ly,n_y) # length coordinate [m]
    t = np.linspace(0,T,n_time) # time coordinate [s]

    # U-current calculation
    u0 = 0.01                     # flow field [m/s]
    u = np.zeros((n_y))
    
    ## constant current u
    u[:] = u0
    ## y-dependent current u 
    #for i in range(n_y): 
    #    u[i] = min(u0 * np.log(y[i] + 1), 3 * u0)
    
    # Impose initial conditions
    c_vals = np.zeros((n_x,n_y,n_time))
    c_old = init(x,y,n_x,n_y)
    # Impose boundary conditions
    c_old = boundary_conditions(c_old,n_x,n_y,dt)
    # Putting intial conditions into results array
    c_vals[:,:,0] = c_old
    # Time step loop using forward euler scheme
    for ts in np.arange(1, n_time):
        # Advance the solution and apply the boundary conditions
        c_new=fw_euler(c_old, u, D, dt, dx, dy, n_x, n_y)
        c_new=boundary_conditions(c_new,n_x,n_y,dt)
        # Store the values in the results array
        c_vals[:,:,ts] = c_new
        c_old = c_new
    return c_vals

if __name__ == '__main__':
    # sys.argv is the command-line arguments as a list. It includes
    # the script name as its 0th element. Check for the degenerate
    # cases of no additional arguments, or the 0th element containing
    # `sphinx-build`. The latter is a necessary hack to accommodate
    # the sphinx plot_directive extension that allows this module to
    # be run to include its graph in sphinx-generated docs.
    #
    #  the following command, executed in the plotfile directory makes a movie on ubuntu called
    #   outputmplt.avi
    #  which can be
    #  looped with mplayer -loop 0
    #
    #  mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o outputmplt.avi
    #
    if len(sys.argv) == 1 or 'sphinx-build' in sys.argv[0]:
        # Default to 50 time steps, and 9 grid points
        adv_dif_1D((50, 9, 50, 30))
    else:
        print ('Usage: adv_dif_1D n_time n_grid L T')
        print ('n_time = number of time steps; default = 5')
        print ('n_grid = number of grid points; default = 9')
        print ('L = Length of domain')
        print ('T = Total run time')
