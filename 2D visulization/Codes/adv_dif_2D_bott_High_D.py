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

def init(x,y,n_x,n_y,n_time):
    """Set the initial condition values with Gaussian distribution."""
    c_vals = np.zeros((n_time+1,n_x+4,n_y))
    x0 = x[n_x//2]
    y0 = y[0]
    a = 1
    s = 1
    
    for i in range(n_x):
        for j in range(n_y):
            c_vals[0,i+2,j] = a*np.exp(-((x[i]-x0)**2+(y[j]-y0)**2)/2*(s**2))

    return c_vals

def boundary_conditions(c_array, n_x, n_y):
    """Set the boundary condition values."""
    c_array[0,:] = 0
    c_array[1,:] = 0
    c_array[2,:] = 0
    c_array[3,:] = 0
    
    c_array[n_x,:] = 0
    c_array[n_x+1,:] = 0
    c_array[n_x+2,:] = 0
    c_array[n_x+3,:] = 0
    
    c_array[:,n_y-1] = 0

    c_array[:,0] = c_array[:,1]

    return c_array

def fw_euler(c_now, D, dt, dx, dy, n_x, n_y): # now only for diffusion
    """Calculate the next time step values using the forward Euler scheme"""
    c_df = np.zeros((n_x+4, n_y))
    for i in np.arange(3, n_x+2):
        for j in np.arange(1, n_y-1): 
            c_df[i,j] = (D*dt/(dx**2)) * (c_now[i+1, j] - 2*c_now[i, j] + c_now[i-1, j]) + (D*dt/(dy**2)) * (c_now[i, j+1] - 2*c_now[i, j] + c_now[i, j-1])
    return c_df

def gettable(order, Numpoints):

    '''read in the corresponding coefficient table for the calculation of coefficients for advection3
    '''

    # create a matrix to store the table to be read in
    temp = np.zeros(5)
    ltable = np.zeros((order + 1, 5))

    fname = 'l{0}_table.txt'.format(order)
    fp = open(fname, 'r')

    for i in range(order+1):
        line = fp.readline()
        temp = line.split()
        ltable[i, :]= temp

    fp.close()
    return ltable

def bott(ts, c_now, order, n_x, n_y, u, dt, dx, dy, epsilon): # For advection (IMPORTANT: Function taken and modified from Lab 10)
    '''Step algorithm for Bott Scheme for 2D domains. The scheme is applied in the x-direction for each y-level, as only the u current is considered.'''
    c_adv = np.zeros((n_x+4, n_y))
    ltable = gettable(order, n_x)

    #calculate advection for each y-level
    for j in range(n_y-1):
        # create a matrix to store the current coefficients a(j, k)
        amatrix = np.zeros((order+1, n_x))
        cmatrix = np.zeros((1, n_x+4))
        cmatrix[0,:] = c_now[:,j]
    
        for base in range(0,5):
            amatrix[0:order+1, 0:n_x] += np.dot(
                ltable[0:order+2, base:base+1],
                cmatrix[:,0+base:n_x+base])
    
        # calculate I of c at j+1/2 , as well as I at j
        # as these values will be needed to calculate i at j+1/2 , as
        # well as i at j
    
        # calculate I of c at j+1/2(Iplus),
        # and at j(Iatj)
        Iplus = np.zeros(n_x)
        Iatj = np.zeros(n_x)
    
        tempvalue= 1 - 2*u[j]*dt/dx
        for k in range(order+1):
            Iplus += amatrix[k] * (1- (tempvalue**(k+1)))/(k+1)/(2**(k+1))
            Iatj += amatrix[k] * ((-1)**k+1)/(k+1)/(2**(k+1))
        Iplus[Iplus < 0] = 0
        Iatj = np.maximum(Iatj, Iplus + epsilon)
    
        # finally, calculate the advective contribution to the concentration
        c_adv[3:n_x+2, j] = (
            c_now[3:n_x+2, j] *
            (-Iplus[1:n_x]/ Iatj[1:n_x]) +
            c_now[2:n_x+1, j]*
            Iplus[0:n_x-1]/ Iatj[0:n_x-1])
        # NOTE: This calculation differs from the equation in Lab 10. The current concentration has been subtracted to get only the advective contribution, and not the concentration at the next time-step

    return c_adv
    
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
    U = int(args[6])
    
#     Alternate implementation:
#     n_time, n_grid = map(int, args)

    # Constants and parameters of the model    
    dx = Lx / (n_x-1)        # grid spacing [m]
    dy = Ly / (n_y-1)        # grid spacing [m]
    dt = T / (n_time-1)        # time step [s]
    c0 = 0.1                   # initial concentration [kg/L]
    # D = 0.01                   # Diffusivity [m^2/s]
    D = 0.02
    x = np.linspace(0,Lx,n_x) # length coordinate [m]
    y = np.linspace(0,Ly,n_y) # length coordinate [m]
    t = np.linspace(0,T,n_time) # time coordinate [s]
    epsilon = 0.0001
    order = 4
    
    # U-current calculation 
    u0 = 0.01 # flow field [m/s]
    u = np.zeros((n_y))
    if U==0: # constant current u
        u[:] = u0
    elif U==1: # y-dependent current u 
        for i in range(n_y): 
            u[i] = min(u0 * np.log(y[i] + 1), 3 * u0)
    else:
        print("Error: U should be 0 or 1")

    # Initialize the concentration array
    c_vals = init(x,y,n_x,n_y,n_time)

    # Impose boundary conditions
    c_now = c_vals[0,:,:]
    c_now = boundary_conditions(c_now,n_x,n_y)
    c_vals[0,:,:] = c_now
    
    # Time step loop 
    for ts in np.arange(0, n_time):
        adv = bott(ts, c_now, order, n_x, n_y, u, dt, dx, dy, epsilon) # Calculate the advection contribution to the current concentration at this time-step
        df = fw_euler(c_now, D, dt, dx, dy, n_x, n_y) # Calculate the diffusion contribution to the current concentration at this time-step
        c_next = c_now + adv + df # Add the advective and diffusive components to the current concentration to get the concentration at the next time-step
        # set the boundary points
        c_next = boundary_conditions(c_next, n_x, n_y)
        c_vals[ts+1,:,:] = c_next # Store the new concentration
        c_now = c_next # Advance in time by one time-step
    # output value = c_vals(time, x+4, y)
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
