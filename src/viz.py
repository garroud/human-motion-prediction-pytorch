"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    # self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    # self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # # Left / right indicator
    # self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.I = np.array([0,1,2,3,4,0,6,7,8,9,0,11,12,13,14,0,16,17,18,19,20,19,22,0,24,25,26,27,30,27,28])
    self.J = \
    np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,30,31,28,29])
    self.ax = ax
    self.LR = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    vals = np.zeros((32, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')

# new Class to plot edge.
class Ax3DPoseEdge(object):
  def __init__(self, ax, num_edge, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    # self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    # self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # # Left / right indicator
    # self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.I = np.array([0,1,2,3,4,0,6,7,8,9,0,11,12,13,14,0,16,17,18,19,20,19,22,0,24,25,26,27,30,27,28])
    self.J = \
    np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,30,31,28,29])
    self.ax = ax
    self.LR = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    vals = np.zeros((32, 3))

    self.edges = np.zeros([2,num_edge],dtype=np.int32)
    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    for i in np.arange(self.edges.shape[1]):
      x = np.array( [vals[self.edges[0,i], 0], vals[self.edges[1,i], 0]] )
      y = np.array( [vals[self.edges[0,i], 1], vals[self.edges[1,i], 1]] )
      z = np.array( [vals[self.edges[0,i], 2], vals[self.edges[1,i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c="#5677ab"))
    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, edges=None, lcolor="#3498db", rcolor="#e74c3c", print_edge=False):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    if print_edge:
        for j in np.arange( edges.shape[1] ):
            x = np.array( [vals[edges[0,j], 0], vals[edges[1,j], 0]] )
            y = np.array( [vals[edges[0,j], 1], vals[edges[1,j], 1]] )
            z = np.array( [vals[edges[0,j], 2], vals[edges[1,j], 2]] )
            self.plots[len(self.I)+j][0].set_xdata(x)
            self.plots[len(self.I)+j][0].set_ydata(y)
            self.plots[len(self.I)+j][0].set_3d_properties(z)
            self.plots[len(self.I)+j][0].set_color("#5677ab")

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
    r = 750;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')
