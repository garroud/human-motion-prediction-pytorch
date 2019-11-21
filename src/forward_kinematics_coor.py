from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import copy
from data_utils import _some_variables, revert_coordinate_space, fkl
import argparse

parser = argparse.ArgumentParser(description="Human Motion Model")
parser.add_argument('--sample_name', default='samples.h5', type=str, metavar='S', help='input sample file.')
parser.add_argument('--action_name', default='walking_0', type=str, metavar='S', help='input action.')
parser.add_argument('--save_name', default='walking_0.gif', type=str, metavar='S', help='input file name')
parser.add_argument('--save', action='store_true', help="Whether to save the gif")
args = parser.parse_args()

def main():

  # Load all the data
  parent, offset, rotInd, expmapInd = _some_variables()

  # numpy implementation
  with h5py.File( args.sample_name, 'r' ) as h5f:
    expmap_gt = h5f['expmap/gt/'+args.action_name][:]
    expmap_pred = h5f['expmap/preds/'+args.action_name][:]


  nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

  # Put them together and revert the coordinate space
  # expmap_all = revert_coordinate_space( np.vstack((expmap_gt, expmap_pred)), np.eye(3), np.zeros(3) )
  # expmap_gt   = expmap_all[:nframes_gt,:]
  # expmap_pred = expmap_all[nframes_gt:,:]

  # Compute 3d points for each frame
  xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
  for i in range( nframes_gt ):
    # xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )
    xyz_gt[i,:] = expmap_gt[i,3:]
  for i in range( nframes_pred ):
    # xyz_pred[i,:] = fkl( expmap_pred[i,:], parent, offset, rotInd, expmapInd )
    xyz_pred[i,:] = expmap_pred[i,3:]

  # === Plot and animate ===
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob = viz.Ax3DPose(ax)

  # Plot the conditioning ground truth
  # for i in range(nframes_gt):
  #   ob.update( xyz_gt[i,:] )
  #   # plt.show(block=False)
  #   # fig.canvas.draw()
  #   plt.pause(0.01)
  #
  # # Plot the prediction
  # for i in range(nframes_pred):
  #   ob.update( xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
  #   # plt.show(block=False)
  #   # fig.canvas.draw()
  #   plt.pause(0.01)
  to_draw = np.append(xyz_gt, xyz_pred,axis=0)

  # dirty workround for generation of gif
  counter = 0
  def update(x):
      nonlocal counter
      if counter < 25:
          counter += 1
          return ob.update(x)
      else:
          if counter == 50:
              counter = 0
          else:
              counter += 1
          return ob.update(x,lcolor="#9b59b6", rcolor="#2ecc71")

  anim = animation.FuncAnimation(fig, update, frames=to_draw, interval=40)
  if args.save:
      anim.save(args.save_name,writer='imagemagick', fps=25)
  else:
      plt.show()


if __name__ == '__main__':
    main()
