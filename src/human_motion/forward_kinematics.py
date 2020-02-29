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
from data_utils import *
import argparse
from data_utils import _some_variables, revert_coordinate_space, fkl
import random



parser = argparse.ArgumentParser(description="Human Motion Model")
parser.add_argument('--sample_name', default='samples.h5', type=str, metavar='S', help='input sample file.')
parser.add_argument('--action_name', default='walking_0', type=str, metavar='S', help='input action.')
parser.add_argument('--save_name', default='walking_0.gif', type=str, metavar='S', help='input file name')
parser.add_argument('--save', action='store_true', help="Whether to save the gif")
parser.add_argument('--print_edge', action='store_true', help="whether to show the edges in the gif")
args = parser.parse_args()

def main():

  node_index = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 25, 26, 27, 28]) - 1
  node_mapping = {}
  for newIdx,oriIdx in enumerate(node_index):
      node_mapping[newIdx] = oriIdx
  off_diag = np.ones([21,21]) - np.eye(21)
  edges = np.array(np.where(off_diag),dtype=np.float32)
  edges = np.vectorize(lambda x: node_mapping[x])(edges)
  # Load all the data
  parent, offset, rotInd, expmapInd = _some_variables()
  num_edges = 10
  # numpy implementation
  with h5py.File( args.sample_name, 'r' ) as h5f:
    expmap_gt = h5f['expmap/gt/'+args.action_name][:]
    expmap_pred = h5f['expmap/preds/'+args.action_name][:]
    edge_weight = h5f['expmap/edge_weight/' + args.action_name][:]
  # expmap_pred *= 10
  nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt, expmap_pred)), np.eye(3), np.zeros(3) )
  # expmap_gt   = expmap_all[:nframes_gt,:]
  # expmap_pred = expmap_all[nframes_gt:,:]
  # Compute 3d points for each frame
  xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
  edges_to_print = np.zeros((nframes_pred, 2, num_edges),dtype=np.int32)
  def getEdges(edge_weight):
      edge_weight = list(edge_weight)
      edge_weight = list(enumerate(edge_weight))
      top_index = sorted(edge_weight, key=lambda x: x[1], reverse=True)
      # return [x[0] for x in top_index[:num_edges] ]
      res = [x[0] for x in top_index if x[1] > 0.5]
      if len(res) > num_edges:
          return res[:num_edges]
      else:
          return res
  for i in range( nframes_gt ):
    xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )
  for i in range( nframes_pred ):
    xyz_pred[i,:] = fkl( expmap_pred[i,:], parent, offset, rotInd, expmapInd )
    indexes = getEdges(edge_weight[i])
    print(indexes)
    edges_to_print[i,:,:len(indexes)] = edges[:,indexes]
  # === Plot and animate ===
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob = viz.Ax3DPoseEdge(ax, num_edge=num_edges)

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
  def update(x, edges):
      nonlocal counter
      if counter < nframes_gt:
          res = ob.update(x,print_edge=False)
          counter += 1
      else:
          res= ob.update(x,edges_to_print[counter-nframes_gt],lcolor="#9b59b6", rcolor="#2ecc71", print_edge=args.print_edge)
          if counter != nframes_gt + nframes_pred - 1:
              counter += 1
      return res
  anim = animation.FuncAnimation(fig, update, frames=to_draw, fargs=(edges_to_print,),interval=40)
  if args.save:
      anim.save(args.save_name,writer='imagemagick', fps=25)
  else:
      plt.show()


if __name__ == '__main__':
    main()
