#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:18:24 2017

@author: hyj
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# x = np.arange(30,-20,-5)
def generate_points(xstart, xend, ystart, yend, zstart, zend, len):
    x = np.linspace(xstart, xend, len, endpoint=False,dtype=int)
    y = np.linspace(ystart, yend, len, endpoint=False,dtype=int)
    z = np.linspace(zstart, zend, len, endpoint=False,dtype=int)
    return np.vstack((x,y,z))

def generate_points_with_endpoints(xstart, xend, ystart, yend, zstart, zend, len):
    x = np.linspace(xstart, xend, len, endpoint=True,dtype=int)
    y = np.linspace(ystart, yend, len, endpoint=True,dtype=int)
    z = np.linspace(zstart, zend, len, endpoint=True,dtype=int)
    return np.vstack((x,y,z))

def generate_vertical_lines(xstart, xend, ystart, yend, zstart, zend, len):
    points1=generate_points(xstart, xend, ystart, yend, zstart, zstart, len)
    points2=generate_points(xstart, xend, ystart, yend, zend, zend, len)
    return np.vstack((points1, points2))

def generate_horizontal_lines(xstart, xend, ystart, yend, zstart, zend, len):
    points1=generate_points(xstart, xstart, ystart, yend, zstart, zstart, len)
    points2=generate_points(xend, xend, yend, yend, zend, zend, len)
    return np.vstack((points1, points2))

def geneeate_square(xmin, xmax, ymin, ymax, zmin, zmax, size):
    right_lines=generate_horizontal_lines(xmax, xmax, ymax, ymin, zmin, zmax, size)
    down_lines=generate_horizontal_lines(xmax, xmin, ymin, ymin, zmin, zmax, size)
    left_lines=generate_horizontal_lines(xmin, xmin, ymin, ymax, zmin, zmax, size)
    up_lines=generate_horizontal_lines(xmin, xmax, ymax, ymax, zmin, zmax, size)

    return np.hstack((right_lines, down_lines, left_lines, up_lines))

def generate_fence(xmin, xmax, ymin, ymax, zmin, zmax, size):
    right_lines=generate_vertical_lines(xmax, xmax, ymax, ymin, zmin, zmax, size)
    down_lines=generate_vertical_lines(xmax, xmin, ymin, ymin, zmin, zmax, size)
    left_lines=generate_vertical_lines(xmin, xmin, ymin, ymax, zmin, zmax, size)
    up_lines=generate_vertical_lines(xmin, xmax, ymax, ymax, zmin, zmax, size)

    return np.hstack((right_lines, down_lines, left_lines, up_lines))

xmin=-15
xmax=25
ymin=-20
ymax=30
zmin=0
zmax=10
height = 10
size=10
fence = generate_fence(xmin, xmax, ymin, ymax, zmin, zmax, size)
# print(fence)
# [m_fence,n_fence]=fence.shape

zmax=0
size=1
square1 = geneeate_square(xmin, xmax, ymin, ymax, zmin, zmax, size)
zmin=10
zmax=10
square2 = geneeate_square(xmin, xmax, ymin, ymax, zmin, zmax, size)
print(square2)
# [m_square,n_square]=square1.shape

model_fence = np.hstack((fence, square1, square2))
print(model_fence)
[m,n]=model_fence.shape

with open("fence.txt", 'w') as f:
    for i in range(n):
        f.write("%10f %10f %10f %10f %10f %10f\n" %
                (model_fence[0][i], model_fence[1][i], model_fence[2][i],
                model_fence[3][i], model_fence[4][i], model_fence[5][i]))



np.set_printoptions(suppress = True)
filepath = os.path.abspath('..')+"/bin"

# imu_circle   imu_spline
position = []
quaterntions = []
timestamp = []
tx_index = 5
position = np.loadtxt(filepath + '/imu_pose.txt', usecols = (tx_index, tx_index + 1, tx_index + 2))

# imu_pose   imu_spline
position1 = []
quaterntions1 = []
timestamp1 = []
data = np.loadtxt(filepath + '/imu_int_pose.txt')
# print "type: ", type(data)
# timestamp1 = data[:,0]
# quaterntions1 = data[:,[tx_index + 6, tx_index + 3, tx_index + 4, tx_index + 5]] # qw,qx,qy,qz
position1 = data[:,[tx_index, tx_index + 1, tx_index + 2]]

# imu_pose   imu_spline
position2 = []
quaterntions2 = []
timestamp2 = []
data = np.loadtxt(filepath + '/imu_pose_noise.txt')
# timestamp2 = data[:,0]
# quaterntions2 = data[:,[tx_index + 6, tx_index + 3, tx_index + 4, tx_index + 5]] # qw,qx,qy,qz
position2 = data[:,[tx_index, tx_index + 1, tx_index + 2]]

# cam_pose_opt_o_0   cam_pose_opt_o_0
position3 = []
quaterntions3 = []
timestamp3 = []
data = np.loadtxt(filepath + '/imu_int_pose_noise.txt')
# timestamp3 = data[:,0]
# quaterntions3 = data[:,[tx_index + 6, tx_index + 3, tx_index + 4, tx_index + 5]] # qw,qx,qy,qz
position3 = data[:,[tx_index, tx_index + 1, tx_index + 2]]


### plot 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(position[:,0], position[:,1], position[:,2], label='imu_gt')
ax.plot(position1[:,0], position1[:,1], position1[:,2], label='imu_int')
# ax.plot(position2[:,0], position2[:,1], position2[:,2], label='imu_noise_gt')
# ax.plot(position3[:,0], position3[:,1], position3[:,2], label='imu_noise_int')
ax.plot([position[0,0]], [position[0,1]], [position[0,2]], 'r.', label='start')

for i in range(n):
    id=2*i
    ax.plot([model_fence[0][i], model_fence[3][i]],
            [model_fence[1][i], model_fence[4][i]],
            [model_fence[2][i], model_fence[5][i]], 'y')
    ax.text(model_fence[0][i], model_fence[1][i], model_fence[2][i], '%d'%(id))
    ax.text(model_fence[3][i], model_fence[4][i], model_fence[5][i], '%d'%(id+1))

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
