#!/usr/bin/env python3
# Original implementation by Federico Magistri
import glob
import os

import numpy as np
import open3d as o3d

from cache import get_cache, memoize


class MickieDataset:
    def __init__(self, data_source, get_color: bool = False, apply_pose: bool = False):
        # Cache
        self.use_cache = True
        self.cache = get_cache(directory="cache/mickie/")
        self.get_color = get_color
        self.apply_pose = apply_pose

        self.data_source = os.path.join(data_source, "")
        self.gt_list = self.read_gt_list(os.path.join(self.data_source, "slam_poses.csv"))
        self.cloud_files = sorted(glob.glob(self.data_source + "*.pcd")) #changed to .pcd rather than .ply
        

    def isRotationMatrix(self, M):
        tag = False
        I = np.identity(M.shape[0])
        if np.all(np.round((np.matmul(M, M.T)),2) == I) and (np.round(np.linalg.det(M),3)): 
            tag = True
        return tag

    def transformation_matrix(self, Q, T, n):
        '''
        Convert a quaternion and translation into a full three-dimensional transformation matrix.
        source https://opensource.docs.anymal.com/doxygen/kindr/master/cheatsheet_latest.pdf
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
        :param T: a 3 element array representing translatrion (x,y,z)
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        
        '''
        # Extract the values from Q
        q0 = Q[3] #w
        q1 = Q[0] #i
        q2 = Q[1] #j
        q3 = Q[2] #k
        t0 = T[0] 
        t1 = T[1]
        t2 = T[2]
            
        # First row of the rotation matrix
        r00 = q0 ** 2 + q1**2 - q2**2 -q3**2 #2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3) 
        r11 = q0**2 - q1**2 + q2**2 - q3**2 #2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = q0**2 - q1**2 - q2**2 + q3**2 #2 * (q0 * q0 + q3 * q3) - 1
        

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        
        #check orthogonality
        if (self.isRotationMatrix(rot_matrix) == False):
            print("Error, rotation matrix ", n," not orthonomal")
            
            

        transf_matrix = np.array([[r00, r01, r02, t0],
                            [r10, r11, r12, t1],      
                            [r20, r21, r22, t2],
                            [0, 0, 0, 1]])
        #print(transf_matrix)
        transf_matrix_row = np.array([r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2, 0, 0, 0, 1])
        
        return rot_matrix , transf_matrix, transf_matrix_row

    #@staticmethod
    def read_gt_list(self, filename):
        Q = np.loadtxt(open(filename, "rb"), delimiter=",", dtype=np.float32, usecols=range(6,10)) #, skiprows=1)
        T = np.loadtxt(open(filename, "rb"), delimiter=",", dtype=np.float32, usecols=range(3,6)) #, skiprows=1)
        
        _,Transf, Transf_row = self.transformation_matrix(Q[0][:], T[0][:], 0)
        for i in range(1, len(Q), 2):
            _, transf_matrix, transf_matrix_row =  self.transformation_matrix(Q[i][:], T[i][:], i)
            #transf_matrix_row = np.array([1,0,0,0, 0,1,0,0,0,0,1,0,0,0,0,1]) # 
            Transf_row = np.vstack((Transf_row, transf_matrix_row))
            

            #Transf = np.dstack((Transf, transf_matrix))
        #print(Transf_row.reshape((len(Transf_row),4,4)).shape)
        return Transf_row.reshape((len(Transf_row),4,4))
        

    @memoize()
    def __getitem__(self, idx):
        pose = self.gt_list[idx]
        pcd = o3d.io.read_point_cloud(self.cloud_files[idx])
        pcd.transform(pose) if self.apply_pose else None
        xyz = np.array(pcd.points)
        colors = np.array(pcd.colors)

        if self.get_color:
            return xyz, pose
        return xyz, pose

    def __len__(self):
        return len(self.cloud_files)
