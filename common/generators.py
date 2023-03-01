from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)


    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        #out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, 'SomeAction1'

    def __len__(self):
        return self._poses_3d.shape[0]
        
        
class PoseGenerator_Multi(Dataset):
    def __init__(self, poses_3d, poses_2d, actions):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)



    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[4*index]
        
        out_pose_2d_v1 = self._poses_2d[4*index]
        out_pose_2d_v2 = self._poses_2d[4*index+1]
        out_pose_2d_v3 = self._poses_2d[4*index+2]
        out_pose_2d_v4 = self._poses_2d[4*index+3]
        #out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        
        out_pose_2d_v1 = torch.from_numpy(out_pose_2d_v1).float()
        out_pose_2d_v2 = torch.from_numpy(out_pose_2d_v2).float()
        out_pose_2d_v3 = torch.from_numpy(out_pose_2d_v3).float()
        out_pose_2d_v4 = torch.from_numpy(out_pose_2d_v4).float()

        return out_pose_3d, out_pose_2d_v1,out_pose_2d_v2,out_pose_2d_v3,out_pose_2d_v4, 'SomeAction1'

    def __len__(self):
        return int(self._poses_3d.shape[0]/4)



class PoseGenerator_Multi_Concat(Dataset):
    def __init__(self, poses_3d_world,poses_3d_cam, poses_2d, actions):
        assert poses_3d_world is not None

        self._poses_3d_world = np.concatenate(poses_3d_world)
        self._poses_3d_cam = np.concatenate(poses_3d_cam)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)



    def __getitem__(self, index):
        out_pose_3d_world = torch.from_numpy(self._poses_3d_world[4*index]).float()
        
        out_pose_3d_cam_1 = torch.from_numpy(self._poses_3d_cam[4*index]).float()
        out_pose_3d_cam_2 = torch.from_numpy(self._poses_3d_cam[4*index+1]).float()
        out_pose_3d_cam_3 = torch.from_numpy(self._poses_3d_cam[4*index+2]).float()
        out_pose_3d_cam_4 = torch.from_numpy(self._poses_3d_cam[4*index+3]).float()
        
        out_pose_2d_v1 = torch.from_numpy(self._poses_2d[4*index]).float()
        out_pose_2d_v2 = torch.from_numpy(self._poses_2d[4*index+1]).float()
        out_pose_2d_v3 = torch.from_numpy(self._poses_2d[4*index+2]).float()
        out_pose_2d_v4 = torch.from_numpy(self._poses_2d[4*index+3]).float()


        return out_pose_3d_world,out_pose_3d_cam_1,out_pose_3d_cam_2,out_pose_3d_cam_3,out_pose_3d_cam_4, out_pose_2d_v1,out_pose_2d_v2,out_pose_2d_v3,out_pose_2d_v4, 'SomeAction1'

    def __len__(self):
        return int(self._poses_3d_world.shape[0]/4)