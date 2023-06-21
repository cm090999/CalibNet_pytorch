from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from utils import so3
# from losses.chamfer_loss import chamfer_distance
from scipy.spatial.transform import Rotation
import numpy as np

# import neuralnet_pytorch.metrics as nnp
from neuralnet_pytorch.metrics import chamfer_loss
# from chamfer_distance import ChamferDistance as chamfer_dist
# import emd_cuda
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if not torch.cuda.is_available():
    device = 'cpu'
    print('CUDA is not available, use CPU to run')
else:
    device = 'cuda:0'

class Photo_Loss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(Photo_Loss, self).__init__()
        assert reduction in ['sum','mean','none'], 'Unknown or invalid reduction'
        self.scale = scale
        self.reduction = reduction
    def forward(self,input:torch.Tensor,target:torch.Tensor):
        """Photo loss

        Args:
            input (torch.Tensor): (B,H,W)
            target (torch.Tensor): (B,H,W)

        Returns:
            torch.Tensor: scaled mse loss between input and target
        """
        return F.mse_loss(input/self.scale,target/self.scale,reduction=self.reduction)
    def __call__(self,input:torch.Tensor,target:torch.Tensor)->torch.Tensor:
        return self.forward(input,target)
    
class ChamferDistanceLoss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(ChamferDistanceLoss, self).__init__()
        assert reduction in ['sum','mean','none'], 'Unknown or invalid reduction'
        self.reduction = reduction
        self.scale = scale
    def forward(self, template, source):
        p0 = template/self.scale
        p1 = source/self.scale
        loss = chamfer_loss(p0,p1,reduce=self.reduction)
        return loss

        return chamfer_loss(p0, p1, reduce=self.reduction)
        if self.reduction == 'none':
            return chamfer_distance(p0, p1)
        elif self.reduction == 'mean':
            return torch.mean(chamfer_distance(p0, p1),dim=0)
        elif self.reduction == 'sum':
            return torch.sum(chamfer_distance(p0, p1),dim=0)
    def __call__(self,template:torch.Tensor,source:torch.Tensor)->torch.Tensor:
        return self.forward(template,source)
    
import sys
sys.path.append('PyTorchEMD/')

from emd import EarthMoverDistance

def earth_mover_distance_(p0, p1, reduce='mean'):
    # Calculate pairwise distances between points in p0 and p1
    dist_matrix = torch.cdist(p0, p1, p=2)

    # Solve the optimal transport problem using linear programming
    emd_loss = torch.mean(torch.min(dist_matrix, dim=1)[0])

    if reduce == 'sum':
        emd_loss = torch.sum(emd_loss)
    elif reduce == 'none':
        return emd_loss

    return emd_loss if reduce == 'mean' else emd_loss.mean()


class EarthMoverDistanceLoss(nn.Module):
    def __init__(self, scale=1.0, reduction='mean'):
        super(EarthMoverDistanceLoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none'], 'Unknown or invalid reduction'
        self.reduction = reduction
        self.scale = scale
        self.emd = EarthMoverDistance()

    def forward(self, template, source):
        p0 = template / self.scale
        p1 = source / self.scale
        loss = self.emd(p0, p1, transpose=True)
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()

    def __call__(self, template: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.forward(template, source)
    
    
class Geodesic_Regression_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        return
    def forward(self, x:torch.Tensor)->tuple:
        dR,dT = geodesic_distance(x)
        return dR, dT
    def __call__(self, x) -> Any:
        return self.forward(x)



def geodesic_distance(x:torch.Tensor,)->tuple:
    """geodesic distance for evaluation

    Args:
        x (torch.Tensor): (B,4,4)

    Returns:
        torch.Tensor(1),torch.Tensor(1): distance of component R and T 
    """
    R = x[:,:3,:3]  # (B,3,3) rotation
    T = x[:,:3,3]  # (B,3) translation
    dR = so3.log(R) # (B,3)
    dR = F.mse_loss(dR,torch.zeros_like(dR).to(dR),reduction='none').mean(dim=1)  # (B,3) -> (B,1)
    dR = torch.sqrt(dR).mean(dim=0)  # (B,1) -> (1,)  Rotation RMSE (mean in batch)
    dT = F.mse_loss(T,torch.zeros_like(T).to(T),reduction='none').mean(dim=1) # (B,3) -> (B,1)
    dT = torch.sqrt(dT).mean(dim=0)  # (B,1) -> (1,) Translation RMSE (mean in batch)
    return dR, dT

def gt2euler(gt:np.ndarray):
    """gt transformer to euler anlges and translation

    Args:
        gt (np.ndarray): 4x4

    Returns:
        angle_gt, trans_gt: (3,1),(3,1)
    """
    R_gt = gt[:3, :3]
    euler_angle = Rotation.from_matrix(R_gt)
    anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
    angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
    trans_gt_t = -R_gt @ gt[:3, 3]
    return angle_gt, trans_gt_t