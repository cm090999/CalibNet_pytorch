import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from utils import so3
# from losses.chamfer_loss import chamfer_distance
from scipy.spatial.transform import Rotation
import numpy as np

from neuralnet_pytorch.metrics import chamfer_loss
from chamfer_distance import ChamferDistance as chamfer_dist
import emd_cuda
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
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(p0,p1)
        loss = (torch.mean(dist1))# + (torch.mean(dist2))
        return loss
        # return chamfer_loss(p0, p1, reduce=self.reduction)
        if self.reduction == 'none':
            return chamfer_distance(p0, p1)
        elif self.reduction == 'mean':
            return torch.mean(chamfer_distance(p0, p1),dim=0)
        elif self.reduction == 'sum':
            return torch.sum(chamfer_distance(p0, p1),dim=0)
    def __call__(self,template:torch.Tensor,source:torch.Tensor)->torch.Tensor:
        return self.forward(template,source)


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



class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)
    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
    Returns:
        cost (torch.Tensor): (b)
    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)

    loss = torch.sum(cost).to(device)
    return loss