from . import se3,so3
import torch
import numpy as np
from math import pi as PI
from collections.abc import Iterable

import torchgeometry as tgm
from pyquaternion import Quaternion

class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, max_deg, max_tran, mag_randomly=True, concat=False):
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        if self.randomly:
            deg = torch.rand(1).item()*self.max_deg
            tran = torch.rand(1).item()*self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran
        amp = deg * PI / 180.0  # deg to rad
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        t = torch.rand(1, 3) * tran

        # the output: twist vectors.
        R = so3.exp(w) # (N, 3) --> (N, 3, 3)
        G = torch.zeros(1, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        x = se3.log(G) # --> (N, 6)
        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [3,N] or [6,N]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        if self.concat:
            return torch.cat([se3.transform(g, p0[:3,:]),so3.transform(g[:,:3,:3], p0[3:,:])], dim=1)  # [1, 4, 4] x [6, N] -> [6, N]
        else:
            return se3.transform(g, p0)   # [1, 4, 4] x [3, N] -> [3, N]

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class UniformTransformSE3:
    def __init__(self, max_deg, max_tran, mag_randomly=True, concat=False, axes=torch.tensor([1,1,1,1,1,1]) ):
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.gt = None
        self.igt = None
        self.axes = axes

    def generate_transform(self):
        # return: a twist-vector
        if self.randomly:
            deg = torch.rand(1).item()*self.max_deg
            tran = torch.rand(1).item()*self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran
        amp = deg * PI / 180.0  # deg to rad
        w = (2*torch.rand(1, 3)-1) # / torch.norm(2*torch.rand(1, 3)-1) * amp
        t = (2*torch.rand(1, 3)-1) # / torch.norm(2*torch.rand(1, 3)-1) * tran
        w = w * amp   / torch.norm(w)
        t = t * tran  / torch.norm(t)
        # w = w * self.axes[3:]
        # t = t * self.axes[:3]

        # the output: twist vectors.
        R = so3.exp(w) # (N, 3) --> (N, 3, 3)
        G = torch.zeros(1, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        x = se3.log(G) # --> (N, 6)
        return x*self.axes # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [3,N] or [6,N]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        if self.concat:
            return torch.cat([se3.transform(g, p0[:3,:]),so3.transform(g[:,:3,:3], p0[3:,:])], dim=1)  # [1, 4, 4] x [6, N] -> [6, N]
        else:
            return se3.transform(g, p0)   # [1, 4, 4] x [3, N] -> [3, N]

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)
    

class UniformTransformSE3_quaternion:
    def __init__(self, max_deg, max_tran, mag_randomly=True, concat=False):
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.gt = None
        self.igt = None
        self.generator = torch.Generator()

    def generate_transform(self):
        if self.randomly:
            deg = torch.rand((1), generator=self.generator).item() * self.max_deg
            tran = torch.rand((1), generator=self.generator).item() * self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran

        amp = deg * torch.pi / 180.0  # deg to rad
        

        # Generate random rotation quaternion
        # q = tgm.random_quaternion(1)  # [1, 4]
        # q = torch.from_numpy(Quaternion.random().q)
        q = self.generate_random_rotation(amp)

        # Generate random translation vector
        t = torch.rand((1, 3), generator=self.generator) * 2 - 1  # [-1, 1] range
        t = t / torch.norm(t)

        # Scale rotation and translation
        q = q * amp
        t = t * tran

        return q, t

    def apply_transform(self, p0, q, t):
        # p0: [3, N] or [6, N]
        # q: [1, 4]
        # t: [1, 3]

        # Convert quaternion to rotation matrix
        # R = tgm.quaternion_to_rotation_matrix(q)  # [1, 3, 3]
        R = tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(q.unsqueeze(0)))[:,:3,:3] 

        # Create 4x4 transformation matrix
        G = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        gt = torch.inverse(G)  # [1, 4, 4]
        self.gt = gt.squeeze(0)  # gt: p1 -> p0
        self.igt = G.squeeze(0)  # igt: p0 -> p1

        if self.concat:
            return torch.cat(
                [
                    tgm.transform_points(G, p0[:3, :]),
                    tgm.transform_points(R, p0[3:, :]),
                ],
                dim=0,
            )  # [6, N]
        else:
            p0_changed = p0.permute(0,2,1)
            p0_changed_tf = tgm.transform_points(G, p0_changed)
            p0_tf = p0_changed_tf.permute(0,2,1)
            return p0_tf
        

    def transform(self, tensor):
        q, t = self.generate_transform()
        return self.apply_transform(tensor, q, t)

    def __call__(self, tensor):
        return self.transform(tensor)
    

    def generate_random_rotation(self, amp):
        # Generate random components
        x = 2 * torch.rand((1), generator=self.generator) - 1
        y = 2 * torch.rand((1), generator=self.generator) - 1
        z = 2 * torch.rand((1), generator=self.generator) - 1

        # Normalize the vector
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm

        # # Get random angle
        # ang = torch.rand(1) * amp / 180 * torch.pi

        # Calculate rotation angle
        u1 = torch.rand(1)
        angle = amp * u1.item()

        # Calculate sin and cos of half the rotation angle
        s = np.sin(angle / 2)
        c = np.cos(angle / 2)

        # Construct quaternion
        q = torch.tensor([c, s * x, s * y, s * z])

        return q
    

class DepthImgGenerator:
    def __init__(self,img_shape:Iterable,InTran:torch.Tensor,pcd_range:torch.Tensor,pooling_size=5):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        # InTran (3,4) or (4,4)
        self.img_shape = img_shape
        self.InTran = torch.eye(3)[None,...]
        self.InTran[0,:InTran.size(0),:InTran.size(1)] = InTran  # [1,3,3]
        self.pcd_range = pcd_range  # (B,N)

    def transform(self,ExTran:torch.Tensor,pcd:torch.Tensor)->tuple:
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        H,W = self.img_shape
        B = ExTran.size(0)
        self.InTran = self.InTran.to(pcd.device)
        pcd = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
        batch_depth_img = torch.zeros(B,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        # size of rev_i is not constant so that a batch-formed operdation cannot be applied
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_depth_img[bi*torch.ones_like(proj_xrev),proj_yrev,proj_xrev] = self.pcd_range[bi,rev_i]
        batch_depth_img = batch_depth_img.unsqueeze(1)
        batch_depth_img = self.pooling(batch_depth_img)
        return batch_depth_img, pcd   # (B,1,H,W), (B,3,N)
    
    def __call__(self,ExTran:torch.Tensor,pcd:torch.Tensor):
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        assert len(ExTran.size()) == 3, 'ExTran size must be (B,4,4)'
        assert len(pcd.size()) == 3, 'pcd size must be (B,3,N)'
        return self.transform(ExTran,pcd)
    
def pcd_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray,range:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        range (np.ndarray): (N,)

    Returns:
        u,v,r,rev: u,v,r (with rev) and rev
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = range[rev]
    return u,v,r,rev

def binary_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray):
    """project pcd on img (binary mode)

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3,N)

    Returns:
        u,v,rev: u,v (without rev filter) and rev
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    return u,v,rev

def nptrans(pcd:np.ndarray,G:np.ndarray)->np.ndarray:
    R,t = G[:3,:3], G[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t


class RangeImageGenerator():
    def __init__(self, v_res: float, h_res: float, v_fov: tuple, h_fov: tuple, scaling = 0.95, axisFront = 'camera'):
        self.v_res = v_res
        self.h_res = h_res
        self.v_fov = v_fov
        self.h_fov = h_fov
        self.scaling = scaling
        self.axisFront = axisFront
        return
    
    def get_shape(self):
        # Get Number of pixels in range image
        vertPix = int((np.absolute(self.v_fov[1] - self.v_fov[0]) / v_res))
        horiPix = int((np.absolute(self.h_fov[1] - self.h_fov[0]) / h_res))
        return horiPix,vertPix

    def generateRangeImage(self,points,recursive: bool, scale = 1):
        """
        v_res,h_res in degrees
        points[B,3,N] 
        """
        v_res, h_res = scale*self.v_res,scale*self.h_res

        batch_size, _, _ = points.size()

        # Get Number of pixels in range image
        vertPix = int((np.absolute(self.v_fov[1] - self.v_fov[0]) / v_res))
        horiPix = int((np.absolute(self.h_fov[1] - self.h_fov[0]) / h_res))

        # Get coordinates and distances
        if self.axisFront is None:
            x = points[:,0,:]
            y = points[:,1,:]
            z = points[:,2,:]
        elif self.axisFront == 'camera':
            x = points[:,2,:]
            y = -points[:,0,:]
            z = -points[:,1,:]
        dist = torch.sqrt(x**2 + y**2 + z**2)

        # Get all vertical angles
        verticalAngles = torch.arctan2(z, dist) / torch.pi * 180 # Degrees

        # Get all horizontal angles
        horizontalAngles = torch.arctan2(-y, x) / torch.pi * 180 # Degrees

        # Filter based on FOV setting
        combined_condition = (verticalAngles < self.v_fov[0]) & (verticalAngles > self.v_fov[1]) & (horizontalAngles > self.h_fov[0]) & (horizontalAngles < self.h_fov[1])

        rangeImage = torch.zeros((batch_size,1,horiPix,vertPix),dtype=torch.float32)

        for batch in range(batch_size):

            verticalAngles_batch = verticalAngles[batch,:][combined_condition[batch,:]]
            horizontalAngles_batch = horizontalAngles[batch,:][combined_condition[batch,:]]
            dist_batch = dist[batch,:][combined_condition[batch,:]]

            # Shift angles to all be positive
            verticalAnglesShifted_batch = (verticalAngles_batch - self.v_fov[0]) * -1
            horizontalAnglesShifted_batch = horizontalAngles_batch - self.h_fov[0]

            # Get image coordinates of all points
            x_img_fl = torch.round(horizontalAnglesShifted_batch / np.absolute(self.h_fov[1] - self.h_fov[0]) * (horiPix - 1))
            y_img_fl = torch.round(verticalAnglesShifted_batch / np.absolute(self.v_fov[1] - self.v_fov[0]) * (vertPix - 1))
            x_img = x_img_fl.int()
            y_img = y_img_fl.int()

            # Fill values in range image
            rangeImage[batch, :,x_img, y_img] = dist_batch

        if recursive == True:
            for i in range(10):
                # Define new scaling 
                scale *= self.scaling
                # Get empty values
                empty_pixels_mask = rangeImage == 0

                red_rangeImage = self.generateRangeImage(points, recursive = False, scale = scale)
                _,_, redwidth, redheight = red_rangeImage.size()

                red_x, red_y = torch.meshgrid(torch.arange(redwidth), torch.arange(redheight),indexing="ij")
                red_y = red_y.flatten().float()
                red_x = red_x.flatten().float()
                red_depth = red_rangeImage.flatten(start_dim=2)

                # Rescale coordinates to match original image
                red_y = red_y / redheight * (vertPix)
                red_x = red_x / redwidth * (horiPix)

                # Round and convert to int
                red_y = torch.round(red_y).int()
                red_x = torch.round(red_x).int()

                for batch in range(batch_size):

                    # Consider only coordinates that are empty in original depth image
                    condition = empty_pixels_mask[batch,0,red_x,red_y]
                    red_x_loc = red_x[condition]
                    red_y_loc = red_y[condition]
                    red_depth_loc = red_depth[batch,0,:][condition]

                    rangeImage[batch, :,red_x_loc, red_y_loc] = red_depth_loc

        return rangeImage
    
    def generateRangeImage_numpy(self,points,recursive: bool, scale = 1):
        """
        v_res,h_res in degrees
        """
        # Get Number of pixels in range image
        v_res, h_res = scale*self.v_res,scale*self.h_res
        vertPix = int((np.absolute(self.v_fov[1] - self.v_fov[0]) / v_res))
        horiPix = int((np.absolute(self.h_fov[1] - self.h_fov[0]) / h_res))

        # Get coordinates and distances
        if self.axisFront is None:
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
        elif self.axisFront == 'camera':
            x = points[:,2]
            y = -points[:,0]
            z = -points[:,1]
        dist = np.sqrt(x**2 + y**2 + z**2)

        # Get all vertical angles
        verticalAngles = np.arctan2(z, dist) / np.pi * 180 # Degrees

        # Get all horizontal angles
        horizontalAngles = np.arctan2(-y, x) / np.pi * 180 # Degrees

        # Filter based on FOV setting
        combined_condition = (verticalAngles < self.v_fov[0]) & (verticalAngles > self.v_fov[1]) & (horizontalAngles > self.h_fov[0]) & (horizontalAngles < self.h_fov[1])

        verticalAngles = verticalAngles[combined_condition]
        horizontalAngles = horizontalAngles[combined_condition]
        dist = dist[combined_condition]

        # Shift angles to all be positive
        verticalAnglesShifted = (verticalAngles - self.v_fov[0]) * -1
        horizontalAnglesShifted = horizontalAngles - self.h_fov[0]

        # Initialize Range image
        rangeImage = np.zeros((horiPix,vertPix),dtype=np.float32)

        # Get image coordinates of all points
        x_img_fl = np.round(horizontalAnglesShifted / np.absolute(self.h_fov[1] - self.h_fov[0]) * (horiPix - 1))
        y_img_fl = np.round(verticalAnglesShifted / np.absolute(self.v_fov[1] - self.v_fov[0]) * (vertPix - 1))
        x_img = x_img_fl.astype(int)
        y_img = y_img_fl.astype(int)

        # Fill values in range image
        rangeImage[x_img, y_img] = dist

        if recursive == True:
            for i in range(10):
                # Define new scaling 
                scale *= self.scaling
                # Get empty values
                empty_pixels_mask = rangeImage == 0

                red_rangeImage = self.generateRangeImage_numpy(points, recursive = False, scale = scale)
                redwidth, redheight = np.shape(red_rangeImage)
                red_x, red_y = np.meshgrid(np.arange(redwidth), np.arange(redheight),indexing="ij")
                red_y = red_y.flatten().astype(float)
                red_x = red_x.flatten().astype(float)
                red_depth = red_rangeImage.flatten()

                # Rescale coordinates to match original image
                red_y = red_y / redheight * (vertPix)
                red_x = red_x / redwidth * (horiPix)

                # Round and convert to int
                red_y = np.round(red_y).astype(int)
                red_x = np.round(red_x).astype(int)

                # Consider only coordinates that are empty in original depth image
                condition = empty_pixels_mask[red_x,red_y]
                red_x = red_x[condition]
                red_y = red_y[condition]
                red_depth = red_depth[condition]

                rangeImage[red_x,red_y] = red_depth

        return rangeImage
    
    def transform_pcd_and_range(self,pcd,Extran):
        pcd = se3.transform(Extran,pcd)
        rangeImage = self.generateRangeImage(pcd,recursive=True)
        return rangeImage, pcd
    
import cv2
if __name__=="__main__":
    pcd = torch.rand(2,3,20000).cuda()
    pcd1 = np.fromfile('KITTI_Odometry_Full/sequences/00/velodyne/000000.bin', dtype=np.float32).reshape((-1,4))[:120000,0:3].T
    pcd2 = np.fromfile('KITTI_Odometry_Full/sequences/00/velodyne/000001.bin', dtype=np.float32).reshape((-1,4))[:120000,0:3].T

    stacked_arr = np.stack([pcd1, pcd2], axis=0)

    pcd = torch.from_numpy(stacked_arr)

    
    v_fov, h_fov = (2, -25), (-60,60) ### START AT TOP LEFT OF IMAGE
    v_res= 0.42 # 0.42
    h_res= 0.35 # 0.35

    rangeGenerator = RangeImageGenerator(v_res=v_res,h_res=h_res,v_fov=v_fov,h_fov=h_fov, axisFront=None)
    rangeImages = rangeGenerator.generateRangeImage(pcd,recursive=True)
    rangeImag_np = rangeGenerator.generateRangeImage_numpy(pcd1.T,recursive=True)

    testimg = rangeImages[0,0,:,:].squeeze().detach().numpy().T
    cv2.imwrite('rangeImg.png', testimg)
    cv2.imwrite('rangeImg_np.png', rangeImag_np)
    print('end')