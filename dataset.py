import os
import json
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as Tf
import numpy as np
import pykitti
import open3d as o3d
# from dataset import BaseKITTIDataset
from utils import transform, se3
from PIL import Image
import cv2

import open3d as o3d

def check_length(root:str,save_name='data_len.json'):
    seq_dir = os.path.join(root,'sequences')
    seq_list = os.listdir(seq_dir)
    seq_list.sort()
    dict_len = dict()
    for seq in seq_list:
        len_velo = len(os.listdir(os.path.join(seq_dir,seq,'velodyne')))
        dict_len[seq]=len_velo
    with open(os.path.join(root,save_name),'w')as f:
        json.dump(dict_len,f)
        
class KITTIFilter:
    def __init__(self,voxel_size=0.3,concat:str = 'none'):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.concat = concat
        
    def __call__(self, x:np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd_xyz = np.array(pcd.points,dtype=np.float32)
        if self.concat == 'none':
            return pcd_xyz
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*3, max_nn=30))
            pcd.normalize_normals()
            pcd_norm = np.array(pcd.normals,dtype=np.float32)
            if self.concat == 'xyz':
                return np.hstack([pcd_xyz,pcd_norm])  # (N,3), (N,3) -> (N,6)
            elif self.concat == 'zero-mean':  # 'zero-mean'
                center = np.mean(pcd_xyz,axis=0,keepdims=True)  # (3,)
                pcd_zero = pcd_xyz - center
                pcd_norm *= np.where(np.sum(pcd_zero*pcd_norm,axis=1,keepdims=True)<0,-1,1)
                return np.hstack([pcd_zero,pcd_norm]) # (N,3),(N,3) -> (N,6)
            else:
                raise RuntimeError('Unknown concat mode: %s'%self.concat)

class Resampler:
    """ [N, D] -> [M, D]\n
    used for training
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, x: np.ndarray):
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num < 0:
            return x[idx]
        elif self.num <= num_points:
            idx = idx[:self.num] # (self.num,dim)
            return x[idx]
        else:
            idx = np.hstack([idx,np.random.choice(num_points,self.num-num_points,replace=True)]) # (self.num,dim)
            return x[idx]

class MaxResampler:
    """ [N, D] -> [M, D] (M<=max_num)\n
    used for testing
    """
    def __init__(self,num,seed=8080):
        self.num = num
        np.random.seed(seed)  # fix randomly sampling in test pipline
    def __call__(self, x:np.ndarray):
        num_points = x.shape[0]
        x_ = np.random.permutation(x)
        if num_points <= self.num:
            return x_  # permutation
        else:
            return x_[:self.num]

class ToTensor:
    def __init__(self,type=torch.float):
        self.tensor_type = type
    
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)



class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str,batch_size:int,seqs=['09','10'],cam_id:int=2,
                 meta_json='data_len.json',skip_frame=1,
                 voxel_size=0.3,pcd_sample_num=4096,resize_ratio=(0.5,0.5),extend_intran=(2.5,2.5),
                 ):
        if not os.path.exists(os.path.join(basedir,meta_json)):
            check_length(basedir,meta_json)
        with open(os.path.join(basedir,meta_json),'r')as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0,dict_len[seq],skip_frame))
            cut_index = len(frame)%batch_size
            if cut_index > 0:
                frame = frame[:-cut_index]
            frame_list.append(frame)
        self.kitti_datalist = [pykitti.odometry(basedir,seq,frames=frame) for seq,frame in zip(seqs,frame_list)]  
        # concat images from different seq into one batch will cause error
        self.cam_id = cam_id
        self.resize_ratio = resize_ratio
        for seq,obj in zip(seqs,self.kitti_datalist):
            self.check(obj,cam_id,seq)
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)
        self.resample_tran = Resampler(pcd_sample_num)
        self.tensor_tran = ToTensor()
        self.img_tran = Tf.ToTensor()
        self.pcd_tran = KITTIFilter(voxel_size,'none')
        self.extend_intran = extend_intran
        
    def __len__(self):
        return self.sumsep[-1]
    @staticmethod
    def check(odom_obj:pykitti.odometry,cam_id:int,seq:str)->bool:
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj,'cam%d_files'%cam_id))
        velo_files_lenght = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:'%seq
        assert cam_files_length>0, head_msg+'None of camera %d files'%cam_id
        assert cam_files_length==velo_files_lenght, head_msg+"number of cam %d (%d) and velo files (%d) doesn't equal!"%(cam_id,cam_files_length,velo_files_lenght)
        assert hasattr(calib,'T_cam0_velo'), head_msg+"Crucial calib attribute 'T_cam0_velo' doesn't exist!"
        
    
    def __getitem__(self, index):
        group_id = np.digitize(index,self.sumsep,right=False)
        data = self.kitti_datalist[group_id]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%self.cam_id)
        K_cam = np.diag([self.resize_ratio[1],self.resize_ratio[0],1]) @ getattr(data.calib,'K_cam%d'%self.cam_id)       
        if group_id > 0:
            sub_index = index - self.sumsep[group_id-1]
        else:
            sub_index = index
        raw_img = getattr(data,'get_cam%d'%self.cam_id)(sub_index)  # PIL Image
        H,W = raw_img.height, raw_img.width
        RH = round(H*self.resize_ratio[0])
        RW = round(W*self.resize_ratio[1])
        REVH,REVW = self.extend_intran[0]*RH,self.extend_intran[1]*RW
        K_cam_extend = K_cam.copy()
        K_cam_extend[0,-1] *= self.extend_intran[0]
        K_cam_extend[1,-1] *= self.extend_intran[1]
        raw_img = raw_img.resize([RW,RH],Image.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd = data.get_velo(sub_index)
        pcd[:,3] = 1.0  # (N,4)
        calibed_pcd = T_cam2velo @ pcd.T  # [4,4] @ [4,N] -> [4,N]
        _calibed_pcd = self.pcd_tran(calibed_pcd[:3,:].T).T  # raw pcd input (3,N)
        *_,rev = transform.binary_projection((REVH,REVW),K_cam_extend,_calibed_pcd)
        _calibed_pcd = _calibed_pcd[:,rev]  
        _calibed_pcd = self.resample_tran(_calibed_pcd.T).T # (3,n)
        _pcd_range = np.linalg.norm(_calibed_pcd,axis=0)  # (n,)
        u,v,r,_ = transform.pcd_projection((RH,RW),K_cam,_calibed_pcd,_pcd_range)
        _depth_img = torch.zeros(RH,RW,dtype=torch.float32)
        _depth_img[v,u] = torch.from_numpy(r).type(torch.float32)
        _calibed_pcd = self.tensor_tran(_calibed_pcd)
        _pcd_range = self.tensor_tran(_pcd_range)
        K_cam = self.tensor_tran(K_cam)
        T_cam2velo = self.tensor_tran(T_cam2velo)
        return dict(img=_img,pcd=_calibed_pcd,pcd_range=_pcd_range,depth_img=_depth_img,
                    InTran=K_cam,ExTran=T_cam2velo)

class KITTI_perturb(Dataset):
    def __init__(self,dataset:BaseKITTIDataset,max_deg:float,max_tran:float,mag_randomly=True,pooling_size=5,file=None):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        self.dataset = dataset
        self.file = file
        if self.file is not None:
            self.perturb = torch.from_numpy(np.loadtxt(self.file,dtype=np.float32,delimiter=','))[None,...]  # (1,N,6)
        else:
            self.transform = transform.UniformTransformSE3(max_deg,max_tran,mag_randomly)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        H,W = data['img'].shape[-2:]  # (RH,RW)
        calibed_pcd = data['pcd']  # (3,N)
        InTran = data['InTran']  # (3,3)
        if self.file is None:  # randomly generate igt
            _uncalibed_pcd = self.transform(calibed_pcd[None,:,:]).squeeze(0)  # (3,N)
            igt = self.transform.igt.squeeze(0)  # (4,4)
        else:
            igt = se3.exp(self.perturb[:,index,:])  # (1,6) -> (1,4,4)
            _uncalibed_pcd = se3.transform(igt,calibed_pcd[None,...]).squeeze(0)  # (3,N)
            igt.squeeze_(0)  # (4,4)
        _uncalibed_depth_img = torch.zeros_like(data['depth_img'],dtype=torch.float32)
        proj_pcd = InTran.matmul(_uncalibed_pcd)  # (3,3)x(3,N) -> (3,N)
        proj_x = (proj_pcd[0,:]/proj_pcd[2,:]).type(torch.long)
        proj_y = (proj_pcd[1,:]/proj_pcd[2,:]).type(torch.long)
        rev = (0<=proj_x)*(proj_x<W)*(0<=proj_y)*(proj_y<H)*(proj_pcd[2,:]>0)
        proj_x = proj_x[rev]
        proj_y = proj_y[rev]
        _uncalibed_depth_img[proj_y,proj_x] = data['pcd_range'][rev]  # H,W
        # add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd,uncalibed_depth_img=_uncalibed_depth_img,igt=igt)
        data.update(new_data)
        data['depth_img'] = self.pooling(data['depth_img'][None,...])
        data['uncalibed_depth_img'] = self.pooling(data['uncalibed_depth_img'][None,...])
        return data
    

class KITTI_Base_rangeImage(BaseKITTIDataset):
    def __init__(self, basedir: str, batch_size: int, rangeImageGenerator: transform.RangeImageGenerator, seqs=['09', '10'], cam_id: int = 2, meta_json='data_len.json', skip_frame=1, voxel_size=0.3, pcd_sample_num=4096, resize_ratio=(0.5, 0.5), extend_intran=(2.5, 2.5)):
        super().__init__(basedir, batch_size, seqs, cam_id, meta_json, skip_frame, voxel_size, pcd_sample_num, resize_ratio, extend_intran)

        self.rangeImageGenerator = rangeImageGenerator

        return
    
    def __getitem__(self, index):
        group_id = np.digitize(index,self.sumsep,right=False)
        data = self.kitti_datalist[group_id]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%self.cam_id)
        K_cam = np.diag([self.resize_ratio[1],self.resize_ratio[0],1]) @ getattr(data.calib,'K_cam%d'%self.cam_id)       
        if group_id > 0:
            sub_index = index - self.sumsep[group_id-1]
        else:
            sub_index = index
        raw_img = getattr(data,'get_cam%d'%self.cam_id)(sub_index)  # PIL Image
        H,W = raw_img.height, raw_img.width
        RH = round(H*self.resize_ratio[0])
        RW = round(W*self.resize_ratio[1])
        REVH,REVW = self.extend_intran[0]*RH,self.extend_intran[1]*RW
        K_cam_extend = K_cam.copy()
        K_cam_extend[0,-1] *= self.extend_intran[0]
        K_cam_extend[1,-1] *= self.extend_intran[1]
        raw_img = raw_img.resize([RW,RH],Image.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd = data.get_velo(sub_index)
        pcd[:,3] = 1.0  # (N,4)
        calibed_pcd = T_cam2velo @ pcd.T  # [4,4] @ [4,N] -> [4,N]
        _calibed_pcd = self.pcd_tran(calibed_pcd[:3,:].T).T  # raw pcd input (3,N)
        *_,rev = transform.binary_projection((REVH,REVW),K_cam_extend,_calibed_pcd)
        _calibed_pcd = _calibed_pcd[:,rev]  
        _calibed_pcd = self.resample_tran(_calibed_pcd.T).T # (3,n)
        _pcd_range = np.linalg.norm(_calibed_pcd,axis=0)  # (n,)
        u,v,r,_ = transform.pcd_projection((RH,RW),K_cam,_calibed_pcd,_pcd_range)

        _depth_img = self.rangeImageGenerator.generateRangeImage_numpy(pcd,recursive=True)

        _calibed_pcd = self.tensor_tran(_calibed_pcd)
        _pcd_range = self.tensor_tran(_pcd_range)
        K_cam = self.tensor_tran(K_cam)
        T_cam2velo = self.tensor_tran(T_cam2velo)
        return dict(img=_img,pcd=_calibed_pcd,pcd_range=_pcd_range,depth_img=_depth_img,
                    InTran=K_cam,ExTran=T_cam2velo)

class KITTI_Perturb_rangeImage(KITTI_perturb):
    def __init__(self, dataset: KITTI_Base_rangeImage, max_deg: float, max_tran: float, mag_randomly=True, pooling_size=5, file=None):
        super().__init__(dataset, max_deg, max_tran, mag_randomly, pooling_size, file)

        return
    
    def __getitem__(self, index):
        data = self.dataset[index]
        H,W = data['img'].shape[-2:]  # (RH,RW)
        calibed_pcd = data['pcd']  # (3,N)
        InTran = data['InTran']  # (3,3)
        if self.file is None:  # randomly generate igt
            _uncalibed_pcd = self.transform(calibed_pcd[None,:,:]).squeeze(0)  # (3,N)
            igt = self.transform.igt.squeeze(0)  # (4,4)
        else:
            igt = se3.exp(self.perturb[:,index,:])  # (1,6) -> (1,4,4)
            _uncalibed_pcd = se3.transform(igt,calibed_pcd[None,...]).squeeze(0)  # (3,N)
            igt.squeeze_(0)  # (4,4)

        rangeImage = self.dataset.rangeImageGenerator.generateRangeImage(_uncalibed_pcd[None],recursive=True)[0,:,:,:]

        # add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd,uncalibed_depth_img=rangeImage,igt=igt)
        data.update(new_data)
        # data['depth_img'] = self.pooling(data['depth_img'][None,...])
        # data['uncalibed_depth_img'] = self.pooling(data['uncalibed_depth_img'][None,...])
        return data

class BaseONCEDataset(Dataset):
    def __init__(self,basedir:str,batch_size:int,seqs=['000076','000080'],cam_id:int=1,skip_frame=1,
                 voxel_size=0.3,pcd_sample_num=20000,resize_ratio=(0.5,0.5),extend_intran=(2.5,2.5),
                 ):
        
        self.skip_frame = skip_frame
        
        self.base_dir = basedir
        self.seqs = seqs
        self.cam_id = cam_id
        self.camera_dir = 'cam0' + str(cam_id)
        self.lidar_dir = 'lidar_roof'

        self.drives_dir = []
        for seq in seqs:
            dir = self.base_dir + '/' + seq
            self.drives_dir.append(dir)

        # Get image and LiDAR paths
        self.images_pths = {key: None for key in seqs}
        self.lidar_pths  = {key: None for key in seqs}
        for i,dir in enumerate(self.drives_dir):
            images = sorted([os.path.join(dir, self.camera_dir, img) for img in os.listdir(os.path.join(dir, self.camera_dir))])
            images_filt = images[::skip_frame]
            self.images_pths[self.seqs[i]] = images_filt
            lidar = sorted([os.path.join(dir, self.lidar_dir, img) for img in os.listdir(os.path.join(dir, self.lidar_dir))])
            lidar_filt = lidar[::skip_frame]
            self.lidar_pths[self.seqs[i]] = lidar_filt

        # Get calibration
        self.calibdata = {key: None for key in seqs}
        for i,dir in enumerate(self.drives_dir):
            fileName = seqs[i] + '.json'
            with open(os.path.join(dir, fileName), 'r')as f:
                calib = json.load(f)
            self.calibdata[seqs[i]] = calib['calib'][self.camera_dir]

        # Concatenate data to enable indexing
        self.elements_per_seq = {}
        for i,seq in enumerate(self.seqs):
            length_im = len(self.images_pths[seq])
            self.elements_per_seq[seq] = length_im

        self.idx_cumsum = {}
        for i,seq in enumerate(self.seqs):
            if i == 0:
                self.idx_cumsum[seq] = self.elements_per_seq[seq]
            else:
                self.idx_cumsum[seq] = self.elements_per_seq[seq] + self.elements_per_seq[self.seqs[i-1]]
                
        self.resize_ratio = resize_ratio
        
        self.resample_tran = Resampler(pcd_sample_num)
        self.tensor_tran = ToTensor()
        self.img_tran = Tf.ToTensor()
        self.pcd_tran = KITTIFilter(voxel_size,'none')
        self.extend_intran = extend_intran

        return
        
    def __len__(self):
        return self.idx_cumsum[self.seqs[-1]]
    
    
    def __getitem__(self, index):

        downsample = 0.5

        # Get image and lidar for given index
        curr_drive = None
        idx_loc = index

        for i,seq in enumerate(self.seqs):
            if (idx_loc - self.elements_per_seq[seq] + 1) <= 0:
                curr_drive = seq
                break
            else:
                idx_loc -= self.elements_per_seq[seq] + 1

        cam_pth = self.images_pths[curr_drive][idx_loc]
        lidr_pth = self.lidar_pths[curr_drive][idx_loc]
        calib_data = self.calibdata[curr_drive]

        K_cam_orig = np.array(calib_data['cam_intrinsic'])
        dist = np.array(calib_data['distortion'])

        raw_img_dis = Image.open(cam_pth).convert('RGB')
        raw_img = cv2.undistort(np.asarray(raw_img_dis),K_cam_orig,distCoeffs=dist)
        pcd = np.fromfile(lidr_pth, dtype=np.float32).reshape((-1,4))

        K_cam_orig = K_cam_orig * downsample
        K_cam_orig[2,2] = 1
        K_cam = np.diag([self.resize_ratio[1],self.resize_ratio[0],1]) @ K_cam_orig
        
        T_cam2velo = np.array(calib_data['cam_to_velo'])

        # T_cam2velo[:3,-1] = -T_cam2velo[:3,-1]
        # rotmat = np.linalg.inv(T_cam2velo[:3,:3])
        # T_cam2velo[:3,:3] = rotmat

        

        
        h,w,c = np.shape(raw_img)
        h = int(h*downsample)
        w = int(w*downsample)
        raw_img = cv2.resize(raw_img,(w,h), np.zeros((4,1)),interpolation=cv2.INTER_NEAREST)
        raw_img = Image.fromarray(raw_img)

        H,W = raw_img.height, raw_img.width
        RH = round(H*self.resize_ratio[0])
        RW = round(W*self.resize_ratio[1])
        REVH,REVW = self.extend_intran[0]*RH,self.extend_intran[1]*RW
        K_cam_extend = K_cam.copy()
        K_cam_extend[0,-1] *= self.extend_intran[0]
        K_cam_extend[1,-1] *= self.extend_intran[1]
        raw_img = raw_img.resize([RW,RH],Image.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd[:,3] = 1.0  # (N,4)
        calibed_pcd = (T_cam2velo @ pcd.T)[:3,:] # [4,4] @ [4,N] -> [4,N]
        _calibed_pcd = self.pcd_tran(calibed_pcd[:3,:].T).T  # raw pcd input (3,N)
        *_,rev = transform.binary_projection((REVH,REVW),K_cam_extend,_calibed_pcd)
        _calibed_pcd = _calibed_pcd[:,rev]  
        _calibed_pcd = self.resample_tran(_calibed_pcd.T).T # (3,n)
        _pcd_range = np.linalg.norm(_calibed_pcd,axis=0)  # (n,)
        u,v,r,_ = transform.pcd_projection((RH,RW),K_cam,_calibed_pcd,_pcd_range)
        _depth_img = torch.zeros(RH,RW,dtype=torch.float32)
        _depth_img[v,u] = torch.from_numpy(r).type(torch.float32)
        _calibed_pcd = self.tensor_tran(_calibed_pcd)
        _pcd_range = self.tensor_tran(_pcd_range)
        K_cam = self.tensor_tran(K_cam)
        T_cam2velo = self.tensor_tran(T_cam2velo)

        return dict(img=_img,pcd=_calibed_pcd,pcd_range=_pcd_range,depth_img=_depth_img,
                    InTran=K_cam,ExTran=T_cam2velo)        
        
if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('Agg')
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    # once_base_dataset = BaseONCEDataset(basedir='ONCE/data',
    #                                     batch_size=1,
    #                                     skip_frame=2,
    #                                     cam_id=1)
    # once_base_data_pert = KITTI_perturb(dataset=once_base_dataset,
    #                                     max_deg=10,
    #                                     max_tran=3)
    # base_dataset = BaseKITTIDataset('KITTI_Odometry_Full',1,seqs=['00','01'],skip_frame=3)
    # dataset = KITTI_perturb(base_dataset,30,3)
    # data = once_base_data_pert[2]
    # for key,value in data.items():
    #     if isinstance(value,torch.Tensor):
    #         shape = value.size()
    #     else:
    #         shape = value
    #     print('{key}: {shape}'.format(key=key,shape=shape))
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(data['depth_img'].squeeze(0).numpy(), cmap='gray')
    # plt.subplot(1,3,2)
    # plt.imshow(data['uncalibed_depth_img'].squeeze(0).numpy(), cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(np.moveaxis(data['img'].squeeze(0).numpy(),0,-1))
    # plt.savefig('dataset_demo.png', dpi = 800)

    from torch.utils.data import DataLoader
    import loss as loss_utils

    datasetpath = 'KITTI_Odometry_Full'
    seqs = ['02']

    dataset = BaseKITTIDataset(basedir=datasetpath,
                               batch_size=1,
                               seqs=seqs)
    dataset_pert = KITTI_perturb(dataset=dataset,
                                 max_deg=10,
                                 max_tran=0.1,
                                 mag_randomly=True,
                                 file='checkpoint/test_seq.csv')
    dataloader = DataLoader(dataset=dataset_pert,
                            batch_size=20,
                            num_workers=12)
    
    n = len(dataset_pert)
    tf_mat = np.zeros((n,6))
    j = 0
    for batch in dataloader:
        for b in range(batch['igt'].size(0)):
            tf_mat[j+b,0:3],tf_mat[j+b,3:] = loss_utils.gt2euler(batch['igt'][b,:,:].squeeze(0).cpu().detach().numpy())

        # tf_mat[j:j+batch['igt'].size(0),:,:] = batch['igt']
        j+=batch['igt'].size(0)
        print(j)

    dist_tran = tf_mat[:,3:]
    dist_tran_abs = dist_tran.mean(axis = 1)

    dist_rot = tf_mat[:,0:3]
    dist_rot_abs = np.degrees(dist_rot.mean(axis=1))

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    # Plot the distribution of translations
    ax1.hist(dist_tran_abs, bins=50, density=True, alpha=0.7)
    ax1.set_xlabel('Translational Perturbation in [m]')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Distribution of Translations')

    # Plot the distribution of rotations
    ax2.hist(dist_rot_abs, bins=50, density=True, alpha=0.7)
    ax2.set_xlabel('Rotational Perturbation in [Deg]')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Distribution of Rotations')

    plt.tight_layout()  # Adjust the spacing between subplots

    # Show the plot
    plt.show()

    plt.savefig('perturbation_distribution.png', dpi = 500)

    pass