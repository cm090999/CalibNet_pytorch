import argparse
import os
import yaml
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb, BaseONCEDataset
from mylogger import get_logger, print_highlight, print_warning
from CalibNet import CalibNet, CalibNet_DINOV2, CalibNet_DINOV2_patch, CalibNet_DINOV2_LTC, CalibNet_DINOV2_patch_CalAgg, CalibNet_DINOV2_patch_RGB_CalAgg, CalibNet_DINOV2_patch_RGB
import loss as loss_utils
import utils
import numpy as np

from utils.visualizations import vis_eval, printStatistics
import cv2
from get_statistics import get_stats

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='KITTI_Odometry_Full')
    parser.add_argument("--dataset", type=str, default = 'kitti', choices=['kitti', 'once'] )
    parser.add_argument("--skip_frame",type=int,default=10,help='skip frame of dataset')
    parser.add_argument("--pcd_sample",type=int,default=10000) # -1 means total sample
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    parser.add_argument("--randomCrop",type=float,default=1.0) # None or float in (0,1)
    parser.add_argument("--perturbationaxes",type=str,default='1,1,1,1,1,1') # Enable 1 or disable 0 perturbation in all directions, rotx(pitch), roty(yaw), rotz(roll) , tslx, tsly, tslz,  camera image coordinates
    parser.add_argument("--pertFile",type=str,default='test_seq.csv')
    parser.add_argument("--singlePerturbation", type=bool, default=False)
    # dataloader
    parser.add_argument("--batch_size",type=int,default=1,choices=[1],help='batch size of test dataloader must be 1')
    parser.add_argument("--num_workers",type=int,default=8)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your CPU memory is insufficient')
    parser.add_argument("--perturb_file",type=str,default='checkpoint/test_seq.csv')
    parser.add_argument("--finetune_tsl",type=bool,default=False)
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--pretrained",type=str,default='checkpoint/download_finetuned_no_randomCrop_ch_best.pth') # checkpoint/download.pth # checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth # checkpoint/download_finetuned_no_randomCrop_ch_best.pth
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--res_dir",type=str,default='res/')
    parser.add_argument("--name",type=str,default='cam2_oneiter')
    parser.add_argument("--model_name", type=str,default = 'CalibNet', choices=['CalibNet', 'CalibNet_DINOV2', 'CalibNet_DINOV2_patch','CalibNet_DINOV2_patch_RGB', 'CalibNet_DINOV2_LTC','CalibNet_DINOV2_patch_RGB_CalAgg', 'CalibNet_DINOV2_patch_CalAgg'])
    parser.add_argument("--depth_modality", type = str, default = 'depthimage', choices=['depthimage', 'rangeimage'])
    parser.add_argument("--visualize", type = bool, default = False)

    # setting
    parser.add_argument("--inner_iter",type=int,default=4,help='inner iter of calibnet')
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()

def test(args,chkpt:dict,test_loader):
    if args.model_name == 'CalibNet':
        model = CalibNet(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2':
        model = CalibNet_DINOV2(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2_patch':
        model = CalibNet_DINOV2_patch(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2_LTC':
        model = CalibNet_DINOV2_LTC(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2_patch_CalAgg':
        model = CalibNet_DINOV2_patch_CalAgg(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2_patch_RGB':
        model = CalibNet_DINOV2_patch_RGB(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2_patch_RGB_CalAgg':
        model = CalibNet_DINOV2_patch_RGB_CalAgg(depth_scale=args.scale)

    device = torch.device(args.device)
    model.to(device)
    model.load_state_dict(chkpt['model'])
    model.eval()

    logger = get_logger('{name}-Test'.format(name=args.name),os.path.join(args.log_dir,args.name+'_test.log'),mode='w')
    logger.debug(args)
    res_npy_angleAxis = np.zeros([len(test_loader),6])
    res_npy_euler = np.zeros([len(test_loader),6])

    zero_res_npy_angleAxis = np.zeros([len(test_loader),6])
    zero_res_npy_euler = np.zeros([len(test_loader),6])

    ## Store stats 
    # # Intermediate Layers 
    # rgb_dinov2 = np.zeros([len(test_loader),384])
    # dep_dinov2 = np.zeros([len(test_loader),384])
    # Decalibration
    igt_npy = np.zeros([len(test_loader),4,4])
    # Model Output 
    recalib_npy = np.zeros([len(test_loader),4,4])

    tf_mat_input = np.zeros((len(test_loader),6))
    tf_mat_output = np.zeros((len(test_loader),6))
    j = 0

    for i,batch in enumerate(test_loader):
        rgb_img = batch['img'].to(device)
        B = rgb_img.size(0)
        pcd_range = batch['pcd_range'].to(device)
        uncalibed_pcd = batch['uncalibed_pcd'].to(device)
        uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
        InTran = batch['InTran'][0].to(device)
        igt = batch['igt'].to(device)
        img_shape = rgb_img.shape[-2:]
        depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
        # model(rgb_img,uncalibed_depth_img)
        miscal_pcd = uncalibed_pcd
        g0 = torch.eye(4).repeat(B,1,1).to(device)
        for _ in range(args.inner_iter):
            twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)

            extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))

            

            if args.finetune_tsl:
                with torch.no_grad():
                    extran_rot = extran.clone()
                    extran_rot[:3, 3] = extran_rot[:3, 3] * 0
                uncalibed_depth_img, uncalibed_pcd = depth_generator(extran_rot, uncalibed_pcd)
                twist_rot, twist_tsl = model(rgb_img, uncalibed_depth_img)
                extran_tsl = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))
                extran = extran_rot.bmm(extran_tsl)
                finetune = 0
            else:
                uncalibed_depth_img, uncalibed_pcd = depth_generator(extran, uncalibed_pcd)
                finetune = 1

            # uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
            g0 = extran.bmm(g0)
        dg = g0.bmm(igt)
        rot_dx,tsl_dx = loss_utils.gt2euler(dg.squeeze(0).cpu().detach().numpy()) # cv2.Rodrigues(dg.squeeze(0).cpu().detach().numpy()[:3, :3])[0], dg.squeeze(0).cpu().detach().numpy()[:3, 3]
        rot_dx = rot_dx.reshape(-1)
        tsl_dx = tsl_dx.reshape(-1)
        res_npy_euler[i,:] = np.abs(np.concatenate([rot_dx,tsl_dx]))
        rot_dx_,tsl_dx_ = cv2.Rodrigues(dg.squeeze(0).cpu().detach().numpy()[:3, :3])[0], dg.squeeze(0).cpu().detach().numpy()[:3, 3]
        rot_dx_ = rot_dx_.reshape(-1)
        tsl_dx_ = tsl_dx_.reshape(-1)
        res_npy_angleAxis[i,:] = np.abs(np.concatenate([rot_dx_,tsl_dx_]))
        logger.info('[{:05d}|{:05d}],mdx:{:.4f}'.format(i+1,len(test_loader),res_npy_euler[i,:].mean().item()))

        # Identity Estimate for comparison
        xident = torch.eye(4, device=device)
        xident = xident.reshape((1,4,4))
        ident = xident.repeat(B, 1, 1)
        d_id = ident.bmm(igt)
        zero_rot_dx,zero_tsl_dx = loss_utils.gt2euler(d_id.squeeze(0).cpu().detach().numpy()) # cv2.Rodrigues(d_id.squeeze(0).cpu().detach().numpy()[:3, :3])[0], d_id.squeeze(0).cpu().detach().numpy()[:3, 3]  
        zero_rot_dx = zero_rot_dx.reshape(-1)
        zero_tsl_dx = zero_tsl_dx.reshape(-1)
        zero_res_npy_euler[i,:] = np.abs(np.concatenate([zero_rot_dx,zero_tsl_dx]))
        zero_rot_dx_,zero_tsl_dx_ = cv2.Rodrigues(d_id.squeeze(0).cpu().detach().numpy()[:3, :3])[0], d_id.squeeze(0).cpu().detach().numpy()[:3, 3]  
        zero_rot_dx_ = zero_rot_dx_.reshape(-1)
        zero_tsl_dx_ = zero_tsl_dx_.reshape(-1)
        zero_res_npy_angleAxis[i,:] = np.abs(np.concatenate([zero_rot_dx_,zero_tsl_dx_]))
        logger.info('[{:05d}|{:05d}],mdx identity:{:.4f}'.format(i+1,len(test_loader),zero_res_npy_euler[i,:].mean().item()))

        igt_npy[i,:] = igt.squeeze().detach().cpu().numpy()
        recalib_npy[i,:] = g0.squeeze().detach().cpu().numpy()

        if args.visualize == True:
            pcd_gt = np.asarray(batch['pcd'].detach().cpu().squeeze())
            pcd_miscalib=np.asarray(miscal_pcd.detach().cpu().squeeze())
            pcd_corrected=np.asarray(uncalibed_pcd.detach().cpu().squeeze())
            rgb=(np.transpose(np.asarray(rgb_img.detach().cpu().squeeze()), (1, 2, 0) ))
            vis_eval(pcd_gt=pcd_gt,
                    pcd_miscalib=pcd_miscalib,
                    pcd_corrected=pcd_corrected,
                    rgb=rgb,
                    intran=np.asarray(InTran.squeeze().detach().cpu()),
                    savePath='visualizations',
                    saveName='frame' + str(i) + '.png',
                    igt=igt.squeeze().detach().cpu(),
                    networkOutput=g0.squeeze().detach().cpu())
        
        for b in range(batch['igt'].size(0)):
            inp_rot_dx, inp_tsl_dx = loss_utils.gt2euler(batch['igt'][b,:,:].squeeze(0).cpu().detach().numpy())
            out_rot_dx, out_tsl_dx = loss_utils.gt2euler(g0[b,:,:].squeeze(0).cpu().detach().numpy())
            
            inp_rot_dx = inp_rot_dx.reshape(-1)
            inp_tsl_dx = inp_tsl_dx.reshape(-1)
            out_rot_dx = out_rot_dx.reshape(-1)
            out_tsl_dx = out_tsl_dx.reshape(-1)
            tf_mat_input[j+b,:] =  np.abs(np.concatenate([inp_rot_dx,inp_tsl_dx]))
            tf_mat_output[j+b,:] =  np.abs(np.concatenate([out_rot_dx,out_tsl_dx]))
        j+=batch['igt'].size(0)

    # np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name+'rgb_dinov2'))),rgb_dinov2)
    # np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name+'dep_dinov2'))),dep_dinov2)
    np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name+'igt_npy'))),igt_npy)
    np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name+'recalib_npy'))),recalib_npy)
    np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name+'res_npy'))),res_npy_euler)
    
    logger.info('Model Output Statistics Euler: ')
    get_stats(res_npy_euler,logger=logger)
    logger.info('Zero Output Statistics Euler: ')
    get_stats(zero_res_npy_euler,logger=logger)
    logger.info('Model Output Statistics Angle Axis: ')
    get_stats(res_npy_angleAxis,logger=logger)
    logger.info('Zero Output Statistics Angle Axis: ')
    get_stats(zero_res_npy_angleAxis,logger=logger)

    printStatistics(tf_mat=res_npy_euler, fileName='visualizations/'+args.name+'StatisticError.png')
    printStatistics(tf_mat=tf_mat_input, fileName='visualizations/'+args.name+'StatisticsInput.png')
    printStatistics(tf_mat=tf_mat_output, fileName='visualizations/'+args.name+'StatisticsOutput.png')

if __name__ == "__main__":
    args = options()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    else:
        args.device = 'cuda:0'

    os.makedirs(args.log_dir,exist_ok=True)
    print('Using device: ' + str(args.device))
    with open(args.config,'r')as f:
        CONFIG : dict= yaml.load(f,yaml.SafeLoader)
    if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
        chkpt = torch.load(args.pretrained, map_location=args.device)
        CONFIG.update(chkpt['config'])
        update_args = ['resize_ratio','name','scale']
        for up_arg in update_args:
            setattr(args,up_arg,chkpt['args'][up_arg]) 
        args.name = args.name + '_' + args.dataset
    else:
        raise FileNotFoundError('pretrained checkpoint {:s} not found!'.format(os.path.abspath(args.pretrained)))
    
    perturbationaxes = args.perturbationaxes.split(',')
    args.perturbationaxes = [int(value) for value in perturbationaxes]

    # Load dataset config
    with open('dataset_paths.yml','r')as f:
        DATA : dict = yaml.load(f,yaml.SafeLoader)
    print_highlight('args have been received, please wait for dataloader...')
    
    test_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['test']] # ['00','01','02','03','04','05','06','07']#['02']# 

    if args.dataset == "once":
        test_dataset = BaseONCEDataset(basedir=DATA['once'] ,
                                        batch_size=args.batch_size,
                                        seqs=['000076'],
                                        skip_frame=args.skip_frame,
                                        voxel_size=CONFIG['dataset']['voxel_size'],
                                        pcd_sample_num=args.pcd_sample,
                                        resize_ratio=[0.5,0.5],
                                        extend_intran=CONFIG['dataset']['extend_ratio'],
                                        randomCrop=args.randomCrop)
    if args.dataset == "kitti":
        test_dataset = BaseKITTIDataset(basedir=DATA['kitti_full'],
                                        batch_size=args.batch_size,
                                        seqs=test_split,
                                        cam_id=CONFIG['dataset']['cam_id'],
                                        meta_json='data_len.json',
                                        skip_frame=args.skip_frame,
                                        voxel_size=CONFIG['dataset']['voxel_size'],
                                        pcd_sample_num=args.pcd_sample,
                                        resize_ratio=args.resize_ratio,
                                        extend_intran=CONFIG['dataset']['extend_ratio'],
                                        randomCrop=args.randomCrop)
        
    os.makedirs(args.res_dir,exist_ok=True)
    test_perturb_file = os.path.join(args.checkpoint_dir,args.pertFile)
    test_length = len(test_dataset)
    
    if args.singlePerturbation == False:
        if not os.path.exists(test_perturb_file):
            print_highlight("validation pertub file dosen't exist, create one.")
            transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly, axes=torch.tensor(args.perturbationaxes))
            perturb_arr = np.zeros([test_length,6])
            for i in range(test_length):
                perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(test_perturb_file,perturb_arr,delimiter=',')
        else:  # check length
            test_seq = np.loadtxt(test_perturb_file,delimiter=',')
            if test_length != test_seq.shape[0]:
                print_warning('Incompatiable test length {}!={}'.format(test_length,test_seq.shape[0]))
                transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly, axes=torch.tensor(args.perturbationaxes))
                perturb_arr = np.zeros([test_length,6])
                for i in range(test_length):
                    perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
                np.savetxt(test_perturb_file,perturb_arr,delimiter=',')
                print_highlight('Validation perturb file rewritten.')
            
    test_dataset = KITTI_perturb(test_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=CONFIG['dataset']['pooling'],axes=torch.tensor(args.perturbationaxes),file=test_perturb_file, singlePerturbation=args.singlePerturbation)
    print('Finished creating test dataset')
    
    test_dataloader = DataLoader(test_dataset,args.batch_size,num_workers=args.num_workers,pin_memory=args.pin_memory)
    print('Finished loading test dataset')

    test(args,chkpt,test_dataloader)
    print('Finished running inference on test dataset')