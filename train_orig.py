import argparse
from asyncio.log import logger
import os
import yaml
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb
from mylogger import get_logger, print_highlight, print_warning
from CalibNet import CalibNet, CalibNet_DINOV2, CalibNet_DINOV2_patch, CalibNet_DINOV2_LTC, CalibNet_DINOV2_patch_CalAgg, CalibNet_DINOV2_patch_RGB_CalAgg, CalibNet_DINOV2_patch_RGB
import loss as loss_utils
import utils
from tqdm import tqdm
import numpy as np
from utils.transform import UniformTransformSE3

import matplotlib.pyplot as plt

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='KITTI_Odometry_Full/')
    parser.add_argument("--skip_frame",type=int,default=10,help='skip frame of dataset')
    parser.add_argument("--pcd_sample",type=int,default=4096)
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0.2)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    parser.add_argument("--randomCrop",type=float,default=1.0)
    parser.add_argument("--singlePerturbation", type=bool, default=True)

    # dataloader
    parser.add_argument("--batch_size",type=int,default=6) ##############################)
    parser.add_argument("--num_workers",type=int,default=16)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your CPU memory is insufficient')
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--resume",type=str,default='')
    parser.add_argument("--pretrained",type=str,default='') #checkpoint/cam2_oneiter_last.pth
    parser.add_argument("--epoch",type=int,default=50)
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--name",type=str,default='CalibNet_DINOV2_patch_fixed_perturbation')
    parser.add_argument("--optim",type=str,default='adam',choices=['sgd','adam'])
    parser.add_argument("--lr0",type=float,default=4e-4)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--weight_decay",type=float,default=5e-6)
    parser.add_argument("--lr_exp_decay",type=float,default=0.98)
    parser.add_argument("--clip_grad",type=float,default=2.0) 
    parser.add_argument("--finetune_tsl",type=bool,default=False)
    # setting
    parser.add_argument("--scale",type=float,default=50.0,help='scale factor of pcd normlization in loss')
    parser.add_argument("--inner_iter",type=int,default=1,help='inner iter of calibnet')
    parser.add_argument("--alpha",type=float,default=1.0,help='weight of photo loss')
    parser.add_argument("--beta",type=float,default=0.3,help='weight of chamfer loss')
    parser.add_argument("--gamma",type=float,default=300,help='weight of emd loss')
    parser.add_argument("--resize_ratio",type=float,nargs=2,default=[1.0,1.0])
    parser.add_argument("--model_name", type=str,default = 'CalibNet_DINOV2_patch', choices=['CalibNet', 'CalibNet_DINOV2', 'CalibNet_DINOV2_patch','CalibNet_DINOV2_patch_RGB', 'CalibNet_DINOV2_LTC','CalibNet_DINOV2_patch_RGB_CalAgg', 'CalibNet_DINOV2_patch_CalAgg'])
    parser.add_argument("--pcd_loss", type=str, default="chamfer", choices=['chamfer', 'emd'] )
    parser.add_argument("--depth_modality", type = str, default = 'depthimage', choices=['depthimage', 'rangeimage'])
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


@torch.no_grad()
def val(args,model,val_loader:DataLoader):
    model.eval()
    device = model.device
    tqdm_console = tqdm(total=len(val_loader),desc='Val')
    photo_loss = loss_utils.Photo_Loss(args.scale)
    if args.pcd_loss == 'chamfer':
        chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
    elif args.pcd_loss == 'emd':
        chamfer_loss = loss_utils.EarthMoverDistanceLoss(args.scale, 'mean')
    alpha = float(args.alpha)
    beta = float(args.beta)
    total_dR = 0
    total_dT = 0
    total_loss = 0
    total_se3_loss = 0
    with tqdm_console:
        tqdm_console.set_description_str('Val')
        for batch in val_loader:
            rgb_img = batch['img'].to(device)
            B = rgb_img.size(0)
            pcd_range = batch['pcd_range'].to(device)
            calibed_depth_img = batch['depth_img'].to(device)
            calibed_pcd = batch['pcd'].to(device)
            uncalibed_pcd = batch['uncalibed_pcd'].to(device)
            uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
            InTran = batch['InTran'][0].to(device)
            igt = batch['igt'].to(device)
            img_shape = rgb_img.shape[-2:]
            depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
            # model(rgb_img,uncalibed_depth_img)
            g0 = torch.eye(4).repeat(B,1,1).to(device)
            # for _ in range(args.inner_iter):
            #     twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
            #     extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))
            #     uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
            #     g0 = extran.bmm(g0)

            for _ in range(args.inner_iter):
                    twist_rot, twist_tsl = model(rgb_img, uncalibed_depth_img)
                    extran = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))

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

                    g0 = extran.bmm(g0)

                    # Detach unnecessary tensors
                    # twist_rot.detach_()
                    # twist_tsl.detach_()
                    # extran.detach_()


            err_g = g0.bmm(igt)
            dR,dT = loss_utils.geodesic_distance(err_g)
            total_dR += dR.item()
            total_dT += dT.item()
            se3_loss = torch.linalg.norm(utils.se3.log(err_g),dim=1).mean()/6
            total_se3_loss += se3_loss.item()
            loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
            loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
            loss = alpha*loss1 + beta*loss2
            total_loss += loss.item()
            tqdm_console.set_postfix_str('dR:{:.4f}, dT:{:.4f},se3_loss:{:.4f}'.format(dR,dT,se3_loss))
            tqdm_console.update(1)
    total_dR /= len(val_loader)
    total_dT /= len(val_loader)
    total_loss /= len(val_loader)
    total_se3_loss /= len(val_loader)
    return total_loss, total_dR, total_dT, total_se3_loss


def train(args,chkpt,train_loader:DataLoader,val_loader:DataLoader):
    device = torch.device(args.device)
    # model = CalibNet(backbone_pretrained=False,depth_scale=args.scale)
    if args.model_name == 'CalibNet':
        model = CalibNet(depth_scale=args.scale)
    if args.model_name == 'CalibNet_DINOV2':
        model = CalibNet_DINOV2(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False
    if args.model_name == 'CalibNet_DINOV2_patch':
        model = CalibNet_DINOV2_patch(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False
    if args.model_name == 'CalibNet_DINOV2_LTC':
        model = CalibNet_DINOV2_LTC(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False
    if args.model_name == 'CalibNet_DINOV2_patch_CalAgg':
        model = CalibNet_DINOV2_patch_CalAgg(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False
    if args.model_name == 'CalibNet_DINOV2_patch_RGB':
        model = CalibNet_DINOV2_patch_RGB(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False
    if args.model_name == 'CalibNet_DINOV2_patch_RGB_CalAgg':
        model = CalibNet_DINOV2_patch_RGB_CalAgg(depth_scale=args.scale)
        for params in model.backbone.parameters():
            params.requires_grad = False

    # if args.finetune_tsl:
    #     # Freeze all weights
    #     for param in model.parameters():
    #         param.requires_grad = False

    #     # Unfreeze translation layers
    #     for param in model.aggregation.fc1.parameters():
    #         param.requires_grad = True
    #     for param in model.aggregation.tr_conv.parameters():
    #         param.requires_grad = True


    model.to(device)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),args.lr0,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),args.lr0,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_exp_decay)
    if args.pretrained:
        if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained)['model'])
            print_highlight('Pretrained model loaded from {:s}'.format(args.pretrained))
        else:
            print_warning('Invalid pretrained path: {:s}'.format(args.pretrained))
    if chkpt is not None:
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        scheduler.load_state_dict(chkpt['scheduler'])
        start_epoch = chkpt['epoch'] + 1
        min_loss = chkpt['min_loss']
        log_mode = 'a'
    else:
        start_epoch = 0
        min_loss = float('inf')
        log_mode = 'w'
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    log_mode = 'a' if chkpt is not None else 'w'
    logger = get_logger("{name}|Train".format(name=args.name),os.path.join(args.log_dir,args.name+'.log'),mode=log_mode)
    if chkpt is None:
        logger.debug(args)
        print_highlight('Start Training')
    else:
        print_highlight('Resume from epoch {:d}'.format(start_epoch+1))
    del chkpt  # free memory
    photo_loss = loss_utils.Photo_Loss(args.scale)
    if args.pcd_loss == 'chamfer':
        chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
        beta = float(args.beta)
    elif args.pcd_loss == 'emd':
        chamfer_loss = loss_utils.EarthMoverDistanceLoss(args.scale, 'sum')
        beta = float(args.gamma)
    regression_loss = loss_utils.Geodesic_Regression_Loss()
    alpha = float(args.alpha)
    
    for epoch in range(start_epoch,args.epoch):
        # model.train()
        tqdm_console = tqdm(total=len(train_loader),desc='Train')
        total_photo_loss = 0
        total_chamfer_loss = 0
        with tqdm_console:
            tqdm_console.set_description_str('Epoch: {:03d}|{:03d}'.format(epoch+1,args.epoch))
            model.train()
            if args.model_name == 'CalibNet_DINOV2' or args.model_name == 'CalibNet_DINOV2_patch' or args.model_name == 'CalibNet_DINOV2_LTC':
                model.backbone.eval()
            for batch in train_loader:
                optimizer.zero_grad()
                rgb_img = batch['img'].to(device)
                B = rgb_img.size(0)
                pcd_range = batch['pcd_range'].to(device)
                calibed_depth_img = batch['depth_img'].to(device)
                calibed_pcd = batch['pcd'].to(device)
                uncalibed_pcd = batch['uncalibed_pcd'].to(device)
                uncalibed_pcd.requires_grad = True
                uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
                InTran = batch['InTran'][0].to(device)
                igt = batch['igt'].to(device)
                img_shape = rgb_img.shape[-2:]
                depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
                # model(rgb_img,uncalibed_depth_img)
                g0 = torch.eye(4).repeat(B,1,1).to(device)
                # model.eval()
                for _ in range(args.inner_iter):
                    twist_rot, twist_tsl = model(rgb_img, uncalibed_depth_img)
                    extran = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))

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

                    g0 = extran.bmm(g0)

                    # Detach unnecessary tensors
                    # twist_rot.detach_()
                    # twist_tsl.detach_()
                    # extran.detach_()

                    
                dR, dT = loss_utils.geodesic_distance(g0.bmm(igt))


                

                calibed_depth_img.requires_grad = True
                uncalibed_depth_img.requires_grad = True
                calibed_pcd.requires_grad = True
                loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
                loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
                loss3, loss4 = regression_loss(g0.bmm(igt))
                loss = alpha*loss1*finetune + beta*loss2 #+ loss3 + loss4*10
                # loss.backward(retain_graph=True)
                nn.utils.clip_grad_value_(model.parameters(),args.clip_grad)
                loss.backward()

                # twist_gt = utils.se3.log(utils.se3.inverse(igt))
                # twist_rot_gt = twist_gt[:,:3]
                # twist_tsl_gt = twist_gt[:,3:]
                # loss_rot_reg = (abs(twist_rot_gt - twist_rot)).mean(dim=0)
                # loss_tsl_reg = (abs(twist_tsl_gt - twist_tsl)).mean(dim=0)
                # rot_mask = torch.ones_like(loss_rot_reg)
                # tsl_mask = torch.ones_like(loss_tsl_reg)
                # loss_rot_reg.backward(rot_mask, retain_graph=True)
                # loss_tsl_reg.backward(tsl_mask)

                optimizer.step()

                tqdm_console.set_postfix_str("p:{:.3f}, c:{:.3f}, dR:{:.3f}, dT:{:.3f}".format(loss1.item()*alpha,loss2.item()*beta,dR.item(),dT.item()))
                tqdm_console.update()
                total_photo_loss += loss1.item()
                total_chamfer_loss += loss2.item()
        N_loader = len(train_loader)
        total_photo_loss /= N_loader
        total_chamfer_loss /= N_loader
        total_loss = alpha*total_photo_loss + beta*total_chamfer_loss
        tqdm_console.set_postfix_str("loss: {:.3f}, photo: {:.3f}, chamfer: {:.3f}".format(total_loss,total_photo_loss,total_chamfer_loss))
        tqdm_console.update()
        tqdm_console.close()
        logger.info('Epoch {:03d}|{:03d}, train loss:{:.4f}'.format(epoch+1,args.epoch,total_loss))
        scheduler.step()
        val_loss, loss_dR, loss_dT, loss_se3 = val(args,model,val_loader)  # float 
        if loss_se3 < min_loss:
            min_loss = loss_se3
            torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_best.pth'.format(name=args.name)))
            logger.debug('Best model saved (Epoch {:d})'.format(epoch+1))
            print_highlight('Best Model (Epoch %d)'%(epoch+1))
            print_highlight('Validation Loss: ' + str(min_loss))
        torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_last.pth'.format(name=args.name)))
        logger.info('Evaluate loss_dR:{:.6f}, loss_dT:{:.6f}, se3_loss:{:.6f}'.format(loss_dR,loss_dT,loss_se3))
            
            
            

if __name__ == "__main__":
    args = options()
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    with open(args.config,'r')as f:
        CONFIG = yaml.load(f,yaml.SafeLoader)
    assert isinstance(CONFIG,dict), 'Unknown config format!'
    if args.resume:
        chkpt = torch.load(args.resume,map_location='cpu')
        CONFIG.update(chkpt['config'])
        args.__dict__.update(chkpt['args'])
        print_highlight('config updated from resumed checkpoint {:s}'.format(args.resume))
    else:
        chkpt = None
    print_highlight('args have been received, please wait for dataloader...')
    train_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['train']]
    val_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['val']]

    # Load dataset config
    with open('dataset_paths.yml','r')as f:
        DATA : dict = yaml.load(f,yaml.SafeLoader)
    print_highlight('args have been received, please wait for dataloader...')

    if args.singlePerturbation == True:
        single_perturb_file = os.path.join(args.checkpoint_dir,"single_pert.csv")
        if not os.path.exists(single_perturb_file):
            print_highlight("Single pertub file dosen't exist, create one.")
            transform = UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
            perturb_arr = np.zeros([1,6])
            perturb_arr[0,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(single_perturb_file,perturb_arr,delimiter=',')
        else:  # check length
            val_seq = np.loadtxt(single_perturb_file,delimiter=',')
            if 6 != val_seq.shape[0]:
                print_warning('Incompatiable validation length {}!={}'.format(1,val_seq.shape[0]))
                transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
                perturb_arr = np.zeros([1,6])
                perturb_arr[0,:] = transform.generate_transform().cpu().numpy()
                np.savetxt(single_perturb_file,perturb_arr,delimiter=',')
                print_highlight('Validation perturb file rewritten.')

        trainFile = single_perturb_file
        val_perturb_file = single_perturb_file
    else:
        trainFile = None

    # dataset
    train_dataset = BaseKITTIDataset(basedir=DATA['kitti_full'],
                                    batch_size=args.batch_size,
                                    seqs=train_split,
                                    cam_id=CONFIG['dataset']['cam_id'],
                                    meta_json='data_len.json',
                                    skip_frame=args.skip_frame,
                                    voxel_size=CONFIG['dataset']['voxel_size'],
                                    pcd_sample_num=args.pcd_sample,
                                    resize_ratio=args.resize_ratio,
                                    extend_intran=CONFIG['dataset']['extend_ratio'],
                                    randomCrop=args.randomCrop)
    
    train_dataset = KITTI_perturb(dataset=train_dataset,
                                  max_deg=args.max_deg,
                                  max_tran=args.max_tran,
                                  mag_randomly=args.mag_randomly,
                                  pooling_size=CONFIG['dataset']['pooling'],
                                  file=trainFile,
                                  singlePerturbation=args.singlePerturbation)
    
    # train_dataset = torch.utils.data.Subset(train_dataset, [0,1])
    
    val_dataset = BaseKITTIDataset(basedir=DATA['kitti_full'],
                                   batch_size=args.batch_size,
                                   seqs=val_split,
                                   cam_id=CONFIG['dataset']['cam_id'],
                                   meta_json='data_len.json',
                                   skip_frame=args.skip_frame,
                                   voxel_size=CONFIG['dataset']['voxel_size'],
                                   pcd_sample_num=args.pcd_sample,
                                   resize_ratio=args.resize_ratio,
                                   extend_intran=CONFIG['dataset']['extend_ratio'],
                                   randomCrop=args.randomCrop)
    
    if args.singlePerturbation == False:
        val_perturb_file = os.path.join(args.checkpoint_dir,"val_seq.csv")
        val_length = len(val_dataset)
        if not os.path.exists(val_perturb_file):
            print_highlight("validation pertub file dosen't exist, create one.")
            transform = UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
            perturb_arr = np.zeros([val_length,6])
            for i in range(val_length):
                perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
        else:  # check length
            val_seq = np.loadtxt(val_perturb_file,delimiter=',')
            if val_length != val_seq.shape[0]:
                print_warning('Incompatiable validation length {}!={}'.format(val_length,val_seq.shape[0]))
                transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
                perturb_arr = np.zeros([val_length,6])
                for i in range(val_length):
                    perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
                np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
                print_highlight('Validation perturb file rewritten.')

    val_dataset = KITTI_perturb(val_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=CONFIG['dataset']['pooling'],
                                file=val_perturb_file, singlePerturbation=args.singlePerturbation)
    # batch normlization does not support batch=1
    train_drop_last = True if len(train_dataset) % args.batch_size == 1 else False  
    val_drop_last = True if len(val_dataset) % args.batch_size == 1 else False
    # dataloader
    train_dataloader = DataLoader(train_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_memory,drop_last=train_drop_last)
    val_dataloder = DataLoader(val_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_memory,drop_last=val_drop_last)
    
        
    train(args,chkpt,train_dataloader,val_dataloder)