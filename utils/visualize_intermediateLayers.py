import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
sys.path.append('../CalibNet_pytorch')

from sklearn.decomposition import PCA

import time

from CalibNet import CalibNet_DINOV2, CalibNet_DINOV2_patch, CalibNet_DINOV2_LTC, CalibNet_DINOV2_patch_CalAgg, CalibNet

def get_self_attention(img: torch.Tensor, model: CalibNet_DINOV2, output_dir = None, imgName = 'attn-head'):
    # add 2 channels to depth image
    bt,c,hd,wd = img.size()
    if c ==1:
        depth_3 = torch.zeros((bt,3,hd,wd))

        depth_3[:, 0, :, :] = img[:, 0, :, :]
        depth_3[:, 1, :, :] = img[:, 0, :, :]
        depth_3[:, 2, :, :] = img[:, 0, :, :]
        img = depth_3

    image_size = (952, 952)
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    attentions = model.backbone.get_intermediate_layers(img.to(device),reshape=True)[0] 

    nh = attentions.shape[1] # number of head

    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = attentions.squeeze()
    attentions = attentions[::patch_size, :,:]  

    _, h,w = np.shape(attentions)
    att_acc = np.zeros((h,w))
    for i in range(attentions.shape[0]):
        att_acc+=attentions[i,:,:]
    
    if output_dir is not None:

        # save attentions heatmaps
        os.makedirs(output_dir, exist_ok=True)

        cumName = "a" + imgName + "_cum.png"
        fname = os.path.join(output_dir, cumName)
        plt.imsave(fname=fname, arr=att_acc, format='png')

        for j in range(nh):
            fname = os.path.join(output_dir, imgName + "_" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

    return attentions

def get_patchembeddings(img: torch.Tensor, model: CalibNet_DINOV2, output_dir = '', imgName = 'attn-head'):
    # add 2 channels to depth image
    bt,c,hd,wd = img.size()
    if c ==1:
        depth_3 = torch.zeros((bt,3,hd,wd))

        depth_3[:, 0, :, :] = img[:, 0, :, :]
        depth_3[:, 1, :, :] = img[:, 0, :, :]
        depth_3[:, 2, :, :] = img[:, 0, :, :]
        img = depth_3

    patch_size = 14
    hp = hd // patch_size
    wp = wd // patch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    attentions = model.backbone.forward_features(img.to(device))['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
    batch,embedd, wh = attentions.shape
    attentions = torch.reshape(attentions, (batch, embedd, hp, wp)).squeeze()

    return attentions

def pca_on_patches(atts, n_components = 12):
    n_patch, h, w = atts.shape
    patch_embeddings = atts.reshape(n_patch, h * w)

    mean = patch_embeddings.mean(axis=0)
    patch_embeddings -= mean

    
    pca = PCA(n_components=n_components)

    pca.fit(patch_embeddings.T)

    principal_components = pca.transform(patch_embeddings.T)
    principal_components = principal_components.T

    principal_components = principal_components + mean[:n_components].reshape(n_components,1)

    principal_components = principal_components.reshape(n_components,h,w)

    return principal_components

def plot_attention(img, attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()
    return

if __name__=="__main__":
    from dataset import BaseKITTIDataset, KITTI_perturb
    from torch.utils.data import DataLoader

    device = 'cuda'

    datasetpath =  '/home/colin/semesterThesis/CalibNet_pytorch/KITTI_Odometry_Full'
    seqs = ['02']

    dataset = BaseKITTIDataset(basedir=datasetpath,
                               batch_size=1,
                               seqs=seqs,
                               pcd_sample_num=10000,
                               resize_ratio=(1.0,1.0))
    dataset_perturb = KITTI_perturb(dataset=dataset,
                                    max_deg=0.0,
                                    max_tran=0.0)
    
    dataloader = DataLoader(dataset=dataset_perturb,batch_size=1)
    
    model_dino = CalibNet_DINOV2_patch_CalAgg().to(device=device)
    model_dino.device = device
    chkpt = torch.load('checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth', map_location=device)
    model_dino.load_state_dict(chkpt['model'])
    model_dino.eval()

    activation = {}
    def get_activation_backbone(name):
        def hook1(model, input, output):

            bt,c,hd,wd = input.size()

            hp = hd // 14
            wp = wd // 14

            rgb_patch_att_ = output['x_norm_patchtokens'].permute(0,2,1)
            batch,embedd, _ = rgb_patch_att_.shape
            rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))
            activation[name] = rgb_patch_att

        return hook1
    
    def get_activation_out(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    model_dino.rgb_features.register_forward_hook(get_activation_out('rgb_output'))
    model_dino.depth_features.register_forward_hook(get_activation_out('depth_output'))

    output_dir = './visualizations/attention_visualizations'
    plt.ion()

    for batch in dataloader:

        model_dino(batch['img'].to(device), batch['depth_img'].to(device))
        n = 9

        activations_rgb = activation['rgb_output'].squeeze().detach().cpu().numpy()
        activations_rgb = pca_on_patches(activations_rgb,n_components=n)

        activations_depth = activation['depth_output'].squeeze().detach().cpu().numpy()
        activations_depth = pca_on_patches(activations_depth,n_components=n)

        # activations_rgb = get_self_attention(batch['img'].to(device),model_dino)
        # activations_depth = get_self_attention(batch['depth_img'].to(device),model_dino)

        # Plot RGB features
        plot_attention(batch['img'].squeeze().permute(1,2,0).detach().cpu().numpy(),attention=activations_rgb[:n,:,:])
        # plot_attention(batch['img'].squeeze().permute(1,2,0).detach().cpu().numpy(),attention=activations_rgb_pca)

        # Plot Depth features 
        plot_attention(batch['depth_img'].squeeze().detach().cpu().numpy(),attention=activations_depth[:n,:,:])
        # plot_attention(batch['depth_img'].squeeze().detach().cpu().numpy(),attention=activations_depth_pca)

        break