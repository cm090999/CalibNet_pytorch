import matplotlib.pyplot as plt
import numpy as np
import os


def vis_eval(pcd_gt, pcd_miscalib, pcd_corrected, rgb, intran, savePath = None,saveName = None, igt = None, networkOutput = None):

    # Get image shape
    H,W = rgb.shape[:2]
    
    # Modify point Cloud, get locations 
    u_gt,v_gt,w_gt,r_gt = get_projectedPoints(pcd_gt,rgb=rgb, intran=intran)
    u_miscalib,v_miscalib,w_miscalib,r_miscalib = get_projectedPoints(pcd_miscalib,rgb=rgb, intran=intran)
    u_corrected,v_corrected,w_corrected,r_corrected = get_projectedPoints(pcd_corrected,rgb=rgb, intran=intran)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, tight_layout=True)

    # Plot 1
    axes[0].axis([W, 0, H, 0])
    axes[0].imshow(rgb)
    axes[0].scatter([u_gt], [v_gt], c=[r_gt], cmap='rainbow_r', alpha=0.5, s=1)
    axes[0].set_title('Ground Truth Calibration')

    # Plot 2
    axes[1].axis([W, 0, H, 0])
    axes[1].imshow(rgb)
    axes[1].scatter([u_miscalib], [v_miscalib], c=[r_miscalib], cmap='rainbow_r', alpha=0.5, s=1)
    axes[1].set_title('Miscalibrated Point Cloud')

    # Plot 3
    axes[2].axis([W, 0, H, 0])
    axes[2].imshow(rgb)
    axes[2].scatter([u_corrected], [v_corrected], c=[r_corrected], cmap='rainbow_r', alpha=0.5, s=1)
    axes[2].set_title('Re-Calibration by network') 

    # if igt is not None:
    #     # Add statistics
    #     gt = np.zeros_like(igt)
    #     gt[:3,3] = -igt[:3,3]
    #     gt[:3,:3] = np.linalg.inv(igt[:3,:3])
    #     gt[3,3] = 1
    #     statistics = "Ground Truth Calibration: \n" + str(gt) + '\n' + 'Network Output Calibration: \n' + str(networkOutput)  # Replace with your actual statistics
    #     plt.text(0, -0.1, statistics, transform=fig.transFigure, fontsize=6)

    # Save the figure
    if savePath:
        save_path = os.path.join(savePath, saveName)
        plt.savefig(save_path,bbox_inches='tight', dpi=500)

    plt.clf()
    plt.close()

    return

def get_projectedPoints(pcd,rgb,intran):
    # Get image shape
    H,W = rgb.shape[:2]

    pcd = intran @ pcd[:3,:]

    u,v,w = pcd[0,:], pcd[1,:], pcd[2,:]
    u = u/w
    v = v/w
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = np.linalg.norm(pcd[:,rev],axis=0)

    return u,v,w,r

def printStatistics(tf_mat, fileName = None):
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

    if fileName is not None:
        plt.savefig(fileName, dpi = 500)

    return