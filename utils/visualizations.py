import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    dist_tran_abs = np.linalg.norm(dist_tran, axis=1)

    dist_rot = tf_mat[:,0:3]
    dist_rot_abs = np.linalg.norm(np.degrees(dist_rot),axis=1)

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    # Plot the distribution of translations
    ax1.hist(dist_tran_abs, bins=50, density=True, alpha=0.7)
    ax1.set_xlabel('Translational Perturbation in [m]')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Distribution of Translations')

    # Add the average line for translations
    ax1.axvline(dist_tran_abs.mean(), color='red', linestyle='dashed', linewidth=1)
    ax1.legend(['Average'])

    # Plot the distribution of rotations
    ax2.hist(dist_rot_abs, bins=50, density=True, alpha=0.7)
    ax2.set_xlabel('Rotational Perturbation in [Deg]')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Distribution of Rotations')

    # Add the average line for rotations
    ax2.axvline(dist_rot_abs.mean(), color='red', linestyle='dashed', linewidth=1)
    ax2.legend(['Average'])

    plt.tight_layout()  # Adjust the spacing between subplots

    if fileName is not None:
        plt.savefig(fileName, dpi = 800)

    return

def plotTestStats(igt: np.ndarray, recalib: np.ndarray, rgb_embedd: np.ndarray, depth_embedd: np.ndarray):
    samples,_,_ = np.shape(igt)
    i = 0

    fig = make_subplots(rows=2,cols=1,
                        specs=[[{"type": "scatter3d"}           ],
                               [{"type": "scatter"}]],
                               subplot_titles=('Model Predicted vs Ground Truth Coordinate Transform',
                                    'DINOV2 Embeddings'))

    samples = samples // 10
    for i in range(samples):   
    
        # Plot Affine Transformation
        # Origin
        coordinate_origin_x = np.array([[0,0,0],[0.1,0,0]])
        coordinate_origin_y = np.array([[0,0,0],[0,0.1,0]])
        coordinate_origin_z = np.array([[0,0,0],[0,0,0.1]])

        # Ground Truth 
        igt_Rot = np.linalg.inv(igt[i,:3,:3])
        igt_t = -igt[i,:3,3]*0
        orig_frame1 = np.eye(3) / 10
        rotd_frame = np.matmul(igt_Rot,orig_frame1)
        t_rotd_frame = rotd_frame + igt_t.T
        coordinate_origin_x_tf = np.array([igt_t.T,t_rotd_frame[:,0] ])
        coordinate_origin_y_tf = np.array([igt_t.T,t_rotd_frame[:,1] ])
        coordinate_origin_z_tf = np.array([igt_t.T,t_rotd_frame[:,2] ])

        # Model Output
        recalib_Rot = recalib[i,:3,:3]
        recalib_t = recalib[i,:3,3]*0
        orig_frame2 = np.eye(3) / 10
        rotd_frame_m = np.matmul(recalib_Rot,orig_frame2)
        t_rotd_frame_m = rotd_frame_m + recalib_t.T
        coordinate_origin_x_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,0] ])
        coordinate_origin_y_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,1] ])
        coordinate_origin_z_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,2] ])
 
        # Define Scatter TF
        origin_frame_x = go.Scatter3d(x=coordinate_origin_x[:,0],y=coordinate_origin_x[:,1],z=coordinate_origin_x[:,2],mode='lines',line=dict(color="#FF0000"), name="X Origin " + str(i))
        origin_frame_y = go.Scatter3d(x=coordinate_origin_y[:,0],y=coordinate_origin_y[:,1],z=coordinate_origin_y[:,2],mode='lines',line=dict(color="#00FF00"), name="Y Origin " + str(i))
        origin_frame_z = go.Scatter3d(x=coordinate_origin_z[:,0],y=coordinate_origin_z[:,1],z=coordinate_origin_z[:,2],mode='lines',line=dict(color="#0000FF"), name="Z Origin " + str(i))
        gt_frame_x = go.Scatter3d(x=coordinate_origin_x_tf[:,0],y=coordinate_origin_x_tf[:,1],z=coordinate_origin_x_tf[:,2],mode='lines',line=dict(color="#FF0000", width=6, dash = 'dot'), name="X GT " + str(i))
        gt_frame_y = go.Scatter3d(x=coordinate_origin_y_tf[:,0],y=coordinate_origin_y_tf[:,1],z=coordinate_origin_y_tf[:,2],mode='lines',line=dict(color="#00FF00", width=6, dash = 'dot'), name="Y GT " + str(i))
        gt_frame_z = go.Scatter3d(x=coordinate_origin_z_tf[:,0],y=coordinate_origin_z_tf[:,1],z=coordinate_origin_z_tf[:,2],mode='lines',line=dict(color="#0000FF", width=6, dash = 'dot'), name="Z GT " + str(i))
        md_frame_x = go.Scatter3d(x=coordinate_origin_x_tf_m[:,0],y=coordinate_origin_x_tf_m[:,1],z=coordinate_origin_x_tf_m[:,2],mode='lines',line=dict(color="#FF0000", width=6, dash = 'dash'), name="X Model " + str(i))
        md_frame_y = go.Scatter3d(x=coordinate_origin_y_tf_m[:,0],y=coordinate_origin_y_tf_m[:,1],z=coordinate_origin_y_tf_m[:,2],mode='lines',line=dict(color="#00FF00", width=6, dash = 'dash'), name="Y Model " + str(i))
        md_frame_z = go.Scatter3d(x=coordinate_origin_z_tf_m[:,0],y=coordinate_origin_z_tf_m[:,1],z=coordinate_origin_z_tf_m[:,2],mode='lines',line=dict(color="#0000FF", width=6, dash = 'dash'), name="Z Model " + str(i))

        # Define Scatter Embeddings 
        rgb_embeddings_scatter = go.Scatter(x=np.arange(len(rgb_embedd[i,:])), y=rgb_embedd[i,:], name="RGB Image Embeddings " + str(i))
        dep_embeddings_scatter = go.Scatter(x=np.arange(len(depth_embedd[i,:])), y=depth_embedd[i,:], name="Depth Image Embeddings " + str(i))

        # Embedding Traces
        fig.add_trace(
                    rgb_embeddings_scatter,
                    row=2, col=1
                    )
        fig.add_trace(
                    dep_embeddings_scatter,
                    row=2, col=1
                    )

        # GT Coordinates Frame Traces 
        fig.add_trace(
                    origin_frame_x,
                    row=1,col=1
                    )
        fig.add_trace(
                    origin_frame_y,
                    row=1,col=1
                    )
        fig.add_trace(
                    origin_frame_z,
                    row=1,col=1
                    )
        fig.add_trace(
                    md_frame_x,
                    row=1,col=1
                    )
        fig.add_trace(
                    md_frame_y,
                    row=1,col=1
                    )
        fig.add_trace(
                    md_frame_z,
                    row=1,col=1
                    )
        
        # Model Output Coordinates Frame Traces
        fig.add_trace(
                    origin_frame_x,
                    row=1,col=1
                    )
        fig.add_trace(
                    origin_frame_y,
                    row=1,col=1
                    )
        fig.add_trace(
                    origin_frame_z,
                    row=1,col=1
                    )
        fig.add_trace(
                    gt_frame_x,
                    row=1,col=1
                    )
        fig.add_trace(
                    gt_frame_y,
                    row=1,col=1
                    )
        fig.add_trace(
                    gt_frame_z,
                    row=1,col=1
                    ) 
            
        
        fig.update_layout(scene=dict(aspectmode="cube"))
        fig.update_layout(scene=dict(
        xaxis=dict(range=[-0.2, 0.2]),
        yaxis=dict(range=[-0.2, 0.2]),
        zaxis=dict(range=[-0.2, 0.2]),
        ))
        # fig.update_layout(scene2=dict(aspectmode="cube"))
        # fig.update_layout(scene2=dict(
        # xaxis=dict(range=[-0.2, 0.2]),
        # yaxis=dict(range=[-0.2, 0.2]),
        # zaxis=dict(range=[-0.2, 0.2]),
        # ))

    steps = []
    for s in range(samples):

        # # Ground Truth 
        # igt_Rot = igt[s,:3,:3]
        # igt_t = igt[s,:3,3]
        # orig_frame1 = np.eye(3) / 10
        # rotd_frame = np.matmul(igt_Rot,orig_frame1)
        # t_rotd_frame = rotd_frame + igt_t
        # coordinate_origin_x_tf = np.array([igt_t.T,t_rotd_frame[:,0] ])
        # coordinate_origin_y_tf = np.array([igt_t.T,t_rotd_frame[:,1] ])
        # coordinate_origin_z_tf = np.array([igt_t.T,t_rotd_frame[:,2] ])

        # # Model Output
        # recalib_Rot = recalib[s,:3,:3]
        # recalib_t = recalib[s,:3,3]
        # orig_frame2 = np.eye(3) / 10
        # rotd_frame_m = np.matmul(recalib_Rot,orig_frame2)
        # t_rotd_frame_m = rotd_frame_m + recalib_t
        # coordinate_origin_x_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,0] ])
        # coordinate_origin_y_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,1] ])
        # coordinate_origin_z_tf_m = np.array([recalib_t.T,t_rotd_frame_m[:,2] ])

        # # # Define Scatter TF
        # # origin_frame_x = go.Scatter3d(x=coordinate_origin_x[:,0],y=coordinate_origin_x[:,1],z=coordinate_origin_x[:,2],mode='lines',line=dict(color="#FF0000"), name="X Origin")
        # # origin_frame_y = go.Scatter3d(x=coordinate_origin_y[:,0],y=coordinate_origin_y[:,1],z=coordinate_origin_y[:,2],mode='lines',line=dict(color="#00FF00"), name="Y Origin")
        # # origin_frame_z = go.Scatter3d(x=coordinate_origin_z[:,0],y=coordinate_origin_z[:,1],z=coordinate_origin_z[:,2],mode='lines',line=dict(color="#0000FF"), name="Z Origin")
        # # gt_frame_x = go.Scatter3d(x=coordinate_origin_x_tf_m[:,0],y=coordinate_origin_x_tf_m[:,1],z=coordinate_origin_x_tf_m[:,2],mode='lines',line=dict(color="#FF0000"), name="X GT")
        # # gt_frame_y = go.Scatter3d(x=coordinate_origin_y_tf_m[:,0],y=coordinate_origin_y_tf_m[:,1],z=coordinate_origin_y_tf_m[:,2],mode='lines',line=dict(color="#00FF00"), name="Y GT")
        # # gt_frame_z = go.Scatter3d(x=coordinate_origin_z_tf_m[:,0],y=coordinate_origin_z_tf_m[:,1],z=coordinate_origin_z_tf_m[:,2],mode='lines',line=dict(color="#0000FF"), name="Z GT")
        # # md_frame_x = go.Scatter3d(x=coordinate_origin_x_tf[:,0],y=coordinate_origin_x_tf[:,1],z=coordinate_origin_x_tf[:,2],mode='lines',line=dict(color="#FF0000"), name="X Model")
        # # md_frame_y = go.Scatter3d(x=coordinate_origin_y_tf[:,0],y=coordinate_origin_y_tf[:,1],z=coordinate_origin_y_tf[:,2],mode='lines',line=dict(color="#00FF00"), name="Y Model")
        # # md_frame_z = go.Scatter3d(x=coordinate_origin_z_tf[:,0],y=coordinate_origin_z_tf[:,1],z=coordinate_origin_z_tf[:,2],mode='lines',line=dict(color="#0000FF"), name="Z Model")

        # # # Define Scatter Embeddings 
        # # rgb_embeddings_scatter = go.Scatter(x=np.arange(len(rgb_embedd[i,:])), y=rgb_embedd[i,:], name="RGB Image Embeddings")
        # # dep_embeddings_scatter = go.Scatter(x=np.arange(len(depth_embedd[i,:])), y=depth_embedd[i,:], name="Depth Image Embeddings")

        # fig.data[0].x = np.arange(len(rgb_embedd[s,:]))
        # fig.data[0].y = rgb_embedd[s,:]
        # fig.data[1].x = np.arange(len(depth_embedd[s,:]))
        # fig.data[1].y = depth_embedd[s,:]

        visible = [False] * (14 * samples)  # Initialize visibility list for all traces
        visible[(s * 14):((s + 1) * 14)] = [True] * 14  # Set visibility for the traces of the selected sample

        step = dict(
            method="update",
            args=[{"visible": visible}],  # Update visibility
            label=f"Sample {s}"
        )

        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Sample: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)

    fig.write_html('results_plotly.html')

    fig.show()  

    return

if __name__ == "__main__":

    dep_dinov2 = np.load('res/dep_dinov2.npy')
    igt_npy = np.load('res/igt_npy.npy')
    recalib_npy = np.load('res/recalib_npy.npy')
    res_npy = np.load('res/res_npy.npy')
    rgb_dinov2 = np.load('res/rgb_dinov2.npy')

    plotTestStats(igt=igt_npy,
                  recalib=recalib_npy,
                  rgb_embedd=rgb_dinov2,
                  depth_embedd=dep_dinov2)