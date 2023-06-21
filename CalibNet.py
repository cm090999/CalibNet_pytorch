import torch
from torch import nn
import torch.nn.functional as F
from Modules import resnet18
import numpy as np
from ncps.torch import LTC
from ncps.wirings import AutoNCP

from dinov2.models.vision_transformer import vit_small

class Aggregation(nn.Module):
    def __init__(self,inplanes=768,planes=96,final_feat=(5,2)):
        super(Aggregation,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(in_channels=planes*4,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(in_channels=planes*4,out_channels=planes*2,kernel_size=(2,1),stride=2)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.tr_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)
        self.tr_drop = nn.Dropout2d(p=0.2)
        self.rot_drop = nn.Dropout2d(p=0.2)
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.fc1 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
        self.fc2 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.xavier_normal_(self.fc1.weight,0.1)
        nn.init.xavier_normal_(self.fc2.weight,0.1)

    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_tr = self.tr_conv(x)
        x_tr = self.tr_bn(x_tr)
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))
        x_rot = self.rot_conv(x)
        x_rot = self.rot_bn(x_rot)
        x_rot = self.rot_drop(x_rot)  
        x_rot = self.rot_pool(x_rot)  # (19.6)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        return x_rot, x_tr

class CalibNet(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet,self).__init__()
        self.scale = depth_scale
        self.rgb_resnet = resnet18(inplanes=3,planes=64)  # outplanes = 512
        self.depth_resnet = nn.Sequential(
            nn.MaxPool2d(kernel_size=5,stride=1,padding=2),  # outplanes = 256
            resnet18(inplanes=1,planes=32),
        )

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.aggregation = Aggregation(inplanes=512+256,planes=96)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if backbone_pretrained:
            self.rgb_resnet.load_state_dict(torch.load("resnetV1C.pth")['state_dict'],strict=False)
        self.to(self.device)
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]
        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale
        x1 = self.rgb_resnet(x1)[-1]
        x2 = self.depth_resnet(x2)[-1]
        
        x1 = self.rgb_features(x1)
        x2 = self.depth_features(x2)

        feat = torch.cat((x1,x2),dim=1)  # [B,C1+C2,H,W]
        x_rot, x_tr =  self.aggregation(feat)
        return x_rot, x_tr
    
class CalibNet_DINOV2(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2,self).__init__()
        self.scale = depth_scale

        self.backbone = vit_small(patch_size=14,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        # self.backbone.to(device)
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)
        
        # aggregate features
        self.fc1 = nn.Linear(768,512)
        self.bn1 = nn.LayerNorm(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.LayerNorm(256)

        # Tranlation
        self.fc_t1 = nn.Linear(256,128)
        self.bn_t1 = nn.LayerNorm(128)
        self.relu_t1 = nn.ReLU()
        self.drop_t1 = nn.Dropout(p=0.2)
        self.fc_t2 = nn.Linear(128,3)

        # Rotation
        self.fc_r1 = nn.Linear(256,128)
        self.bn_r1 = nn.LayerNorm(128)
        self.relu_r1 = nn.ReLU()
        self.drop_r1 = nn.Dropout(p=0.2)
        self.fc_r2 = nn.Linear(128,3)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]

        # add 2 channels to depth image
        bt,c,hd,wd = depth.size()
        if c ==1:
            depth_3 = torch.zeros((bt,3,hd,wd))

        depth_3[:, 0, :, :] = depth[:, 0, :, :]
        depth_3[:, 1, :, :] = depth[:, 0, :, :]
        depth_3[:, 2, :, :] = depth[:, 0, :, :]

        # resize input
        x1,x2 = rgb,depth_3.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        feat = torch.cat((x1,x2),dim=1)  # [B,C1+C2,H,W]

        # Aggregate
        x = self.fc1(feat)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)

        # translation
        x_tr = self.fc_t1(x)
        x_tr = self.bn_t1(x_tr)
        x_tr = self.relu_t1(x_tr)
        x_tr = self.drop_t1(x_tr)
        x_tr = self.fc_t2(x_tr)

        # Rotation
        x_rot = self.fc_r1(x)
        x_rot = self.bn_r1(x_rot)
        x_rot = self.relu_r1(x_rot)
        x_rot = self.drop_r1(x_rot)
        x_rot = self.fc_r2(x_rot)

        return x_rot, x_tr

class Aggregate_Patches(nn.Module):
    def __init__(self,inrgb=384,indepth=192) -> None:
        super(Aggregate_Patches, self).__init__()
        self.conv0rgb = nn.Conv2d(in_channels=inrgb, out_channels=256, kernel_size=3, padding=1)
        self.conv1rgb = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1)
        self.conv2rgb = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)

        self.conv0depth = nn.Conv2d(in_channels=indepth, out_channels=168, kernel_size=3, padding=1)
        self.conv1depth = nn.Conv2d(in_channels=168, out_channels=156, kernel_size=3, padding=1)
        self.conv2depth = nn.Conv2d(in_channels=156, out_channels=128, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(2*384, 256)

        self.fc1_t = nn.Linear(1344, 512)
        self.layerNorm_tr1 = nn.LayerNorm(512)
        self.dropout1t = nn.Dropout(p=0.2)
        self.fc2_t = nn.Linear(512,128)
        self.fc3_t = nn.Linear(128, 3)

        self.fc1_r = nn.Linear(1344, 512)
        self.layerNorm_rot1 = nn.LayerNorm(512)
        self.dropout1r = nn.Dropout(p=0.2)
        self.fc2_r = nn.Linear(512,128)
        self.fc3_r = nn.Linear(128, 3)

        self.relu = nn.ReLU()

        
    
    def forward(self, rgb_patches, depth_patches):
        bt = rgb_patches.shape[0]
        x1 = self.conv0rgb(rgb_patches)
        x1 = self.relu(x1)
        x1 = self.conv1rgb(x1)
        x1 = self.relu(x1)
        x1 = self.conv2rgb(x1)
        x1 = self.relu(x1)

        x2 = self.conv0depth(depth_patches)
        x2 = self.relu(x2)
        x2 = self.conv1depth(x2)
        x1 = self.relu(x1)
        x2 = self.conv2depth(x2)
        x1 = self.relu(x1)

        agg_modalities = torch.cat((x1, x2), dim=1)
        x = self.conv1(agg_modalities)
        
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = spatial_pyramid_pool(x,bt,[int(x.size(2)),int(x.size(3))],[4,2,1])

        tr1 = self.fc1_t(x)
        tr1 = self.dropout1t(tr1)
        tr1 = self.fc2_t(tr1)
        tr = self.fc3_t(tr1)

        rot1 = self.fc1_r(x)
        rot1 = self.dropout1r(rot1)
        rot1 = self.fc2_r(rot1)
        rot = self.fc3_r(rot1)

        return rot, tr
    
class CalibNet_DINOV2_patch(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2_patch,self).__init__()
        self.scale = depth_scale
        self.patch_size = 14
 
        self.backbone = vit_small(patch_size=self.patch_size,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)

        self.depth_pretrained = FeatureExtractDepth()
        pretrained_dict = torch.load('pretrained_depth/best_model.pth.tar')['state_dict'] 
        filtered_dict = {k: v for k, v in pretrained_dict.items() if 'FeatureExtractDepth.' in k}
        filtered_dict = {k.replace('FeatureExtractDepth.', ''): v for k, v in filtered_dict.items() if 'FeatureExtractDepth.' in k}
        model_dict = self.depth_pretrained.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(k in 'FeatureExtractDepth.'+i for i in model_dict)} #'FeatureExtractorDepth.'+model_dict 
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.depth_pretrained.load_state_dict(filtered_dict)

        self.depth_aptmaxpool = None

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.dropout_rgb_feat = nn.Dropout2d(p=0.2)
        self.dropout_dep_feat = nn.Dropout2d(p=0.2)        

        self.aggregation = Aggregate_Patches()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]

        bt,c,hd,wd = depth.size()

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att_ = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att_.shape
        rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))

        depth_patch_att_1, depth_patch_att_2, depth_patch_att_3, depth_patch_att_4, depth_patch_att_5 = self.depth_pretrained(x2.to(self.device))

        depth_patch_att = torch.cat((depth_patch_att_1,
                                depth_patch_att_2,
                                depth_patch_att_3,
                                depth_patch_att_4,
                                depth_patch_att_5), 1)
        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)

        rgb_patch_att = self.rgb_features(rgb_patch_att)
        depth_patch_att = self.depth_features(depth_patch_att)

        rgb_patch_att = self.dropout_rgb_feat(rgb_patch_att)
        depth_patch_att = self.dropout_dep_feat(depth_patch_att)


        return self.aggregation(rgb_patch_att, depth_patch_att)
    

class CalibNet_DINOV2_LTC(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2_LTC,self).__init__()
        self.scale = depth_scale
        self.patch_size = 14

        self.backbone = vit_small(patch_size=self.patch_size,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        # self.backbone.to(device)
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)

        self.depth_resnet = nn.Sequential(
            nn.MaxPool2d(kernel_size=5,stride=1,padding=2),  # outplanes = 256
            resnet18(inplanes=1,planes=32),
        )
        self.depth_aptmaxpool = None

        self.depth_pretrained = FeatureExtractDepth()

        pretrained_dict = torch.load('pretrained_depth/best_model.pth.tar')['state_dict'] 
        filtered_dict = {k: v for k, v in pretrained_dict.items() if 'FeatureExtractDepth.' in k}
        filtered_dict = {k.replace('FeatureExtractDepth.', ''): v for k, v in filtered_dict.items() if 'FeatureExtractDepth.' in k}

        model_dict = self.depth_pretrained.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(k in 'FeatureExtractDepth.'+i for i in model_dict)} #'FeatureExtractorDepth.'+model_dict 
        model_dict.update(pretrained_dict)
        self.depth_pretrained.load_state_dict(filtered_dict)

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.dropout_rgb_feat = nn.Dropout2d(p=0.2)
        self.dropout_dep_feat = nn.Dropout2d(p=0.2)        

        self.conv_hspp = nn.Conv2d(in_channels=384+192, out_channels=256, kernel_size=4)

        wiring = AutoNCP(1024, 512)  # 16 units, 1 motor neuron

        self.ltc = LTC(8960, wiring, batch_first=True)

        self.fcc1 = nn.Linear(512,256)
        self.dropout_fcc1 = nn.Dropout(p=0.2)
        self.relu_fcc1 = nn.ReLU()
        self.fcc2 = nn.Linear(256,192)
        self.dropout_fcc2 = nn.Dropout(p=0.2)
        self.relu_fcc2 = nn.ReLU()
        self.fcc3 = nn.Linear(192,128)
        self.relu_fcc3 = nn.ReLU()

        self.fcc_r = nn.Linear(128,3)
        self.fcc_t = nn.Linear(128,3)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]

        bt,c,hd,wd = depth.size()

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att_ = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att_.shape
        rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))

        depth_patch_att_1, depth_patch_att_2, depth_patch_att_3, depth_patch_att_4, depth_patch_att_5 = self.depth_pretrained(x2)

        depth_patch_att = torch.cat((depth_patch_att_1,
                                depth_patch_att_2,
                                depth_patch_att_3,
                                depth_patch_att_4,
                                depth_patch_att_5), 1)
        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)

        rgb_patch_att = self.rgb_features(rgb_patch_att)
        depth_patch_att = self.depth_features(depth_patch_att)

        rgb_patch_att = self.dropout_rgb_feat(rgb_patch_att)
        depth_patch_att = self.dropout_dep_feat(depth_patch_att)

        feat = torch.cat((rgb_patch_att, depth_patch_att), dim=1)

        feat = self.conv_hspp(feat)

        spp = spatial_pyramid_pool(feat,bt,[int(feat.size(2)),int(feat.size(3))],[5,3,1])

        out,_ = self.ltc(spp)

        out = self.fcc1(out)
        out = self.dropout_fcc1(out)
        out = self.relu_fcc1(out)
        out = self.fcc2(out)
        out = self.dropout_fcc2(out)
        out = self.relu_fcc2(out)
        out = self.fcc3(out)
        out = self.relu_fcc3(out)
        rot = self.fcc_r(out)
        tsl = self.fcc_t(out)

        return rot, tsl


class CalibNet_DINOV2_patch_CalAgg(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2_patch_CalAgg,self).__init__()
        self.scale = depth_scale
        self.patch_size = 14
 
        self.backbone = vit_small(patch_size=self.patch_size,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)

        self.depth_pretrained = FeatureExtractDepth()
        pretrained_dict = torch.load('pretrained_depth/best_model.pth.tar')['state_dict'] 
        filtered_dict = {k: v for k, v in pretrained_dict.items() if 'FeatureExtractDepth.' in k}
        filtered_dict = {k.replace('FeatureExtractDepth.', ''): v for k, v in filtered_dict.items() if 'FeatureExtractDepth.' in k}
        model_dict = self.depth_pretrained.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(k in 'FeatureExtractDepth.'+i for i in model_dict)} #'FeatureExtractorDepth.'+model_dict 
        model_dict.update(pretrained_dict)
        self.depth_pretrained.load_state_dict(filtered_dict)

        self.depth_aptmaxpool = None

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.dropout_rgb_feat = nn.Dropout2d(p=0.2)
        self.dropout_dep_feat = nn.Dropout2d(p=0.2)        

        self.aggregation = Aggregation(inplanes=384+192, planes=96)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        
        bt,c,hd,wd = depth.size()

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att_ = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att_.shape
        rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))

        depth_patch_att_1, depth_patch_att_2, depth_patch_att_3, depth_patch_att_4, depth_patch_att_5 = self.depth_pretrained(x2.to(self.device))

        depth_patch_att = torch.cat((depth_patch_att_1,
                                depth_patch_att_2,
                                depth_patch_att_3,
                                depth_patch_att_4,
                                depth_patch_att_5), 1)
        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)

        rgb_patch_att = self.rgb_features(rgb_patch_att)
        depth_patch_att = self.depth_features(depth_patch_att)

        rgb_patch_att = self.dropout_rgb_feat(rgb_patch_att)
        depth_patch_att = self.dropout_dep_feat(depth_patch_att)

        feat = torch.cat((rgb_patch_att,depth_patch_att),dim=1)

        return self.aggregation(feat)
    

class CalibNet_DINOV2_patch_RGB_CalAgg(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2_patch_RGB_CalAgg,self).__init__()
        self.scale = depth_scale
        self.patch_size = 14
 
        self.backbone = vit_small(patch_size=self.patch_size,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)

        self.depth_pretrained = nn.Sequential(
            nn.MaxPool2d(kernel_size=5,stride=1,padding=2),  # outplanes = 256
            resnet18(inplanes=1,planes=32),
        )

        self.depth_aptmaxpool = None

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.dropout_rgb_feat = nn.Dropout2d(p=0.2)
        self.dropout_dep_feat = nn.Dropout2d(p=0.2)        

        self.aggregation = Aggregation(inplanes=384+256, planes=96)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # add 2 channels to depth image
        bt,c,hd,wd = depth.size()

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att_ = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att_.shape
        rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))

        depth_patch_att = self.depth_pretrained(x2.to(self.device))[-1]

        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)

        rgb_patch_att = self.rgb_features(rgb_patch_att)
        depth_patch_att = self.depth_features(depth_patch_att)

        rgb_patch_att = self.dropout_rgb_feat(rgb_patch_att)
        depth_patch_att = self.dropout_dep_feat(depth_patch_att)

        feat = torch.cat((rgb_patch_att,depth_patch_att),dim=1)

        return self.aggregation(feat)
    


class CalibNet_DINOV2_patch_RGB(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2_patch_RGB,self).__init__()
        self.scale = depth_scale
        self.patch_size = 14
 
        self.backbone = vit_small(patch_size=self.patch_size,
                                    img_size=526,
                                    init_values=1.0,
                                    block_chunks=0
                                )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')
        self.backbone.load_state_dict(state_dict, strict=True)

        self.depth_pretrained = nn.Sequential(
            nn.MaxPool2d(kernel_size=5,stride=1,padding=2),  # outplanes = 256
            resnet18(inplanes=1,planes=32),
        )

        self.depth_aptmaxpool = None

        self.rgb_features = nn.Identity()
        self.depth_features = nn.Identity()

        self.dropout_rgb_feat = nn.Dropout2d(p=0.2)
        self.dropout_dep_feat = nn.Dropout2d(p=0.2)        

        self.aggregation = Aggregate_Patches(indepth=256)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        bt,c,hd,wd = depth.size()

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att_ = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att_.shape
        rgb_patch_att = torch.reshape(rgb_patch_att_, (batch, embedd, hp, wp))

        depth_patch_att = self.depth_pretrained(x2.to(self.device))[-1]

        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)

        rgb_patch_att = self.rgb_features(rgb_patch_att)
        depth_patch_att = self.depth_features(depth_patch_att)

        rgb_patch_att = self.dropout_rgb_feat(rgb_patch_att)
        depth_patch_att = self.dropout_dep_feat(depth_patch_att)


        return self.aggregation(rgb_patch_att, depth_patch_att)


    
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(np.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(np.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
    


def ConvBN(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """
    Perform 2D Convolution with Batch Normalization
    """
    return nn.Sequential(nn.Conv2d(in_planes,
                                   out_planes,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = dilation if dilation > 1 else pad,
                                   dilation = dilation,
                                   bias=False),
                         nn.BatchNorm2d(out_planes))

class FeatureExtractDepth(nn.Module):
    """
    Feature extraction block for Depth branch
    """
    def __init__(self):
        super(FeatureExtractDepth, self).__init__()
        self.inplanes = 32

        self.conv_block1 = nn.Sequential(ConvBN(1, 16, 11, 1, 5, 1),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(ConvBN(16, 32, 7, 2, 3, 1),
                                         nn.ReLU())
        self.conv_block3 = nn.Sequential(ConvBN(32, 64, 5, 2, 2, 1),
                                         nn.ReLU())

        self.level64_pool = nn.MaxPool2d((64, 64), stride=(64,64))
        self.level64_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level64_relu = nn.ReLU()

        self.level32_pool = nn.MaxPool2d((32, 32), stride=(32,32))
        self.level32_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level32_relu = nn.ReLU()

        self.level16_pool = nn.MaxPool2d((16, 16), stride=(16,16))
        self.level16_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level16_relu = nn.ReLU()

        self.level8_pool = nn.MaxPool2d((8, 8), stride=(8,8))
        self.level8_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level8_relu = nn.ReLU()

    def forward(self, x):
        # x = x.unsqueeze(1)
        m_in = (x > 0).detach().float()
        new_conv1 = self.conv_block1(x)
        new_conv2 = self.conv_block2(new_conv1)
        new_conv3 = self.conv_block3(new_conv2)
        interp_size = (new_conv3.size()[2], new_conv3.size()[3])
        op_maskconv = new_conv3

        op_l64_pool     = self.level64_pool(op_maskconv)
        op_l64_conv     = self.level64_conv(op_l64_pool)
        op_l64          = self.level64_relu(op_l64_conv)
        op_l64_upsample = F.interpolate(input = op_l64,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l32_pool     = self.level32_pool(op_maskconv)
        op_l32_conv     = self.level32_conv(op_l64_pool)
        op_l32          = self.level32_relu(op_l64_conv)
        op_l32_upsample = F.interpolate(input = op_l32,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l16_pool     = self.level16_pool(op_maskconv)
        op_l16_conv     = self.level16_conv(op_l16_pool)
        op_l16          = self.level16_relu(op_l16_conv)
        op_l16_upsample = F.interpolate(input = op_l16,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l8_pool      = self.level8_pool(op_maskconv)
        op_l8_conv      = self.level8_conv(op_l8_pool)
        op_l8           = self.level8_relu(op_l8_conv)
        op_l8_upsample = F.interpolate(input = op_l8,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')
        return op_maskconv, op_l8_upsample, op_l16_upsample,\
               op_l32_upsample, op_l64_upsample


if __name__=="__main__":
    from torchview import draw_graph

    model = CalibNet_DINOV2_patch()
    model_graph = draw_graph(model, input_size = [(2,3,1246,378),(2,1,1246,378)]  ,  expand_nested=True)
    # model_graph.visual_graph
    model_graph.resize_graph(scale=5.0) # scale as per the view 
    model_graph.visual_graph.render(format='svg')


