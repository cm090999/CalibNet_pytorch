import torch
from torch import nn
import torch.nn.functional as F
from Modules import resnet18
import numpy as np

# from dinov2.dinov2.models.vision_transformer import vit_small
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
        feat = torch.cat((x1,x2),dim=1)  # [B,C1+C2,H,W]
        x_rot, x_tr =  self.aggregation(feat)
        return x_rot, x_tr
    
class CalibNet_DINOV2(nn.Module):
    def __init__(self,backbone_pretrained=False,depth_scale=100.0):
        super(CalibNet_DINOV2,self).__init__()
        self.scale = depth_scale

        # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

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
        # rgb_res = resizeToMultiple(rgb,14)
        # depth_res = resizeToMultiple(depth_3,14)
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
    def __init__(self) -> None:
        super(Aggregate_Patches, self).__init__()
        self.conv0rgb = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv1rgb = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1)
        self.conv2rgb = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)

        self.conv0depth = nn.Conv2d(in_channels=192, out_channels=168, kernel_size=3, padding=1)
        self.conv1depth = nn.Conv2d(in_channels=168, out_channels=156, kernel_size=3, padding=1)
        self.conv2depth = nn.Conv2d(in_channels=156, out_channels=128, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.adaptmaxpool = nn.AdaptiveAvgPool2d((5,5))

        self.fc1 = nn.Linear(2*384, 256)

        self.fc1_t = nn.Linear(32*5*5, 512)
        self.layerNorm_tr1 = nn.LayerNorm(512)
        self.dropout1t = nn.Dropout(p=0.2)
        self.fc2_t = nn.Linear(512,128)
        self.fc3_t = nn.Linear(128, 3)

        self.fc1_r = nn.Linear(32*5*5, 512)
        self.layerNorm_rot1 = nn.LayerNorm(512)
        self.dropout1r = nn.Dropout(p=0.2)
        self.fc2_r = nn.Linear(512,128)
        self.fc3_r = nn.Linear(128, 3)

        self.relu = nn.ReLU()

        
    
    def forward(self, rgb_patches, depth_patches):
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

        x = self.conv3(x)
        x = self.maxpool2(x)



        x = self.adaptmaxpool(x)

        x = x.reshape(x.shape[0], -1)

        # cls = torch.cat((cls_embedd_rgb, cls_embedd_depth), dim=1)

        # cls = self.fc1(cls)

        # total = torch.cat((cls,x), dim=1)

        tr1 = self.fc1_t(x)
        # tr1 = self.layerNorm_tr1(tr1)
        tr1 = self.dropout1t(tr1)
        tr1 = self.fc2_t(tr1)
        tr = self.fc3_t(tr1)

        rot1 = self.fc1_r(x)
        # rot1 = self.layerNorm_rot1(rot1)
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

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(k in 'FeatureExtractDepth.'+i for i in model_dict)} #'FeatureExtractorDepth.'+model_dict 
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.depth_pretrained.load_state_dict(filtered_dict)
        

        self.aggregation = Aggregate_Patches()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        return
    
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]

        # dp = self.depth_pretrained(depth.to(self.device))

        # add 2 channels to depth image
        bt,c,hd,wd = depth.size()
        if c ==1:
            depth_3 = torch.zeros((bt,3,hd,wd))
        depth_3[:, 0, :, :] = depth[:, 0, :, :]
        depth_3[:, 1, :, :] = depth[:, 0, :, :]
        depth_3[:, 2, :, :] = depth[:, 0, :, :]

        hp = hd // self.patch_size
        wp = wd // self.patch_size

        self.depth_aptmaxpool = nn.AdaptiveMaxPool2d((hp, wp))

        x1,x2 = rgb,depth_3.clone()  # clone dpeth, or it will change depth in '/' operation
        x2 /= self.scale

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        rgb_att = self.backbone.forward_features(x1)
        rgb_patch_att = rgb_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        batch,embedd, _ = rgb_patch_att.shape
        rgb_patch_att = torch.reshape(rgb_patch_att, (batch, embedd, hp, wp))
        rgb_cls_att = rgb_att['x_norm_clstoken'] 

        # depth_att = self.backbone.forward_features(x2)
        # depth_patch_att = depth_att['x_norm_patchtokens'].permute(0,2,1) # (batch,embeddings,hp*wp)
        # batch,embedd, _ = depth_patch_att.shape
        # depth_patch_att = torch.reshape(depth_patch_att, (batch, embedd, hp, wp))
        # depth_cls_att = depth_att['x_norm_clstoken']
        # depth_patch_att = self.depth_resnet(depth.to(self.device))[-1]
        # depth_patch_att = self.depth_aptmaxpool(depth_patch_att)
        depth_patch_att_1, depth_patch_att_2, depth_patch_att_3, depth_patch_att_4, depth_patch_att_5 = self.depth_pretrained(depth.to(self.device))

        depth_patch_att = torch.cat((depth_patch_att_1,
                                depth_patch_att_2,
                                depth_patch_att_3,
                                depth_patch_att_4,
                                depth_patch_att_5), 1)
        depth_patch_att = self.depth_aptmaxpool(depth_patch_att)


        return self.aggregation(rgb_patch_att, depth_patch_att)
    

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


def resizeToMultiple(image: torch.tensor,mult: int):
    batch,c,width,height = image.size()
    heigth_new = ((height // mult) + 1) * mult
    width_new = ((width // mult) + 1) * mult

    img_new = torch.zeros((batch,c,width_new,heigth_new))
    
    img_new[:,:,:width, :height] = image
    return img_new

if __name__=="__main__":
    x = (torch.rand(2,3,1246,378),torch.rand(2,1,1246,378))
    # x_resized = (resizeToMultiple(x[0],14).cuda(), resizeToMultiple(x[0],14).cuda())

    model_dino = CalibNet_DINOV2_patch().cuda()
    model_dino.eval()
    rotationd,translationd = model_dino(*x)

    model = CalibNet(backbone_pretrained=False).cuda()
    model.eval()
    rotation,translation = model(*x)
    print("translation size:{}".format(translation.size()))
    print("rotation size:{}".format(rotation.size()))


