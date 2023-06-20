#!/bin/bash

# Define the arguments and their corresponding values
arg0="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"
arg3="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"
arg6="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg9="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"
arg12="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg17="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB_CalAgg"
arg18="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB"

arg19="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet"
arg20="--pertFile test_seq_yaw.csv --perturbationaxes 0,1,0,0,0,0 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
 

arguments=(
  "${arg0}"
  "${arg1}"
  "${arg2}"
  "${arg3}"
  "${arg4}"
  "${arg5}"
  "${arg6}"
  "${arg7}"
  "${arg8}"
  "${arg9}"
  "${arg10}"
  "${arg11}"
  "${arg12}"
  "${arg13}"
  "${arg14}"
  "${arg15}"
  "${arg16}"
  "${arg17}"
  "${arg18}"
  "${arg19}"
  "${arg20}"
)

# Activate the virtual environment
source .venvCalib/bin/activate

# Loop through the arguments and run the script
for arg_value in "${arguments[@]}"; do
  echo "Running script with argument: $arg_value"
  if python test_orig.py $arg_value; then
    echo "Script completed successfully"
  else
    echo "Script encountered an error"
  fi
done

# Deactivate the virtual environment
deactivate