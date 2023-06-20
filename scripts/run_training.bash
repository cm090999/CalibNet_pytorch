#!/bin/bash

# Define the arguments and their corresponding values
arg6="--name CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch --randomCrop 1.0 --pretrained '' --model_name CalibNet_DINOV2_patch_CalAgg"

arg0="--name CalibNet_DINOV2_cls_with_randomCrop_ch --randomCrop 0.8 --pretrained '' --model_name CalibNet_DINOV2"
arg1="--name download_finetuned_with_randomCrop_ch_cont --randomCrop 0.8 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg2="--name CalibNet_DINOV2_cls_with_randomCrop_ch_cont --randomCrop 0.8 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg3="--name CalibNet_DINOV2_patch_with_randomCrop_ch_cont --randomCrop 0.8 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg4="--name CalibNet_DINOV2_LTC_with_randomCrop_ch_cont --randomCrop 0.8 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5="--name CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont --randomCrop 0.8 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arguments=(
  "${arg6}"
  "${arg0}"
  "${arg1}"
  "${arg2}"
  "${arg3}"
  "${arg4}"
  "${arg5}"
)

# Activate the virtual environment
source .venvCalib/bin/activate

# Loop through the arguments and run the script
for arg_value in "${arguments[@]}"; do
  echo "Running script with argument: $arg_value"
  if python train_orig.py $arg_value; then
    echo "Script completed successfully"
  else
    echo "Script encountered an error"
  fi
done

# Deactivate the virtual environment
deactivate

# arg2="--name CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch --randomCrop 0.8 --pretrained '' --model_name CalibNet_DINOV2_patch_CalAgg"

