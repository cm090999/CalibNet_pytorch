#!/bin/bash

# Define the arguments and their corresponding values
arg0="--epoch 35 --batch_size 4 --name download_finetuned_no_randomCrop_ch_ft_tsl --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg1="--epoch 35 --batch_size 4 --name CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_ft_tsl --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg2="--epoch 35 --batch_size 4 --name CalibNet_DINOV2_patch_no_randomCrop_ch__ft_tsl --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"

arguments=(
  "${arg1}"
  "${arg0}"
  "${arg2}"
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