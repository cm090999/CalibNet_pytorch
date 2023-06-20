#!/bin/bash

# Define the arguments and their corresponding values
arg0="--name CalibNet_DINOV2_patch_RGB_CalAgg_no_randomCrop_ch --randomCrop 1.0 --pretrained '' --model_name CalibNet_DINOV2_patch_RGB_CalAgg"
arg1="--name CalibNet_DINOV2_patch_RGB_no_randomCrop_ch --randomCrop 1.0 --pretrained '' --model_name CalibNet_DINOV2_patch_RGB"

arguments=(
  # "${arg0}"
  "${arg1}"
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

