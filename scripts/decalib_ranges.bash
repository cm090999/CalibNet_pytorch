#!/bin/bash

# Define the arguments and their corresponding values
arg0="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"

arg3="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"

arg6="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arg9="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"

arg12="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16="--max_deg 10 --max_tran 0.2 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg0a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"

arg3a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"

arg6a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arg9a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"

arg12a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16a="--max_deg 5 --max_tran 0.1 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg0b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"

arg3b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"

arg6b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arg9b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"

arg12b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16b="--max_deg 2.5 --max_tran 0.05 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg0c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"

arg3c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"

arg6c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arg9c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"

arg12c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16c="--max_deg 0.5 --max_tran 0.01 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg0d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"

arg3d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"

arg6d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"

arg9d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"

arg12d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16d="--max_deg 15 --max_tran 0.3 --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

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

  "${arg0a}"
  "${arg1a}"
  "${arg2a}"
  "${arg3a}"
  "${arg4a}"
  "${arg5a}"
  "${arg6a}"
  "${arg7a}"
  "${arg8a}"
  "${arg9a}"
  "${arg10a}"
  "${arg11a}"
  "${arg12a}"
  "${arg13a}"
  "${arg14a}"
  "${arg15a}"
  "${arg16a}"

  "${arg0b}"
  "${arg1b}"
  "${arg2b}"
  "${arg3b}"
  "${arg4b}"
  "${arg5b}"
  "${arg6b}"
  "${arg7b}"
  "${arg8b}"
  "${arg9b}"
  "${arg10b}"
  "${arg11b}"
  "${arg12b}"
  "${arg13b}"
  "${arg14b}"
  "${arg15b}"
  "${arg16b}"

  "${arg0c}"
  "${arg1c}"
  "${arg2c}"
  "${arg3c}"
  "${arg4c}"
  "${arg5c}"
  "${arg6c}"
  "${arg7c}"
  "${arg8c}"
  "${arg9c}"
  "${arg10c}"
  "${arg11c}"
  "${arg12c}"
  "${arg13c}"
  "${arg14c}"
  "${arg15c}"
  "${arg16c}"

  "${arg0d}"
  "${arg1d}"
  "${arg2d}"
  "${arg3d}"
  "${arg4d}"
  "${arg5d}"
  "${arg6d}"
  "${arg7d}"
  "${arg8d}"
  "${arg9d}"
  "${arg10d}"
  "${arg11d}"
  "${arg12d}"
  "${arg13d}"
  "${arg14d}"
  "${arg15d}"
  "${arg16d}"
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