#!/bin/bash

# Define the arguments and their corresponding values
arg0="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"
arg3="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"
arg6="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg9="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"
arg12="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg17="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB_CalAgg"
arg18="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB"

arg19="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet"
arg20="--pertFile test_seq_kitti.csv --dataset kitti --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
 

arg0a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg1a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2"
arg2a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_cls_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2"
arg3a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg4a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_LTC"
arg5a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_LTC_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_LTC"
arg6a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg7a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg8a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
arg9a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg10a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch"
arg11a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_with_randomCrop_ch_cont_best.pth --model_name CalibNet_DINOV2_patch"
arg12a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_best.pth --model_name CalibNet"
arg13a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_best.pth --model_name CalibNet"
arg14a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_best.pth --model_name CalibNet"
arg15a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_best.pth --model_name CalibNet"
arg16a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_with_randomCrop_ch_cont_best.pth --model_name CalibNet"

arg17a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_CalAgg_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB_CalAgg"
arg18a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_RGB_no_randomCrop_ch_best.pth --model_name CalibNet_DINOV2_patch_RGB"

arg19a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/download_finetuned_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet"
arg20a="--pertFile test_seq_once.csv --dataset once --randomCrop 1.0 --pretrained checkpoint/CalibNet_DINOV2_patch_CalAgg_no_randomCrop_ch_ft_tsl_best.pth --model_name CalibNet_DINOV2_patch_CalAgg"
 

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
  "${arg17a}"
  "${arg18a}"
  "${arg19a}"
  "${arg20a}"
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