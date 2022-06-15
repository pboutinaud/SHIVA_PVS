#!/bin/bash
MODEL_DIR=./PVS/v1/T1.PVS
IMAGE_DIR=./images
python ./predict_one_file.py \
    --verbose --gpu 0 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_0_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_1_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_2_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_3_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_4_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_5_model.h5 \
    -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
    -o ./predicted/test.nii.gz