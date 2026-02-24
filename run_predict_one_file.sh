#!/bin/bash
# Example script to run segmentation prediction for PVS (Perivascular Spaces)
# Supports both old .h5 (v1/v2) and new SavedModel (v3) model formats.
#
# Adjust MODEL_DIR, IMAGE_DIR and output path to your setup.
# --batch_size: number of slices per batch (increase for GPU, default 1 for CPU)
# --gpu: GPU index to use (-1 for CPU)
#
# @author : Philippe Boutinaud - Fealinx

# ---- New T1-only models (v3, SavedModel, 5 folds) ----
MODEL_DIR=./T1-PVS
IMAGE_DIR=./images

python ./predict_one_file.py \
    --verbose --gpu 0 --batch_size 1 \
    -m $MODEL_DIR/20241214-162357_ResUnet3D-8.9.2-1.5-T1.VRS_prodT1only_fold_0_bestvalloss.tf_inference \
    -m $MODEL_DIR/20241214-162517_ResUnet3D-8.9.2-1.5-T1.VRS_prodT1only_fold_1_bestvalloss.tf_inference \
    -m $MODEL_DIR/20241214-162357_ResUnet3D-8.9.2-1.5-T1.VRS_prodT1only_fold_2_bestvalloss.tf_inference \
    -m $MODEL_DIR/20241214-162517_ResUnet3D-8.9.2-1.5-T1.VRS_prodT1only_fold_3_bestvalloss.tf_inference \
    -m $MODEL_DIR/20241214-162357_ResUnet3D-8.9.2-1.5-T1.VRS_prodT1only_fold_4_bestvalloss.tf_inference \
    -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
    -o ./predicted/test_pvs_t1.nii.gz

# ---- New T1+FLAIR models (v3, SavedModel, 5 folds) ----
# MODEL_DIR=./T1.FLAIR-PVS
# python ./predict_one_file.py \
#     --verbose --gpu 0 --batch_size 1 \
#     -m $MODEL_DIR/20250109-180804_ResUnet3D-8.9.2-1.5-T1_FLAIR.VRS_prodT1only_fold_0_bestvalloss.tf_inference \
#     -m $MODEL_DIR/20250109-182235_ResUnet3D-8.9.2-1.5-T1_FLAIR.VRS_prodT1only_fold_1_bestvalloss.tf_inference \
#     -m $MODEL_DIR/20250109-180804_ResUnet3D-8.9.2-1.5-T1_FLAIR.VRS_prodT1only_fold_2_bestvalloss.tf_inference \
#     -m $MODEL_DIR/20250109-182235_ResUnet3D-8.9.2-1.5-T1_FLAIR.VRS_prodT1only_fold_3_bestvalloss.tf_inference \
#     -m $MODEL_DIR/20250109-180804_ResUnet3D-8.9.2-1.5-T1_FLAIR.VRS_prodT1only_fold_4_bestvalloss.tf_inference \
#     -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
#     -i $IMAGE_DIR/test_FLAIR_Axial_resampled_111_cropped_intensity_normed.nii.gz \
#     -o ./predicted/test_pvs_t1flair.nii.gz

# ---- Legacy T1-only models (v1, .h5, 6 folds) ----
# MODEL_DIR=./PVS/v1/T1.PVS
# python ./predict_one_file.py \
#     --verbose --gpu 0 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_0_model.h5 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_1_model.h5 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_2_model.h5 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_3_model.h5 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_4_model.h5 \
#     -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_5_model.h5 \
#     -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
#     -o ./predicted/test_pvs.nii.gz