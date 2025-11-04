#!/bin/bash
MODEL_DIR=./WMH/v0/T1-FLAIR.WMH

#IMAGE_DIR=./Amsterdam_GE1T5_150
#T1_IMAGE=$IMAGE_DIR/T1_1mm.nii.gz
#FLAIR_IMAGE=$IMAGE_DIR/FLAIR_1mm.nii.gz
#ARGS="--normalize -a"

#IMAGE_DIR=./MRIandPET
#T1_IMAGE=$IMAGE_DIR/25101315_t1_mp2rage_sag_p3_iso_20250922142804_11_Crop_1.nii.gz
#FLAIR_IMAGE=$IMAGE_DIR/25101315_3DT2FLAIR_space_dark-fluid_sag_p2_iso_20250922142804_6_Crop_1.nii.gz
#ARGS="--normalize -a"

IMAGE_DIR=./26-oct
T1_IMAGE=$IMAGE_DIR/20200000_t1_sag_3D_20251018113337_5.nii.gz
FLAIR_IMAGE=$IMAGE_DIR/20580000_t2_sag_FLAIR_3D_20251018113337_14.nii.gz
ARGS="--normalize -a"

ARGS="--verbose --gpu 0 $ARGS"
python ./predict_one_file.py \
    $ARGS \
    -m $MODEL_DIR/20220212-100713_Unet3Dv2-10.7.2-1.8-T1_FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_0_model.h5 \
    -m $MODEL_DIR/20220212-104458_Unet3Dv2-10.7.2-1.8-T1_FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_1_model.h5 \
    -m $MODEL_DIR/20220212-164523_Unet3Dv2-10.7.2-1.8-T1_FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_2_model.h5 \
    -m $MODEL_DIR/20220212-201148_Unet3Dv2-10.7.2-1.8-T1_FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_3_model.h5 \
    -m $MODEL_DIR/20220213-093307_Unet3Dv2-10.7.2-1.8-T1_FLAIR.WMH_fold_WMH_1x5_2ndUnat_fold_4_model.h5 \
    -i $T1_IMAGE \
    -i $FLAIR_IMAGE \
    -o ./predicted/prediction.nii.gz \
    -t ./intermediates

commit=
if [ -d .git ] ; then
    commit=`git rev-parse --short --verify HEAD`
    if git diff-index --name-only HEAD | read dummy ; then
        commit="$commit (dirty)"
    fi
fi

echo "T1   : $T1_IMAGE"
echo "FLAIR: $FLAIR_IMAGE"
echo "ARGS : $ARGS"
echo "GIT  : $commit"
