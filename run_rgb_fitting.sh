#!/bin/bash
set -e


######################### Configuration #########################
# input_dir: the directory of the input images
# output_dir: the directory of the output results
# checkpoints_dir: the directory of the used checkpoints
# topo_assets_dir: the directory of the topo assets, e.g., 3DMM, masks, etc.
#################################################################
input_dir=/root/autodl-tmp/ori1
output_dir=/root/autodl-tmp/ori1/outputs
checkpoints_dir=../checkpoints
topo_assets_dir=../topo_assets
uv3dmm_model_dir=./model/uv3dmm

rm -rf ${output_dir}
#################### Step 1. Preprocess Data ####################
# Read the input images in ${input_dir}
# Save the processed data in ${input_dir}/processed_data and ${input_dir}/processed_data_vis
#################################################################
cd ./RGB_Fitting
python step1_process_data.py \
    --input_dir ${input_dir} \
    --output_dir ${input_dir}/processed_data \
    --checkpoints_dir ${checkpoints_dir} \
    --topo_dir ${topo_assets_dir}

# python step1_process_data.py  --input_dir ../examples/fitting_examples/inputs --output_dir ../examples/fitting_examples/inputs/processed_data --checkpoints_dir ../checkpoints --topo_dir ../topo_assets

###################### Step 2. RGB Fitting ######################
# Read the processed data in ${input_dir}/processed_data
# Save the output results in ${output_dir}
#################################################################
python step2_fit_processed_data.py \
    --input_dir ${input_dir}/processed_data \
    --output_dir ${output_dir} \
    --checkpoints_dir ${checkpoints_dir} \
    --topo_dir ${topo_assets_dir} \
    --uv3dmm_model_dir ${uv3dmm_model_dir}
# python step2_fit_processed_data.py --input_dir ../examples/fitting_examples/inputs/processed_data --output_dir ../examples/fitting_examples/outputs --checkpoints_dir ../checkpoints --topo_dir ../topo_assets --texgan_model_name texgan_ffhq_uv.pth