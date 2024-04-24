#!/bin/bash

checkpoints=$1
dataset=$2 
ddpm_steps=${3:-1000}

out_dir="${dataset}/outputs_test_${ddpm_steps}"

python ./scripts/sr_val_ddpm_lensless.py --init-img "${dataset}/inputs_512" --outdir "${out_dir}" --ddpm_steps "${ddpm_steps}" --ckpt "${checkpoints}"

./scripts/iqa.sh "${out_dir}" "${dataset}/gts"