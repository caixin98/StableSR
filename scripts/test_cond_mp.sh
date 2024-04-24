# python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ckpts/stablesr_000117.ckpt --vqgan_ckpt ckpts/vqgan_cfw_00011.ckpt --init-img data/flatnet2single/gts --outdir data/flatnet2single/outputs_gts_dec_w1 --ddpm_steps 200 --dec_w 1.0 --colorfix_type adain
gpu_num=`nvidia-smi --list-gpus | wc -l`
lensless_guidance_weight=${1:-1e-3}
echo "lensless_guidance_weight: $lensless_guidance_weight"

lensless_guidance_range_left=${2:-"100"}
lensless_guidance_range_right=${3:-"1000"}
echo "lensless_guidance_range: $lensless_guidance_range_left $lensless_guidance_range_right"
lensless_guidance_flag=${4:-0}
echo "lensless_guidance_flag: $lensless_guidance_flag"
lensless_guidance_steps=${5:-1}
echo "lensless_guidance_steps: $lensless_guidance_steps"
exp_name=lensless_guidance_weight_${lensless_guidance_weight}_range_${lensless_guidance_range_left}_${lensless_guidance_range_right}_flag_${lensless_guidance_flag}_step_${lensless_guidance_steps}
echo "exp_name: $exp_name"
rm -rf data/flatnet_val/$1
for i in $(seq 0 $((gpu_num - 1))); do
CUDA_VISIBLE_DEVICES=$i python ./scripts/sr_val_ddpm_lensless_guidance.py  --init-img data/flatnet_val/inputs --outdir data/flatnet_val/$exp_name --ddpm_steps 1000 --ckpt logs/2024-01-04T16-36-14_flatnet/checkpoints/last.ckpt --precision full --n_samples 2 --guidance_weight $lensless_guidance_weight --guidance_step $lensless_guidance_steps --guidance_range $lensless_guidance_range_left $lensless_guidance_range_right --guidance_flag $lensless_guidance_flag --gpu_num $gpu_num --gpu_id $i &
done
wait


# --colorfix_type adain

# --ckpt logs/2024-01-01T18-14-14_lensless/checkpoints/last.ckpt

# 