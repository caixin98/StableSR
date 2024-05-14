# python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ckpts/stablesr_000117.ckpt --vqgan_ckpt ckpts/vqgan_cfw_00011.ckpt --init-img data/flatnet2single/gts --outdir data/flatnet2single/outputs_gts_dec_w1 --ddpm_steps 200 --dec_w 1.0 --colorfix_type adain
if [ $# -eq 0 ]; then
  echo "必须提供参数" >&2
  exit 1
fi
gpu_num=`nvidia-smi --list-gpus | wc -l`
rm -rf data/diffusercam_unet_padding_decode_sim_val/$1
for i in $(seq 0 $((gpu_num - 1))); do
  CUDA_VISIBLE_DEVICES=$i python ./scripts/sr_val_ddpm_lensless.py  --init-img data/diffusercam_unet_padding_decode_sim_val/inputs --outdir data/diffusercam_unet_padding_decode_sim_val/$1  --ckpt /mnt/data/oss_beijing/caixin/updated/logs/2024-05-12T20-37-48_diffusercam_decoded_sim_padding/checkpoints/last.ckpt --n_samples 5   --ddpm_steps 100 --gpu_num $gpu_num --gpu_id $i &
done
wait
./scripts/iqa_diff.sh data/diffusercam_unet_padding_decode_sim_val/$1  

  # /mnt/data/oss_beijing/caixin/updated/logs/2024-05-12T20-37-48_diffusercam_decoded_sim_padding/checkpoints/last.ckpt

  # /mnt/data/oss_beijing/caixin/updated/logs/2024-05-12T23-03-20_diffusercam_decoded_sim_padding/checkpoints/last.ckpt