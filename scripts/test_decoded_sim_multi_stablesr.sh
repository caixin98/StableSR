# python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ckpts/stablesr_000117.ckpt --vqgan_ckpt ckpts/vqgan_cfw_00011.ckpt --init-img data/flatnet2single/gts --outdir data/flatnet2single/outputs_gts_dec_w1 --ddpm_steps 200 --dec_w 1.0 --colorfix_type adain
if [ $# -eq 0 ]; then
  echo "必须提供参数" >&2
  exit 1
fi
rm -rf data/flatnet_val_multi_decoded_sim_old/$1
python ./scripts/sr_val_ddpm_text_T_vqganfin_old.py  --init-img data/flatnet_val_multi_decoded_sim_old/inputs --outdir data/flatnet_val_multi_decoded_sim_old/$1  --ckpt /root/caixin/StableSR/ckpts/stablesr_000117.ckpt --n_samples 5   --ddpm_steps 200  --vqgan_ckpt /root/caixin/StableSR/ckpts/vqgan_cfw_00011.ckpt --colorfix_type adain --dec_w 0.0

./scripts/iqa.sh data/flatnet_val_multi_decoded_sim_old/$1  
#  --colorfix_type adain

# --ckpt logs/2024-01-01T18-14-14_lensless/checkpoints/last.ckpt
# --precision full
# 