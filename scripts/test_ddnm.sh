# python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ckpts/stablesr_000117.ckpt --vqgan_ckpt ckpts/vqgan_cfw_00011.ckpt --init-img data/flatnet2single/gts --outdir data/flatnet2single/outputs_gts_dec_w1 --ddpm_steps 200 --dec_w 1.0 --colorfix_type adain
if [ $# -eq 0 ]; then
  echo "必须提供参数" >&2
  exit 1
fi
rm -rf data/flatnet_val/$1
python ./scripts/sr_val_ddnm_lensless.py  --init-img data/flatnet_val/inputs --outdir data/flatnet_val/$1  --ckpt logs/2024-01-04T16-36-14_flatnet/checkpoints/last.ckpt --n_samples 4 --precision full 

./scripts/iqa.sh data/flatnet_val/$1  data/flatnet_val/gts
#  --colorfix_type adain

# --ckpt logs/2024-01-01T18-14-14_lensless/checkpoints/last.ckpt

# 