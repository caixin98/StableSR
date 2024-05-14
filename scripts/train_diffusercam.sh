# python main.py --train --base configs/stableSRNew/v2-finetune_lensless_T_512.yaml --gpus 0,1,2,3,4,5,6,7, --scale_lr False -r logs/2024-01-01T18-14-14_lensless
# python main.py --train --base configs/stableSRNew/v2-finetune_diffusercam_updn_T_512.yaml --gpus 0,1,2,3,4,5,6,7, --scale_lr False --name diffusercam_updn

python main.py --train --base configs/stableSRNew/v2-finetune_diffusercam_decoded_sim_multi_T_512.yaml --gpus 0,1,2,3,4,5,6,7, --scale_lr False --name diffusercam_decoded_sim_padding --logdir /mnt/data/oss_beijing/caixin/updated/logs
# v2-finetune_diffusercam_decoded_sim_multi_T_512