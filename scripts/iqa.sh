dir=${2:-/root/caixin/StableSR/data/flatnet_output_384_val/gts}

python data_process/resize_data.py --input_dir $1 --output_dir $1\_384 --size 384

python  IQA-PyTorch/inference_iqa.py -m ssimc -i $1\_384 -r $dir

python  IQA-PyTorch/inference_iqa.py -m lpips -i $1\_384 -r $dir

python  IQA-PyTorch/inference_iqa.py -m psnr -i $1\_384 -r $dir

# python  IQA-PyTorch/inference_siqa.py -m SSIM -i $1 -r $2

# python  IQA-PyTorch/inference_iqa.py -m LPIPS -i $1 -r $2

# python  IQA-PyTorch/inference_piqa.py -m PSNR -i $1 -r $2