dir=${2:-/root/caixin/StableSR/data/fft-mulnew9-diffusercam_val/gts_resize}

python data_process/resize_data_diff.py --input_dir $1 --output_dir $1\_resize 

mkdir $1_result

python  IQA-PyTorch/inference_iqa.py -m psnr -i $1\_resize -r $dir --save_file $1_result/psnr.csv

python  IQA-PyTorch/inference_iqa.py -m ssimc -i $1\_resize -r $dir --save_file $1_result/ssimc.csv

python  IQA-PyTorch/inference_iqa.py -m lpips -i $1\_resize -r $dir --save_file $1_result/lpips.csv




python MANIQA/predict_folder.py $1\_resize

python  IQA-PyTorch/inference_iqa.py -m clipiqa -i $1\_resize --save_file $1_result/clipiqa.csv


python  IQA-PyTorch/inference_iqa.py -m musiq -i $1\_resize --save_file $1_result/musiq.csv


# ./scripts/nr_iqa.sh $1\_resize 

# python  IQA-PyTorch/inference_iqa.py -m ssimc -i $1 -r $dir

# python  IQA-PyTorch/inference_iqa.py -m lpips -i $1 -r $dir

# python  IQA-PyTorch/inference_iqa.py -m psnr -i $1 -r $dir

# python  IQA-PyTorch/inference_siqa.py -m SSIM -i $1 -r $2

# python  IQA-PyTorch/inference_iqa.py -m LPIPS -i $1 -r $2

# python  IQA-PyTorch/inference_piqa.py -m PSNR -i $1 -r $2