import os
import argparse
import cv2
import numpy as np

def find_matching_images(folders):
    images_dict = {}
    for folder_path in folders:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    if filename not in images_dict:
                        images_dict[filename] = []
                    images_dict[filename].append(os.path.join(root, file))
    return images_dict

def concat_images(image_paths):
    images = [cv2.imread(img) for img in image_paths]
    images = [img for img in images if img is not None]
    if len(images) == 0:
        return None  # 如果没有图像，返回 None
    min_height = min(img.shape[0] for img in images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
    return cv2.hconcat(resized_images)

def main(folders):
    matching_images = find_matching_images(folders)
    
    # 确定保存的基础文件夹, 这里假设保存在第一个输入文件夹中的 "compare" 目录
    base_folder = "data"
    compare_folder = os.path.join(base_folder, 'compare_diff')
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
    
    for name, file_list in matching_images.items():
        if len(file_list) > 1:
            concatenated_image = concat_images(file_list)
            if concatenated_image is not None:
                save_path = os.path.join(compare_folder, f'{name}.png')
                cv2.imwrite(save_path, concatenated_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='拼接多个文件夹中相同名字的图像文件')
    parser.add_argument('folder_paths', type=str, nargs='+', help='需要处理的文件夹路径')

    args = parser.parse_args()
    main(args.folder_paths)


# compare
#python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val/inputs_384 data/flatnet_output_384_val/inputs data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384
# compare_sd_input
# python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val/ddpm_200_384 data/flatnet_output_val/outputs_test_1000_384  data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384
# compare_diff
# python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val_multi_decoded_sim_old/ddpm_200_stablesr_384 data/flatnet_val_multi_decoded_sim_old/ddnm_stablesr_384 data/flatnet_val_multi_decoded_sim_old/diffbirv1_384 data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384

# data/flatnet_val_decoded_sim/test_ddpm_200_384