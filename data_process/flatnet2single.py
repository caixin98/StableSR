# transfer flatnet dataset to single dataset in one folder, resize to 512x512
output_dir = '../data/flatnet'
source_flatnet_dir = '/root/caixin/flatnet/output_phase_mask_Feb_2020_size_384/ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-train/val_latest_tag_384_gain_1.0'

# output_dir = '../data/flatnet_val'
# source_flatnet_dir = '/root/caixin/flatnet/output_phase_mask_Feb_2020_size_384/ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-val/val_latest_tag_384_gain_1.0'

source_orig_dir = "/root/caixin/data/orig"
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
output_inputs = os.path.join(output_dir, 'inputs')
output_gts = os.path.join(output_dir, 'gts')

for dir in [output_dir, output_inputs, output_gts]:
    if not os.path.exists(dir):
        os.makedirs(dir)

index = 0
for cls in tqdm(os.listdir(source_flatnet_dir)):
    cls_dir = os.path.join(source_flatnet_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for file in os.listdir(cls_dir):
        if file.endswith('png') and file.startswith('fft_'):
            #remove fft_ prefix
            shutil.copy(os.path.join(cls_dir, file), os.path.join(output_inputs, file[4:]))
            gt_file = os.path.join(source_orig_dir, cls, file[4:]).replace('png', 'JPEG')
            shutil.copy(gt_file, os.path.join(output_gts, file[4:]))
            #resize to 512x512
            img = cv2.imread(os.path.join(output_inputs, file[4:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_inputs, file[4:]), img)
            img = cv2.imread(os.path.join(output_gts, file[4:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_gts, file[4:]), img)
            index += 1
            print("index: ", index, file[4:])