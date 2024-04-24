# transfer flatnet dataset to single dataset in one folder, resize to 512x512
exp_name = "fft-mulnew9-1280-1408-learn-1280-1408-meas-decoded_sim_spatial_weight"
output_name = "flatnet_multi_decode_sim"

output_dir = '../data/%s'%output_name
source_flatnet_dir = '/root/caixin/flatnet/output_phase_mask_Feb_2020_size_384/%s_val_train/val_latest_tag_384_gain_1.0'%exp_name

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
        if file.endswith('png') and file.startswith('output_'):
            #remove fft_ prefix
            shutil.copy(os.path.join(cls_dir, file), os.path.join(output_inputs, file[7:]))
            gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
            shutil.copy(gt_file, os.path.join(output_gts, file[7:]))
            #resize to 512x512
            img = cv2.imread(os.path.join(output_inputs, file[7:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_inputs, file[7:]), img)
            img = cv2.imread(os.path.join(output_gts, file[7:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_gts, file[7:]), img)
            index += 1
            print("index: ", index, file[7:])

#for validation set
output_dir = '../data/%s_val'%output_name
source_flatnet_dir = '/root/caixin/flatnet/output_phase_mask_Feb_2020_size_384/%s/val_latest_tag_384_gain_1.0'%exp_name

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
        if file.endswith('png') and file.startswith('output_'):
            #remove fft_ prefix
            shutil.copy(os.path.join(cls_dir, file), os.path.join(output_inputs, file[7:]))
            gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
            shutil.copy(gt_file, os.path.join(output_gts, file[7:]))
            #resize to 512x512
            img = cv2.imread(os.path.join(output_inputs, file[7:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_inputs, file[7:]), img)
            img = cv2.imread(os.path.join(output_gts, file[7:]))
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(output_gts, file[7:]), img)
            index += 1
            print("index: ", index, file[7:])