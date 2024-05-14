# transfer the diffusercam dataset output to single dataset in one folder, 
# padding it to 380 * 380, then resize to 512 * 512
import os
import argparse
import shutil
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='transfer diffusercam dataset output to single dataset in one folder, padding it to 380 * 380, then resize to 512 * 512')
    parser.add_argument('--exp_name', default='fft-mulnew9-diffusercam', type=str, help='exp_name')
    parser.add_argument('--source_orig_dir', default='/root/caixin/flatnet/data/diffusercam/ground_truth_lensed_png', type=str, help='source original directory')
    parser.add_argument('--output_name', default='diffusercam-updn', type=str, help='output name')
    args = parser.parse_args()
    return args
def region_of_interest(x):
    return x[60:270, 60:440, ...]
def pad_resize(img, size):
    h, w = img.shape[:2]
    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    img = cv2.resize(img, (new_w, new_h))
    pad_h = (size - new_h) // 2
    pad_w = (size - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, size - new_h - pad_h, pad_w, size - new_w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img
def convert_diffusercam_to_single(args, source_diffusercam_dir, output_dir, num = None):
    output_gts = os.path.join(output_dir, 'gts')
    output_inputs = os.path.join(output_dir, 'inputs')
    for dir in [output_dir, output_inputs, output_gts]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    file_list = os.listdir(source_diffusercam_dir)
    if num is not None:
        file_list = file_list[:num]
    for file in file_list:
        if file.endswith('png'):
            shutil.copy(os.path.join(source_diffusercam_dir, file), os.path.join(output_inputs, file))
            gt_file = os.path.join(args.source_orig_dir, file)
            shutil.copy(gt_file, os.path.join(output_gts, file))
            # print(os.path.join(args.source_orig_dir, file))
            img = cv2.imread(os.path.join(output_inputs, file))
            # print(os.path.join(output_inputs, file))
            img = pad_resize(img, 512)
            cv2.imwrite(os.path.join(output_inputs, file), img)
            img = cv2.imread(os.path.join(output_gts, file))
            if img.shape[0] == 270:
                img = region_of_interest(img)
            img = pad_resize(img, 512)
            cv2.imwrite(os.path.join(output_gts, file), img)
if __name__ == '__main__':
    args = parse_args()
    # source_diffusercam_dir = "/root/caixin/lensless-primal-dual/results/learned-primal-dual-and-five-models"
    # output_dir = '../data/%s'%args.output_name + "_val"
    # convert_diffusercam_to_single(args, source_diffusercam_dir, output_dir)
    # source_diffusercam_dir = "/root/caixin/flatnet/output_diffusercam/%s_val_train/val_latest_tag_384_gain_1.0"%args.exp_name
    source_diffusercam_dir = "/root/caixin/lensless-primal-dual/results/learned-primal-dual-and-five-models"
    output_dir = '../data/%s'%args.output_name
    convert_diffusercam_to_single(args, source_diffusercam_dir, output_dir)

    # source_diffusercam_dir = "/root/caixin/lensless-primal-dual/results/learned-primal-dual-and-five-models"
    # output_dir =   output_dir = '../data/%s'%args.output_name + "_val_sub"
    # convert_diffusercam_to_single(args, source_diffusercam_dir, output_dir, num = 10)