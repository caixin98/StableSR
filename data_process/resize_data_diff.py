# resize the images to 384x384
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
def resize_images(input_dir, output_dir, size = 380):
    if output_dir is None:
        output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in tqdm(os.listdir(input_dir)):
        img = cv2.imread(os.path.join(input_dir, file))
        img = cv2.resize(img, (size, size))
        img = img[85:85 + 210, 0:380, :]
        cv2.imwrite(os.path.join(output_dir, file), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/root/caixin/StableSR/data/fft-mulnew9-diffusercam_val/gts", )
    parser.add_argument('--output_dir', type=str, default="/root/caixin/StableSR/data/fft-mulnew9-diffusercam_val/gts_resize", )
    parser.add_argument('--size', type=int, default=380)
    args = parser.parse_args()
    resize_images(args.input_dir, args.output_dir, args.size)