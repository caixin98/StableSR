import os
import argparse
import cv2
import numpy as np
compare_list = []
compare_list.append("data/flatnet_output_val/gts")
compare_list.append("data/flatnet_val/inputs_512")
compare_list.append("data/flatnet_output_val/inputs")
compare_list.append("data/flatnet_output_val/diffbir_v1")
compare_list.append("data/flatnet_multi_decode_sim_val/ddpm_1000")

def create_zoomed_image(img, coordinates, zoom_factor=2):
    """
    Create a zoomed-in overlay on the bottom-right of the image.
    
    :param image_path: Path to the input image.
    :param coordinates: A tuple (x, y, w, h) representing the top-left corner and size of the rectangle.
    :param zoom_factor: The factor by which to zoom the selected area.
    :param save_path: Path where to save the generated image.
    """
    # Read the original image
    # img = cv2.imread(image_path)
    
    # Define the coordinates of the rectangle (top-left corner, width, height)
    x, y, w, h = coordinates
    
    # Crop the desired part
    cropped_part = img[y:y+h, x:x+w].copy()
    
    # Resize (zoom in) the cropped part
    cropped_part = cv2.resize(cropped_part, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    
    # Get the dimensions of the zoomed part and the original image
    zoomed_h, zoomed_w = cropped_part.shape[:2]
    img_h, img_w = img.shape[:2]
    
    # Calculate the position of the zoomed image on the bottom-right
    top_left_x = img_w - zoomed_w
    top_left_y = img_h - zoomed_h
    
    # Make a copy of the original image to draw the rectangle and paste the zoomed part
    img_with_zoom = img.copy()
    
    # Draw a green rectangle around the zoom area on the original image
    cv2.rectangle(img_with_zoom, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Check if the zoomed image fits within the original image dimensions
    if top_left_x >= 0 and top_left_y >= 0:
        # Paste the zoomed part onto the original image
        img_with_zoom[top_left_y:top_left_y + zoomed_h, top_left_x:top_left_x + zoomed_w] = cropped_part
        cv2.rectangle(img_with_zoom, (top_left_x, top_left_y), (top_left_x + zoomed_w-3, top_left_y + zoomed_h-3), (0, 255, 0), 3)
        return img_with_zoom
    else:
        print("Zoomed image is too large to fit on the original image at the bottom-right corner.")
        return img_with_zoom

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

def concat_zoom_images(image_paths, coordinates):
    images = [cv2.imread(img) for img in image_paths]
    images = [img for img in images if img is not None]
    images = [create_zoomed_image(img, coordinates) for img in images]
    if len(images) == 0:
        return None  # 如果没有图像，返回 None
    min_height = min(img.shape[0] for img in images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
    return cv2.hconcat(resized_images)

def main(save_folder, folders, coordinates, img_name = None):
    if img_name is None:
        matching_images = find_matching_images(folders)
    else:
        matching_images = {img_name: []}
        for folder_path in folders:
           matching_images[img_name] += [os.path.join(folder_path, img_name)]
    # 确定保存的基础文件夹, 这里假设保存在第一个输入文件夹中的 "compare" 目录
    print(matching_images)
    base_folder = "data"
    compare_folder = os.path.join(base_folder, save_folder)
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
    
    for name, file_list in matching_images.items():
        if len(file_list) > 1:
            concatenated_image = concat_zoom_images(file_list, coordinates)
            if concatenated_image is not None:
                save_path = os.path.join(compare_folder, f'{name}')
                cv2.imwrite(save_path, concatenated_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='拼接多个文件夹中相同名字的图像文件')
    parser.add_argument('--save_folder', type=str, default="compare_with_zoom", help='需要处理的文件夹路径')
    parser.add_argument('--img_name', default='n01818515_11302.png', type=str, help='img_name for compare')
    # parser.add_argument('--img_name', default='n02980441_33851.png', type=str, help='img_name for compare')

    parser.add_argument('--folder_paths', type=str, nargs='+', 
                        default=compare_list,
                        help='需要处理的文件夹路径')

    args = parser.parse_args()
    print(args.folder_paths)
    coordinates = (215, 175, 80, 80)
    # coordinates = (110, 155, 80, 80)
    main(args.save_folder, args.folder_paths,coordinates, args.img_name)


# compare
#python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val/inputs_384 data/flatnet_output_384_val/inputs data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384
# compare_sd_input
# python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val/ddpm_200_384 data/flatnet_output_val/outputs_test_1000_384  data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384
# compare_diff
# python scripts/compare.py data/flatnet_output_384_val/gts data/flatnet_val_multi_decoded_sim_old/ddpm_200_stablesr_384 data/flatnet_val_multi_decoded_sim_old/ddnm_stablesr_384 data/flatnet_val_multi_decoded_sim_old/diffbirv1_384 data/flatnet_val_multi_decoded_sim_old/test_ddpm_200_384

# data/flatnet_val_decoded_sim/test_ddpm_200_384