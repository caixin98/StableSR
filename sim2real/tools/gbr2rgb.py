# transfer the gbr image to rgb image
import cv2
import os
origin_path = '/root/caixin/StableSR/sim2real/visualization/2024-01-29-21-52-53/15.png'
target_path = '/root/caixin/StableSR/sim2real/visualization/2024-01-29-21-52-53/15_rgb.png'
origin_img = cv2.imread(origin_path)
target_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
cv2.imwrite(target_path, target_img)
