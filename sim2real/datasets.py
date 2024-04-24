# dataset for paired sim and real capture
import torch 
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import torch.nn.functional as F
import time

# # totensor
transform = transforms.Compose([
		transforms.ToTensor(),
	])

class PairedCaptureDataset(data.Dataset):
    def __init__(self, sim_capture_path, real_capture_path, transform=transform, return_name=False):
        self.sim_capture_path = sim_capture_path
        self.real_capture_path = real_capture_path
        self.return_name = return_name
        sim_captures = os.listdir(sim_capture_path)
        real_captures = os.listdir(real_capture_path)
        sim_captures.sort()
        real_captures.sort()
        self.paired_captures = []
        for real_capture in real_captures:
            if real_capture not in sim_captures:
                print(real_capture)
                raise ValueError
            self.paired_captures.append((real_capture, real_capture))


        # combine sim and real capture file name with same name to a pair
  
        # self.paired_captures = []
        # for sim_capture, real_capture in zip(sim_captures, real_captures):
        #     if sim_capture.split('.')[0] == real_capture.split('.')[0]:
        #         self.paired_captures.append((sim_capture, real_capture))
        #     else:
        #         print(sim_capture, real_capture)
        #         print('sim and real capture not match')
        #         raise ValueError
     
        self.transform = transform
        # self.resize = transforms.Resize((256, 256))
        self.resize = None
    

    def padding2square(self, img):
        # Assuming `img` is a tensor with shape (C, H, W)
        _, h, w = img.shape
        # Find the max dimension
        max_dim = max(h, w)

        # Calculate padding for height and width
        # The difference between the max dimension and the actual dimension
        # is divided by 2 to pad both sides equally
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        
        pad_w1 = pad_w if pad_w % 2 == 0 else pad_w + 1
        pad_h1 = pad_h if pad_h % 2 == 0 else pad_h + 1

        # Pad the image
        # Padding format (left, right, top, bottom)
        padded_img = F.pad(img, (pad_w, pad_w1, pad_h, pad_h1), value=0)

        return padded_img

    def center_crop(self, image, new_width, new_height):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the starting point of the crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        # Perform the center crop
        cropped_image = image[start_y:start_y+new_height, start_x:start_x+new_width]
        return cropped_image

    def __getitem__(self, index):
        # print(self.paired_captures[index][0], self.paired_captures[index][1])
        start = time.time()

        sim_capture_path = os.path.join(self.sim_capture_path, self.paired_captures[index][0])
        real_capture_path = os.path.join(self.real_capture_path, self.paired_captures[index][1])
        sim_capture = cv2.imread(sim_capture_path)
        real_capture = cv2.imread(real_capture_path)
        #center crop to 1536*1536
        sim_capture = self.center_crop(sim_capture, 1536, 1280)
        real_capture = self.center_crop(real_capture, 1536, 1280)

        # rescale to 0-255
        # sim_capture = sim_capture.astype(float) - sim_capture.min()
        # sim_capture = sim_capture / sim_capture.max() * 255
        # real_capture = real_capture.astype(float) - real_capture.min()
        # real_capture = real_capture / real_capture.max() * 255
        if self.transform:
            sim_capture = self.transform(sim_capture)
            real_capture = self.transform(real_capture)
        sim_capture = self.padding2square(sim_capture).float()
        real_capture = self.padding2square(real_capture).float()
        if self.resize is not None:
            sim_capture = self.resize(sim_capture)
            real_capture = self.resize(real_capture)
        if self.return_name:
            return sim_capture, real_capture, self.paired_captures[index][1]
        else:
            return sim_capture, real_capture
      

    def __len__(self):
        return len(self.paired_captures)