# write a pytorch lightning module for the sim2real model
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchvision import transforms
from models import UNet
import time
from piqa import SSIM, PSNR
import os
import cv2
from torchvision.utils import save_image



class Sim2real(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, display_step=25, visual_path = 'visualization', test_path = 'test'):

        super().__init__()
        self.save_hyperparameters()
        
        self.display_step = display_step
        self.gen = UNet(in_channels, out_channels)
        # intializing weights
        self.gen.initialize_weights()
        self.recon_criterion = nn.L1Loss()
        self.visual_path  = os.path.join(self.hparams.visual_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        self.ssim_cuda = SSIM(n_channels = 3).cuda()
        self.psnr_cuda = PSNR().cuda() 
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        return opt
    
    def forward(self, x):
        return self.gen(x)

    def visualize(self, x, y, y_hat, batch_idx):
        # show only 1 image from batch
        x = x[:1]
        y = y[:1]
        y_hat = y_hat[:1]
        # concatenate images into grid
        grid = make_grid(torch.cat([x, y, y_hat], dim=0), normalize=True)
        # log image
        #transfer the image to rgb
        # print(grid.shape)
        grid = grid[[2, 1, 0], :, :]
        self.logger.experiment.add_image("images", grid, global_step=batch_idx)
        # save image
        # print(self.visual_path)
        os.makedirs(self.visual_path, exist_ok=True)
        save_path = os.path.join(self.visual_path, f"{batch_idx}.png")
        # grid

        save_image(grid, save_path)
        

    def psnr(self, origin_img, target_img, mean = 0.5, std = 0.5):
        mean = torch.tensor(mean).view(-1, 1, 1).cuda()
        std = torch.tensor(std).view(-1, 1, 1).cuda()

        origin_img = origin_img * std + mean
        target_img = target_img * std + mean
        # print(origin_img.max(), origin_img.min())
        # print(target_img.max(), target_img.min())
        # 确保值在 [0, 1] 范围内
        origin_img = torch.clamp(origin_img, 0, 1)
        target_img = torch.clamp(target_img, 0, 1)
        psnr_value = self.psnr_cuda(origin_img, target_img)
        return psnr_value

    def ssim(self, origin_img, target_img, mean = 0.5, std = 0.5):

        mean = torch.tensor(mean).view(-1, 1, 1).cuda()
        std = torch.tensor(std).view(-1, 1, 1).cuda()

        origin_img = origin_img * std + mean
        target_img = target_img * std + mean
        # print(origin_img.max(), origin_img.min())
        # print(target_img.max(), target_img.min())
        # 确保值在 [0, 1] 范围内
        origin_img = torch.clamp(origin_img, 0, 1)
        target_img = torch.clamp(target_img, 0, 1)
        ssim_value = self.ssim_cuda(origin_img, target_img)
        return ssim_value
        
    def training_step(self, batch, batch_idx):
        input, output = batch
        prediction = self(input)
        loss = self.recon_criterion(output, prediction)
        self.logger.experiment.add_scalar("loss", loss, global_step=batch_idx)
        return loss
    def validation_step(self, batch, batch_idx):
        input, output = batch
        prediction = self(input)
        loss = self.recon_criterion(output, prediction)
        self.logger.experiment.add_scalar("loss", loss, global_step=batch_idx)

        # calculate PSNR
        psnr = self.psnr(output, prediction)
        self.logger.experiment.add_scalar("psnr", psnr, global_step=batch_idx)

        ssim = self.ssim(output, prediction)
        self.logger.experiment.add_scalar("ssim", ssim, global_step=batch_idx)
        # visualize images
        # if batch_idx % self.display_step == 0:
        self.visualize(input, output, prediction, batch_idx)
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.test_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        input, output, file_name = batch
        prediction = self(input)
        loss = self.recon_criterion(output, prediction)
        self.logger.experiment.add_scalar("loss", loss, global_step=batch_idx)

        # calculate PSNR
        psnr = self.psnr(output, prediction)
        self.logger.experiment.add_scalar("psnr", psnr, global_step=batch_idx)

        # calculate SSIM
        ssim = self.ssim(output, prediction)
        self.logger.experiment.add_scalar("ssim", ssim, global_step=batch_idx)

        # visualize images
        # if batch_idx % self.display_step == 0:
        # save the predicted image to the test_path + file_name
        os.makedirs(self.hparams.test_path, exist_ok=True)

        for i in range(len(file_name)):
            save_path = os.path.join(self.hparams.test_path, file_name[i])
            # transfer the image to rgb
            prediction[i] = prediction[i][[2, 1, 0], :, :]
            save_image(prediction[i], save_path, normalize=True)
            print(f"save image to {save_path}")
        outputs = {"loss": loss, "psnr": psnr, "ssim": ssim}
        return outputs 
    # print loss and psnr
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        print(f"avg_loss={avg_loss}, avg_psnr={avg_psnr}, avg_ssim={avg_ssim}")
    

