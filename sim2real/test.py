import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from sim2real import Sim2real  # Make sure sim2real.py is in the same directory or adjust the import path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets2 import PairedCaptureDataset  # Replace with your dataset
from pytorch_lightning.loggers import TensorBoardLogger
def main(opt):
    # Set up data transforms if necessary
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5),
        # Add more transforms as needed
    ])

    # Load dataset
    dataset = PairedCaptureDataset(sim_capture_path=opt.sim_capture_path,   real_capture_path=opt.real_capture_path, transform=transform, return_name=True)  # Update CustomDataset with real dataset

    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Instantiate the Sim2real model
    model = Sim2real(in_channels=opt.in_channels, out_channels=opt.out_channels, learning_rate=opt.learning_rate, visual_path = opt.visual_path, test_path = opt.test_path)
    model = model.load_from_checkpoint(opt.ckpt)
    model.hparams.test_path = opt.test_path
    model.hparams.visual_path = opt.visual_path
    model.eval()
    # Configure TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name=opt.name)

    # Set up PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,  
                        accelerator='cuda',
                        logger=logger,
                        gpus=opt.gpus,
                        # devices=1, num_nodes=1
                        )
    print('Start testing')
    print('ckpt path:', opt.ckpt)
    trainer.test(model, data_loader)
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--name', type=str, default='sim2real', help='name of the experiment. It decides where to store samples and models')

    # Add data path argument
    parser.add_argument('--sim_capture_path', type=str, default='/root/caixin/data/lfw/lfw-deepfunneled-172x172-single', help='path to sim capture')
    parser.add_argument('--real_capture_path', type=str, default='/root/caixin/data/lfw/lfw-deepfunneled-172x172-single-optical', help='path to real capture')

    # Hyperparameters
    parser.add_argument('--in_channels', type=int, default=3, help='input channel size')
    parser.add_argument('--out_channels', type=int, default=3, help='output channel size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--visual_path', type=str, default='visualization', help='path to save visualized images')
    parser.add_argument('--test_path', type=str, default='/root/caixin/data/lfw/lfw-deepfunneled-recon-single', help='path to save test prediction images')

    # Training settings
    parser.add_argument('--gpus', type=int, default=8, help='number of GPUs to use')

    # checkpoint
    parser.add_argument('--ckpt', type=str, default=None, help='resume from checkpoint')

    opt = parser.parse_args()

    main(opt)