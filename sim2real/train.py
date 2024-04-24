import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from sim2real import Sim2real  # Make sure sim2real.py is in the same directory or adjust the import path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import PairedCaptureDataset  # Replace with your dataset
from pytorch_lightning.loggers import TensorBoardLogger
def main(opt):
    # Set up data transforms if necessary
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5),
        # Add more transforms as needed
    ])

    # Load dataset
    dataset = PairedCaptureDataset(sim_capture_path=opt.sim_capture_path,   real_capture_path=opt.real_capture_path, transform=transform)  # Update CustomDataset with real dataset
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    

    # Set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    test_dataset = PairedCaptureDataset(sim_capture_path=opt.sim_capture_path,   real_capture_path="/root/caixin/StableSR/data/flatnet_val/sim_captures", transform=transform, return_name=False)  # Update CustomDataset with real dataset

    test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Instantiate the Sim2real model
    model = Sim2real(in_channels=opt.in_channels, out_channels=opt.out_channels, learning_rate=opt.learning_rate, visual_path = opt.visual_path)
    # Configure TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="real2sim")

    # Set up PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,  
                        accelerator='ddp',
                        logger=logger,
                        gpus=opt.gpus,
                        )
    
    trainer.fit(model, train_loader, test_data_loader)

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data path argument
    parser.add_argument('--sim_capture_path', type=str, default='/root/caixin/StableSR/data/flatnet/real_captures', help='path to sim capture')
    parser.add_argument('--real_capture_path', type=str, default='/root/caixin/StableSR/data/flatnet/sim_captures', help='path to real capture')

    # Hyperparameters
    parser.add_argument('--in_channels', type=int, default=3, help='input channel size')
    parser.add_argument('--out_channels', type=int, default=3, help='output channel size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--visual_path', type=str, default='visualization', help='path to save visualized images')

    # Training settings
    parser.add_argument('--gpus', type=int, default=8, help='number of GPUs to use')

    opt = parser.parse_args()

    main(opt)