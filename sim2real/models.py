import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, num_filters=64, ):
        super(UNet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv_down1 = nn.Conv2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv_down2 = nn.Conv2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv_down3 = nn.Conv2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv_down4 = nn.Conv2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)

        self.tconv1 = nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv_up1 = nn.ConvTranspose2d(16 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv_up2 = nn.ConvTranspose2d(16 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv_up3 = nn.ConvTranspose2d(16 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(16 * num_filters, 4 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose2d(8 * num_filters, 2 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv4 = nn.ConvTranspose2d(4 * num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.out_tconv = nn.ConvTranspose2d(2 * num_filters, num_classes, kernel_size=4, stride=2, padding=1)

        self.ins_norm_out = nn.InstanceNorm2d(num_classes)  # 添加
        self.ins_norm0 = nn.InstanceNorm2d(num_filters)
        self.ins_norm1 = nn.InstanceNorm2d(2 * num_filters)
        self.ins_norm2 = nn.InstanceNorm2d(4 * num_filters)
        self.ins_norm3 = nn.InstanceNorm2d(8 * num_filters)

        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout2d(p=0.5)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.leakyrelu(x1)

        x2 = self.conv1(x1)
        x2 = self.ins_norm1(x2)
        x2 = self.leakyrelu(x2)

        x3 = self.conv2(x2)
        x3 = self.ins_norm2(x3)
        x3 = self.leakyrelu(x3)

        x4 = self.conv3(x3)
        x4 = self.ins_norm3(x4)
        x4 = self.leakyrelu(x4)

        x5 = self.conv_down1(x4)
        x5 = self.ins_norm3(x5)
        x5 = self.leakyrelu(x5)

        x6 = self.conv_down2(x5)
        x6 = self.ins_norm3(x6)
        x6 = self.leakyrelu(x6)

        x7 = self.conv_down3(x6)
        x7 = self.ins_norm3(x7)
        x7 = self.dropout(x7)  # 添加
        x7 = self.leakyrelu(x7)

        x = self.conv_down4(x7)
        x = self.dropout(x)   # 添加
        x = self.relu(x)

        x = self.tconv1(x)
        x = self.ins_norm3(x)
        x = self.dropout(x)   # 添加
        x = torch.cat([x, x7], dim=1)
        x = self.relu(x)

        x = self.tconv_up1(x)
        x = self.ins_norm3(x)
        x = torch.cat([x, x6], dim=1)
        x = self.relu(x)

        x = self.tconv_up2(x)
        x = self.ins_norm3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.relu(x)

        x = self.tconv_up3(x)
        x = self.ins_norm3(x)
        x = torch.cat([x, x4], dim=1)
        x = self.relu(x)

        x = self.tconv2(x)
        x = self.ins_norm2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.relu(x)

        x = self.tconv3(x)
        x = self.ins_norm1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.relu(x)

        x = self.tconv4(x)
        x = self.ins_norm0(x)
        x = torch.cat([x, x1], dim=1)
        x = self.relu(x)

        x = self.out_tconv(x)
        x = self.tanh(x)

        return x
    