import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = encoder(3, 32)
        self.encoder_2 = encoder(32, 48)
        self.encoder_3 = encoder(48, 72)
        self.encoder_4 = encoder(72, 108)

        self.middle = convolution_block(108, 162)
        
        self.decoder_1 = decoder(162, 108)
        self.decoder_2 = decoder(108, 72)
        self.decoder_3 = decoder(72, 48)
        self.decoder_4 = decoder(48, 32)

        self.classifier = nn.Conv2d(32, 19, kernel_size=1, padding=0)

    def forward(self, inputs):
        skip_1, x1 = self.encoder_1(inputs)
        skip_2, x2 = self.encoder_2(x1)
        skip_3, x3 = self.encoder_3(x2)
        skip_4, x4 = self.encoder_4(x3)

        x5 = self.middle(x4)

        x6 = self.decoder_1(x5, skip_4)
        x7 = self.decoder_2(x6, skip_3)
        x8 = self.decoder_3(x7, skip_2)
        x9 = self.decoder_4(x8, skip_1)

        out = self.classifier(x9)

        return out


class convolution_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(output_channels)

        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.dropout(x)
        x = self.batchnorm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.batchnorm_2(x)
        x = self.relu(x)

        return x

class encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv = convolution_block(input_channels, output_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        skip = self.conv(inputs)
        x = self.pool(skip)

        return skip, x

class decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.up_pool = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0)
        self.conv = convolution_block(2*output_channels, output_channels)

    def forward(self, inputs, skip):
        x = self.up_pool(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x