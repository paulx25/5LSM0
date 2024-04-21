import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = encoder(3, 32)
        self.encoder_2 = encoder(32, 40)
        self.encoder_3 = encoder(40, 50)

        self.middle = convolution_block(50, 63)
        
        self.decoder_1 = decoder(63, 50)
        self.decoder_2 = decoder(50, 40)
        self.decoder_3 = decoder(40, 32)

        self.classifier = nn.Conv2d(32, 19, kernel_size=1, padding=0)

    def forward(self, inputs):
        skip_1, x1 = self.encoder_1(inputs)
        skip_2, x2 = self.encoder_2(x1)
        skip_3, x3 = self.encoder_3(x2)

        x4 = self.middle(x3)

        x5 = self.decoder_1(x4, skip_3)
        x6 = self.decoder_2(x5, skip_2)
        x7 = self.decoder_3(x6, skip_1)

        out = self.classifier(x7)

        return out


class convolution_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(output_channels)
        
        self.dropout = nn.Dropout2d(p=0.2)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.dropout(x)

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